from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import logging
import sys
import asyncio
import uuid
import subprocess

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ingest import ingest_all_files
from src.supabase_client import SupabaseClient

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.secret_key = os.urandom(24)  # Required for session

# Configure logging to show in console
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize Supabase client
supabase_client = SupabaseClient()

# Configure upload settings
ALLOWED_EXTENSIONS = {'pdf', 'md', 'html'}
MAX_FILES = 5

def get_user_id():
    """Get or create a user ID for the current session."""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return session['user_id']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
async def upload_files():
    logger.info("Received upload request")
    try:
        if 'files' not in request.files:
            logger.error("No files in request")
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        logger.info(f"Received {len(files)} files")
        
        if len(files) > MAX_FILES:
            logger.error(f"Too many files: {len(files)} > {MAX_FILES}")
            return jsonify({'error': f'Maximum {MAX_FILES} files allowed'}), 400
        
        user_id = get_user_id()
        logger.info(f"Using user_id: {user_id}")
        uploaded_files = []
        errors = []
        
        for file in files:
            logger.debug(f"Processing file: {file.filename}")
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                # Save temporarily
                temp_path = os.path.join('/tmp', filename)
                logger.debug(f"Saving temporary file to: {temp_path}")
                try:
                    file.save(temp_path)
                    logger.debug(f"Successfully saved temporary file")
                except Exception as save_error:
                    logger.error(f"Error saving temporary file: {str(save_error)}")
                    errors.append(f"Failed to save {filename}: {str(save_error)}")
                    continue
                
                # Upload to Supabase
                logger.debug(f"Attempting to upload to Supabase: {filename}")
                try:
                    result = await supabase_client.upload_file(temp_path, filename, user_id)
                    logger.debug(f"Supabase upload result: {result}")
                    
                    # Clean up temp file
                    try:
                        os.remove(temp_path)
                        logger.debug(f"Successfully removed temporary file")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to remove temporary file: {str(cleanup_error)}")
                    
                    if result['success']:
                        uploaded_files.append(filename)
                        logger.info(f"Successfully uploaded {filename} to Supabase")
                    else:
                        error_msg = f"Failed to upload to Supabase: {result.get('error', 'Unknown error')}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                except Exception as upload_error:
                    logger.error(f"Error during Supabase upload: {str(upload_error)}")
                    errors.append(f"Failed to upload {filename}: {str(upload_error)}")
            else:
                error_msg = f"Invalid file type: {file.filename}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        response = {
            'success': len(uploaded_files),
            'uploaded_files': uploaded_files,
            'errors': errors,
            'fileCount': len(uploaded_files)
        }
        
        logger.info(f"Upload complete. Success: {len(uploaded_files)}, Errors: {len(errors)}")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/check-files', methods=['GET'])
def check_files():
    try:
        user_id = get_user_id()
        files = supabase_client.list_files(user_id)
        logger.info(f"Found {len(files)} files in Supabase storage for user {user_id}")
        return jsonify({
            'hasFiles': len(files) > 0,
            'fileCount': len(files),
            'files': [f['name'] for f in files]
        })
    except Exception as e:
        logger.error(f"Error checking files: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/clear-files', methods=['POST'])
def clear_files():
    try:
        user_id = get_user_id()
        success = supabase_client.clear_all_files(user_id)
        if success:
            logger.info(f"Successfully cleared all files from Supabase storage for user {user_id}")
            return jsonify({
                'success': True,
                'message': 'All files cleared',
                'fileCount': 0
            })
        else:
            return jsonify({'error': 'Failed to clear files'}), 500
    except Exception as e:
        logger.error(f"Error clearing files: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/process-documents', methods=['POST'])
async def process_documents():
    try:
        user_id = get_user_id()
        logger.info(f"Processing documents for user: {user_id}")
        
        # First check if there are any files to process
        try:
            files = supabase_client.list_files(user_id)
            if not files:
                logger.warning(f"No files found for user {user_id}")
                return jsonify({
                    'error': 'No files found to process',
                    'details': 'Please upload files first'
                }), 400
            
            logger.info(f"Found {len(files)} files to process")
        except Exception as list_error:
            logger.error(f"Error listing files: {str(list_error)}", exc_info=True)
            return jsonify({
                'error': 'Error checking files',
                'details': str(list_error)
            }), 500
        
        # Step 1: Ingest all files
        try:
            logger.info("Step 1: Ingesting files...")
            ingested_paths = await asyncio.to_thread(ingest_all_files, user_id)
            logger.info(f"Successfully processed {len(ingested_paths)} files")
        except Exception as ingest_error:
            logger.error(f"Error during ingestion: {str(ingest_error)}", exc_info=True)
            return jsonify({
                'error': 'Error during ingestion',
                'details': str(ingest_error)
            }), 500

        # Step 2: Generate QA pairs
        try:
            logger.info("Step 2: Generating QA pairs...")
            querier_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'querier.py')
            result = subprocess.run(
                ['python3', querier_script, '--user-id', user_id, '--num-questions', '5'],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise Exception(f"Error generating QA pairs: {result.stderr}")
                
            logger.info("Successfully generated QA pairs")
            
            # Return success response after QA generation
            return jsonify({
                'success': True,
                'message': 'Successfully processed documents and generated QA pairs',
                'details': {
                    'ingested_files': len(ingested_paths),
                    'user_id': user_id
                }
            })
            
        except Exception as qa_error:
            logger.error(f"Error during QA generation: {str(qa_error)}", exc_info=True)
            return jsonify({
                'error': 'Error during QA generation',
                'details': str(qa_error)
            }), 500

        # Commenting out all steps after QA generation
        """
        # Step 3: Run chunking for each strategy and model from config
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'default.yaml')
        try:
            with open(config_path, 'r') as f:
                import yaml
                config = yaml.safe_load(f)
        except Exception as config_error:
            logger.error(f"Error loading config file: {str(config_error)}", exc_info=True)
            return jsonify({
                'error': 'Error loading configuration',
                'details': str(config_error)
            }), 500

        # Get strategies and models from config
        strategies = config.get('strats', [])
        models = []
        for emb in config.get('embedding', []):
            provider = emb.get('provider')
            if provider == 'huggingface':
                for model in config.get('huggingface', []):
                    models.append({
                        'name': model.get('model'),
                        'provider': 'huggingface'
                    })
            elif provider == 'openai':
                for model in config.get('openai', []):
                    models.append({
                        'name': model.get('model'),
                        'provider': 'openai'
                    })

        if not strategies or not models:
            logger.error("No strategies or models found in config")
            return jsonify({
                'error': 'Invalid configuration',
                'details': 'No strategies or models found in config file'
            }), 500

        chunking_results = []
        
        for strategy in strategies:
            for model in models:
                try:
                    logger.info(f"Step 2: Running chunking with strategy: {strategy} and model: {model['name']}")
                    chunking_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'run_chunking.py')
                    
                    result = subprocess.run(
                        [
                            'python3', chunking_script,
                            '--strategy', strategy,
                            '--model', model['name'],
                            '--provider', model['provider'],
                            '--user-id', user_id,
                            '--config', config_path
                        ],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode == 0:
                        logger.info(f"Successfully completed chunking with {strategy} strategy and {model['name']} model")
                        chunking_results.append(f"{strategy}_{model['name']}")
                    else:
                        logger.error(f"Error during chunking with {strategy} strategy and {model['name']} model: {result.stderr}")
                        
                except Exception as chunking_error:
                    logger.error(f"Error running chunking script for {strategy} and {model['name']}: {str(chunking_error)}", exc_info=True)
                    continue

        if not chunking_results:
            return jsonify({
                'error': 'No chunking strategies completed successfully',
                'details': 'Failed to process documents'
            }), 500

        # Step 4: Run embeddings
        try:
            logger.info("Step 4: Running embeddings...")
            embeddings_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'run_embeddings.py')
            result = subprocess.run(
                ['python3', embeddings_script, '--user-id', user_id],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise Exception(f"Error running embeddings: {result.stderr}")
                
            logger.info("Successfully completed embeddings")
            
        except Exception as embedding_error:
            logger.error(f"Error during embeddings: {str(embedding_error)}", exc_info=True)
            return jsonify({
                'error': 'Error during embeddings',
                'details': str(embedding_error)
            }), 500

        # Step 5: Evaluate retrieval
        try:
            logger.info("Step 5: Evaluating retrieval performance...")
            evaluation_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'evaluate_retrieval.py')
            result = subprocess.run(
                ['python3', evaluation_script, '--user-id', user_id],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise Exception(f"Error during evaluation: {result.stderr}")
                
            logger.info("Successfully completed evaluation")
            
        except Exception as eval_error:
            logger.error(f"Error during evaluation: {str(eval_error)}", exc_info=True)
            return jsonify({
                'error': 'Error during evaluation',
                'details': str(eval_error)
            }), 500

        # Return success response
        return jsonify({
            'success': True,
            'message': 'Successfully processed all documents',
            'details': {
                'ingested_files': len(ingested_paths),
                'chunking_results': chunking_results,
                'user_id': user_id
            }
        })
        """
        
    except Exception as e:
        logger.error(f"Error in process_documents: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/')
def index():
    return send_from_directory(os.path.dirname(os.path.dirname(__file__)), 'index.html')

@app.route('/nextscreen')
def nextscreen():
    return send_from_directory(os.path.dirname(__file__), 'nextscreen.html')

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)
