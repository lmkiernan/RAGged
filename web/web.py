from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import logging
import sys

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging to show in console
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configure upload settings
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
ALLOWED_EXTENSIONS = {'pdf', 'md', 'html'}
MAX_FILES = 5

logger.info(f"Upload folder set to: {UPLOAD_FOLDER}")

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_files():
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
        
        uploaded_files = []
        errors = []
        
        for file in files:
            logger.debug(f"Processing file: {file.filename}")
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                logger.debug(f"Saving file to: {file_path}")
                file.save(file_path)
                uploaded_files.append(filename)
                logger.info(f"Successfully saved {filename}")
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
        files = [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]
        logger.info(f"Found {len(files)} files in {UPLOAD_FOLDER}")
        return jsonify({
            'hasFiles': len(files) > 0,
            'fileCount': len(files),
            'files': files
        })
    except Exception as e:
        logger.error(f"Error checking files: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/clear-files', methods=['POST'])
def clear_files():
    try:
        files = [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]
        for file in files:
            file_path = os.path.join(UPLOAD_FOLDER, file)
            os.remove(file_path)
            logger.info(f"Removed file: {file}")
        
        logger.info(f"Cleared {len(files)} files from {UPLOAD_FOLDER}")
        return jsonify({
            'success': True,
            'message': f'Cleared {len(files)} files',
            'fileCount': 0
        })
    except Exception as e:
        logger.error(f"Error clearing files: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return send_from_directory(os.path.dirname(os.path.dirname(__file__)), 'index.html')

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(debug=True, port=5001)
