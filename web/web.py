from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure upload settings
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
ALLOWED_EXTENSIONS = {'pdf', 'md', 'html'}
MAX_FILES = 5

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        if 'files[]' not in request.files:
            logger.error("No files in request")
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files[]')
        logger.debug(f"Received {len(files)} files")
        
        if len(files) > MAX_FILES:
            return jsonify({'error': f'Maximum {MAX_FILES} files allowed'}), 400
        
        uploaded_files = []
        errors = []
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(file_path)
                uploaded_files.append(filename)
                logger.debug(f"Successfully saved {filename}")
            else:
                error_msg = f"Invalid file type: {file.filename}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        response = {
            'success': len(uploaded_files),
            'uploaded_files': uploaded_files,
            'errors': errors
        }
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/check-files', methods=['GET'])
def check_files():
    try:
        files = [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]
        return jsonify({'has_files': len(files) > 0})
    except Exception as e:
        logger.error(f"Error checking files: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
