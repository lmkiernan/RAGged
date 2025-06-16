import fitz
import json
import os
from bs4 import BeautifulSoup
from markdown import markdown
import re
import tempfile
from src.supabase_client import SupabaseClient
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_file(file_path: str, user_id: str) -> str:
    """
    Ingest a single file based on its extension and save the result to Supabase.
    Returns the storage path of the processed JSON file.
    """
    if file_path.endswith(".pdf"):
        return ingest_pdf(file_path, user_id)
    elif file_path.endswith(".md"):
        return ingest_markdown(file_path, user_id)
    elif file_path.endswith(".html"):
        return ingest_html(file_path, user_id)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


def clean_text(text: str) -> str:
    """Remove invisible or problematic unicode chars, preserve newlines and tabs."""
    return re.sub(r'[\u200b\u200c\u200d\ufeff\xa0\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\u009f]', '', text)


def pdf_to_text(file_path: str) -> str:
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return clean_text(text).strip()


def markdown_to_text(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    html_ver = markdown(text)
    return ''.join(BeautifulSoup(html_ver, 'html.parser').find_all(text=True))


def html_to_text(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    return ''.join(BeautifulSoup(html_content, 'html.parser').find_all(text=True))


def ingest_pdf(file_path: str, user_id: str, original_file_path: str) -> str:
    """
    Process a PDF file and save the result.
    Args:
        file_path: Path to the temporary file to process
        user_id: User ID for storage
        original_file_path: Original file path from Supabase
    """
    data = {
        "text": pdf_to_text(file_path),
        "source": original_file_path,
        "file_type": "pdf"
    }
    ingested_json = json.dumps(data, ensure_ascii=False)
    return save_ingested_json(ingested_json, original_file_path, user_id)


def ingest_markdown(file_path: str, user_id: str, original_file_path: str) -> str:
    """
    Process a Markdown file and save the result.
    Args:
        file_path: Path to the temporary file to process
        user_id: User ID for storage
        original_file_path: Original file path from Supabase
    """
    data = {
        "text": markdown_to_text(file_path),
        "source": original_file_path,
        "file_type": "md"
    }
    ingested_json = json.dumps(data, ensure_ascii=False)
    return save_ingested_json(ingested_json, original_file_path, user_id)


def ingest_html(file_path: str, user_id: str, original_file_path: str) -> str:
    """
    Process an HTML file and save the result.
    Args:
        file_path: Path to the temporary file to process
        user_id: User ID for storage
        original_file_path: Original file path from Supabase
    """
    data = {
        "text": html_to_text(file_path),
        "source": original_file_path,
        "file_type": "html"
    }
    ingested_json = json.dumps(data, ensure_ascii=False)
    return save_ingested_json(ingested_json, original_file_path, user_id)


def save_ingested_json(ingested_json: str, original_file_path: str, user_id: str) -> str:
    """
    Save the ingested JSON to Supabase storage.
    Args:
        ingested_json: The JSON string to save
        original_file_path: The original file path (used to generate the new path)
        user_id: The user ID to associate with the file
    Returns:
        The path where the file was saved in Supabase
    """
    try:
        # Initialize Supabase client
        supabase = SupabaseClient()
        
        # Generate the path for the processed file
        base_name = os.path.basename(original_file_path)
        # Replace the last period with an underscore
        json_filename = base_name.rsplit('.', 1)[0] + '_' + base_name.rsplit('.', 1)[1] + '.json'
        storage_path = f"processed/{user_id}/{json_filename}"
        
        # Upload to Supabase
        result = supabase.supabase.storage.from_('documents').upload(
            storage_path,
            ingested_json.encode('utf-8'),
            {'content-type': 'application/json'}
        )
        
        logger.info(f"Saved processed file to Supabase: {storage_path}")
        return storage_path
        
    except Exception as e:
        logger.error(f"Error saving processed file to Supabase: {str(e)}")
        raise


def ingest_all_files(user_id: str) -> list:
    """
    Ingest all user files from the 'documents' bucket in Supabase storage.
    Returns a list of processed JSON file paths.
    """
    supabase = SupabaseClient()
    processed_paths = []
    errors = []

    try:
        logger.info(f"Fetching files for user_id: {user_id}")
        # Look in the users directory for original files
        files = supabase.list_files(user_id, prefix="users/")
        logger.info(f"Found {len(files) if files else 0} files in Supabase storage")
        
        if not files:
            logger.warning(f"No files found in Supabase storage for user_id: {user_id}")
            raise ValueError("No files found for user")

        for file_info in files:
            original_file_path = file_info.get('name')
            if not original_file_path:
                logger.error("File info missing 'name' field")
                continue
                
            logger.info(f"Processing file: {original_file_path}")
            file_ext = os.path.splitext(original_file_path)[1].lower()
            
            if file_ext not in ['.pdf', '.md', '.html']:
                logger.warning(f"Skipping unsupported file type: {file_ext}")
                continue
                
            try:
                # Download the file data
                logger.info(f"Downloading file: {original_file_path}")
                file_data = supabase.download_file(original_file_path, user_id)
                if not file_data:
                    raise ValueError(f"Failed to download file: {original_file_path}")

                # Write to a temporary local file
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                    temp_file.write(file_data)
                    temp_path = temp_file.name
                    logger.info(f"Created temporary file: {temp_path}")

                try:
                    # Process based on file extension
                    logger.info(f"Processing file with extension: {file_ext}")
                    if file_ext == '.pdf':
                        processed_path = ingest_pdf(temp_path, user_id, original_file_path)
                    elif file_ext == '.md':
                        processed_path = ingest_markdown(temp_path, user_id, original_file_path)
                    elif file_ext == '.html':
                        processed_path = ingest_html(temp_path, user_id, original_file_path)
                    else:
                        raise ValueError(f"Unsupported file type: {file_ext}")

                    processed_paths.append(processed_path)
                    logger.info(f"Successfully processed file: {original_file_path} -> {processed_path}")
                finally:
                    # Always clean up the temp file
                    try:
                        os.unlink(temp_path)
                        logger.info(f"Cleaned up temporary file: {temp_path}")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to clean up temporary file: {cleanup_error}")

            except Exception as e:
                error_msg = f"Error processing {original_file_path}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

        if errors:
            error_summary = "\n".join(errors)
            logger.error(f"Errors occurred during ingestion:\n{error_summary}")
            raise Exception(f"Errors occurred during ingestion:\n{error_summary}")

        if not processed_paths:
            raise ValueError("No files were successfully processed")

        return processed_paths

    except Exception as e:
        logger.error(f"Error ingesting files from Supabase: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    # Example usage
    test_user_id = "example-user-id"
    processed = ingest_all_files(test_user_id)
    print("Processed files:", processed)
