# contains how to load and parse all files into a uniform format described in the schema

import fitz
import json
import os
from bs4 import BeautifulSoup
from markdown import markdown
import re
import tempfile
from supabase_client import SupabaseClient
import logging

logger = logging.getLogger(__name__)

def ingest_file(file_path):
    if file_path.endswith(".pdf"):
        return ingest_pdf(file_path)
    elif file_path.endswith(".md"):
        return ingest_markdown(file_path)
    elif file_path.endswith(".html"):
        return ingest_html(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def clean_text(text):
    # Remove invisible and problematic unicode characters, but preserve \n, \r, and \t
    return re.sub(r'[\u200b\u200c\u200d\ufeff\xa0\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

def pdf_to_text(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return clean_text(text).strip()

def markdown_to_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    html_ver = markdown(text)
    return ''.join(BeautifulSoup(html_ver, features="html.parser").findAll(text=True))

def html_to_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return ''.join(BeautifulSoup(text, features="html.parser").findAll(text=True))

def ingest_pdf(file_path):
    dictionaryReturn = {}
    dictionaryReturn["text"] = pdf_to_text(file_path)
    dictionaryReturn["source"] = file_path
    dictionaryReturn["file_type"] = "pdf"
    return save_ingested_json(json.dumps(dictionaryReturn), file_path)

def ingest_markdown(file_path):
    dictionaryReturn = {}
    dictionaryReturn["text"] = markdown_to_text(file_path)
    dictionaryReturn["source"] = file_path
    dictionaryReturn["file_type"] = "md"
    return save_ingested_json(json.dumps(dictionaryReturn), file_path)

def ingest_html(file_path):
    dictionaryReturn = {}
    dictionaryReturn["text"] = html_to_text(file_path)
    dictionaryReturn["source"] = file_path
    dictionaryReturn["file_type"] = "html"
    return save_ingested_json(json.dumps(dictionaryReturn), file_path)

def save_ingested_json(ingested_json, original_file_path, user_id):
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
        json_filename = os.path.splitext(base_name)[0] + ".json"
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

def ingest_all_files(user_id):
    """
    Ingest all files from a user's Supabase storage.
    Args:
        user_id: The user ID to fetch files for
    Returns:
        List of paths to ingested JSON files in Supabase
    """
    # Initialize Supabase client
    supabase = SupabaseClient()
    
    try:
        # Get all files for the user
        files = supabase.list_files(user_id)
        if not files:
            raise ValueError("No files found for user")
        
        processed_paths = []
        errors = []
        
        for file_info in files:
            try:
                file_path = file_info['name']
                file_ext = os.path.splitext(file_path)[1].lower()
                
                # Download file from Supabase
                file_data = supabase.download_file(file_path, user_id)
                if not file_data:
                    raise ValueError(f"Failed to download file: {file_path}")
                
                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                    temp_file.write(file_data)
                    temp_path = temp_file.name
                
                try:
                    # Process the file based on its type
                    if file_ext == '.pdf':
                        result = ingest_pdf(temp_path)
                    elif file_ext == '.md':
                        result = ingest_markdown(temp_path)
                    elif file_ext == '.html':
                        result = ingest_html(temp_path)
                    else:
                        raise ValueError(f"Unsupported file type: {file_ext}")
                    
                    # Add Supabase path to the result
                    result_data = json.loads(result)
                    result_data["supabase_path"] = file_path
                    result = json.dumps(result_data)
                    
                    # Save the processed file to Supabase
                    processed_path = save_ingested_json(result, file_path, user_id)
                    processed_paths.append(processed_path)
                    
                finally:
                    # Clean up temporary file
                    os.unlink(temp_path)
                    
            except Exception as e:
                error_msg = f"Error processing {file_path}: {str(e)}"
                errors.append(error_msg)
        
        if errors:
            raise Exception(f"Errors occurred during ingestion: {', '.join(errors)}")
            
        return processed_paths
            
    except Exception as e:
        raise Exception(f"Error ingesting files from Supabase: {str(e)}")