import json
import os
from supabase import create_client, Client
from typing import List, Dict, Any
import uuid
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class SupabaseClient:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Get Supabase credentials from environment variables
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            raise ValueError("Missing required environment variables: SUPABASE_URL and/or SUPABASE_KEY")
        
        # Ensure URL is properly formatted
        if not supabase_url.startswith('https://'):
            supabase_url = f'https://{supabase_url}'
        
        # Remove any trailing slashes
        supabase_url = supabase_url.rstrip('/')
        
        logger.info(f"Initializing Supabase client with URL: {supabase_url}")
        
        try:
            self.supabase: Client = create_client(
                supabase_url,
                supabase_key
            )
            logger.info("Successfully created Supabase client")
        except Exception as e:
            logger.error(f"Failed to create Supabase client: {str(e)}")
            raise
    
    def _get_user_path(self, user_id: str, file_name: str) -> str:
        """Get the storage path for a user's file."""
        return f"users/{user_id}/{file_name}"
    
    async def upload_file(self, file_path: str, filename: str, user_id: str) -> dict:
        """Upload a file to Supabase storage."""
        try:
            logger.info(f"Uploading file {filename} for user {user_id}")
            storage_path = f"users/{user_id}/{filename}"
            logger.info(f"Using storage path: {storage_path}")
            
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            result = self.supabase.storage.from_('documents').upload(
                storage_path,
                file_data,
                {'content-type': self._get_content_type(filename)}
            )
            
            logger.info(f"Successfully uploaded file to {storage_path}")
            return {'success': True, 'path': storage_path}
        except Exception as e:
            logger.error(f"Error uploading file {filename}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_content_type(self, file_name: str) -> str:
        """Get the content type based on file extension."""
        ext = file_name.lower().split('.')[-1]
        content_types = {
            'pdf': 'application/pdf',
            'html': 'text/html',
            'md': 'text/markdown'
        }
        return content_types.get(ext, 'application/octet-stream')
    
    def list_files(self, user_id: str, prefix: str) -> list:
        """List files for a user in the 'documents' bucket.
        
        Args:
            user_id: The ID of the user whose files to list
            prefix: Prefix to filter files (e.g., 'users/', 'processed/', 'chunks/', 'qa_pairs/')
        
        Returns:
            List of file information dictionaries
        """
        try:
            logger.info(f"Listing files for user_id: {user_id} with prefix: {prefix}")
            
            # Construct the path using the provided prefix
            path = f"{prefix}{user_id}/"
            logger.info(f"Using storage path: {path}")
            
            # List files in the specified directory
            response = self.supabase.storage.from_('documents').list(path)
            logger.info(f"Found {len(response) if response else 0} files in {path}")
            
            # Log the actual files found
            if response:
                for file in response:
                    logger.debug(f"Found file: {file['name']}")
                    
            return response
            
        except Exception as e:
            logger.error(f"Error listing files for user {user_id}: {e}")
            raise
    
    def delete_file(self, file_name: str, user_id: str) -> bool:
        """Delete a file from Supabase storage."""
        try:
            storage_path = self._get_user_path(user_id, file_name)
            self.supabase.storage.from_('documents').remove([storage_path])
            return True
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return False
    
    def clear_all_files(self, user_id: str = None) -> bool:
        """Delete all files from the documents bucket, optionally for a specific user."""
        try:
            if user_id:
                # Clear only user's files
                path = f"users/{user_id}"
                files = self.supabase.storage.from_('documents').list(path)
                if files:
                    self.supabase.storage.from_('documents').remove([f['name'] for f in files])
            else:
                # Clear all files
                files = self.list_files()
                if files:
                    self.supabase.storage.from_('documents').remove([f['name'] for f in files])
            return True
        except Exception as e:
            logger.error(f"Error clearing files: {e}")
            return False
    
    def download_file(self, file_name: str, user_id: str, prefix: str) -> bytes:
        """Download a file from Supabase storage."""
        try:
            # Construct the path using the provided prefix
            storage_path = f"{prefix}{user_id}/{file_name}"
            logger.info(f"Downloading file from: {storage_path}")
            
            try:
                return self.supabase.storage.from_('documents').download(storage_path)
            except Exception as e:
                logger.error(f"Error downloading file: {e}")
                return None
            
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            return None
    
    def get_json_field(self, file_name: str, user_id: str, prefix: str, field: str) -> str:
        try:
            # 1) Download the file from storage
            download_resp = self.download_file(file_name, user_id, prefix)

        # Supabase Python client sometimes returns (data, error)
            if isinstance(download_resp, tuple):
                data, error = download_resp
                if error:
                    raise RuntimeError(f"Storage download error for {file_name}: {error}")
            else:
                data = download_resp

        # 2) Turn it into text
            if isinstance(data, (bytes, bytearray)):
                text = data.decode("utf-8")
            elif hasattr(data, "text"):
                text = data.text
            else:
                text = data.read().decode("utf-8")

        # 3) Parse JSON
            obj = json.loads(text)

        # 4) Drill down into nested keys if needed
            value = obj
            for part in field.split("."):
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    value = None
                if value is None:
                    break

            return value

        except Exception as e:
            logger.error(f"Error fetching field '{field}' from {file_name}: {e}")
            raise

    async def upload_json(
    self,
    file,
    fname: str,
    user_id: str,
    prefix: str
) -> dict:
        try:
        # 1) Build the object path in your bucket
            storage_path = f"{prefix}/{user_id}/{fname}"
            logger.info(f"Uploading JSON to documents/{storage_path}")

        # 2) Serialize to bytes
            json_text = json.dumps(file)
            json_bytes = json_text.encode("utf-8")

        # 3) Upload to Supabase Storage
            result, error = await self.supabase.storage.from_("documents").upload(
                storage_path,
                json_bytes,
                {"content-type": "application/json"}
            )

            if error:
                logger.error(f"Upload error for {storage_path}: {error}")
                return {"success": False, "error": str(error)}

            logger.info(f"Successfully uploaded JSON to {storage_path}")
            return {"success": True, "path": storage_path}

        except Exception as e:
            logger.error(f"Exception uploading JSON file {fname}: {e}")
            return {"success": False, "error": str(e)}