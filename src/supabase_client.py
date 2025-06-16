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
        
        # Create storage bucket if it doesn't exist
        self._ensure_storage_bucket()
    
    def _ensure_storage_bucket(self):
        """Ensure the documents bucket exists in Supabase storage."""
        try:
            # First try to get the bucket
            self.supabase.storage.get_bucket('documents')
            logger.info("Documents bucket already exists")
        except Exception as e:
            logger.warning(f"Bucket not found or error accessing it: {str(e)}")
            try:
                # Try to create the bucket with minimal options
                logger.info("Attempting to create documents bucket...")
                result = self.supabase.storage.create_bucket(
                    id='documents',
                    options={'public': False}
                )
                logger.info(f"Bucket creation result: {result}")
                logger.info("Successfully created documents bucket")
            except Exception as create_error:
                logger.error(f"Failed to create bucket. Error details: {str(create_error)}")
                logger.error(f"Error type: {type(create_error)}")
                if hasattr(create_error, 'response'):
                    logger.error(f"Response: {create_error.response}")
                raise  # Re-raise the error since we need the bucket
    
    def _get_user_path(self, user_id: str, file_name: str) -> str:
        """Get the storage path for a user's file."""
        return f"users/{user_id}/{file_name}"
    
    async def upload_file(self, file_path: str, file_name: str, user_id: str = None) -> Dict[str, Any]:
        """Upload a file to Supabase storage."""
        try:
            logger.info(f"Reading file from: {file_path}")
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # Generate user_id if not provided
            if not user_id:
                user_id = str(uuid.uuid4())
            
            # Create user-specific path
            storage_path = self._get_user_path(user_id, file_name)
            logger.info(f"Uploading to storage path: {storage_path}")
            
            # Upload to Supabase storage
            logger.info("Attempting Supabase upload...")
            result = self.supabase.storage.from_('documents').upload(
                storage_path,
                file_data,
                {'content-type': self._get_content_type(file_name)}
            )
            logger.info(f"Upload result: {result}")
            
            # Get the public URL
            url = self.supabase.storage.from_('documents').get_public_url(storage_path)
            logger.info(f"Generated URL: {url}")
            
            return {
                'success': True,
                'url': url,
                'path': result.path,
                'user_id': user_id
            }
        except Exception as e:
            logger.error(f"Error in upload_file: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_content_type(self, file_name: str) -> str:
        """Get the content type based on file extension."""
        ext = file_name.lower().split('.')[-1]
        content_types = {
            'pdf': 'application/pdf',
            'html': 'text/html',
            'md': 'text/markdown'
        }
        return content_types.get(ext, 'application/octet-stream')
    
    def list_files(self, user_id: str = None) -> List[Dict[str, Any]]:
        """List all files in the documents bucket, optionally filtered by user."""
        try:
            if user_id:
                # List files for specific user
                path = f"users/{user_id}"
                result = self.supabase.storage.from_('documents').list(path)
            else:
                # List all files
                result = self.supabase.storage.from_('documents').list()
            return result
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []
    
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
    
    def download_file(self, file_name: str, user_id: str) -> bytes:
        """Download a file from Supabase storage."""
        try:
            storage_path = self._get_user_path(user_id, file_name)
            return self.supabase.storage.from_('documents').download(storage_path)
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            return None 