import json
import os
from supabase import create_client, Client
from typing import List, Dict, Any

class SupabaseClient:
    def __init__(self):
        # Load API keys
        with open('APIKeys.json', 'r') as f:
            keys = json.load(f)
        
        self.supabase: Client = create_client(
            keys['supabaseURL'],
            keys['supabaseKey']
        )
        
        # Create storage bucket if it doesn't exist
        self._ensure_storage_bucket()
    
    def _ensure_storage_bucket(self):
        """Ensure the documents bucket exists in Supabase storage."""
        try:
            self.supabase.storage.get_bucket('documents')
        except Exception:
            # Create bucket if it doesn't exist
            self.supabase.storage.create_bucket('documents', {'public': False})
    
    async def upload_file(self, file_path: str, file_name: str) -> Dict[str, Any]:
        """Upload a file to Supabase storage."""
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # Upload to Supabase storage
            result = self.supabase.storage.from_('documents').upload(
                file_name,
                file_data,
                {'content-type': self._get_content_type(file_name)}
            )
            
            # Get the public URL
            url = self.supabase.storage.from_('documents').get_public_url(file_name)
            
            return {
                'success': True,
                'url': url,
                'path': result.path
            }
        except Exception as e:
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
    
    def list_files(self) -> List[Dict[str, Any]]:
        """List all files in the documents bucket."""
        try:
            result = self.supabase.storage.from_('documents').list()
            return result
        except Exception as e:
            print(f"Error listing files: {e}")
            return []
    
    def delete_file(self, file_name: str) -> bool:
        """Delete a file from Supabase storage."""
        try:
            self.supabase.storage.from_('documents').remove([file_name])
            return True
        except Exception as e:
            print(f"Error deleting file: {e}")
            return False
    
    def clear_all_files(self) -> bool:
        """Delete all files from the documents bucket."""
        try:
            files = self.list_files()
            if files:
                self.supabase.storage.from_('documents').remove([f['name'] for f in files])
            return True
        except Exception as e:
            print(f"Error clearing files: {e}")
            return False 