from dotenv import load_dotenv
import os
import requests
from datetime import datetime, timezone
import jwt
from label_studio_sdk import Client

def test_direct_api():
    """Test direct API access with token refresh"""
    print('\n1. Testing Direct API Access:')
    try:
        # Get refresh token from env
        refresh_token = os.getenv('LABEL_STUDIO_API_KEY')
        base_url = os.getenv('LABEL_STUDIO_URL')
        
        if not refresh_token or not base_url:
            print('❌ Error: Missing environment variables')
            print(f'URL: {base_url}')
            print(f'Token: {"Found" if refresh_token else "Missing"}')
            return
        
        # Get access token
        print('Getting access token...')
        response = requests.post(
            f'{base_url}/api/token/refresh',
            json={'refresh': refresh_token},
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code != 200:
            print('❌ Failed to get access token:')
            print(response.text)
            return
            
        access_token = response.json()['access']
        print('✅ Got access token')
        
        # Test projects endpoint with access token
        print('\nTesting projects endpoint...')
        headers = {'Authorization': f'Bearer {access_token}'}
        response = requests.get(f'{base_url}/api/projects', headers=headers)
        
        if response.status_code != 200:
            print('❌ Failed to get projects:')
            print(response.text)
            return
            
        projects = response.json()
        print('✅ Successfully retrieved projects')
        print(f'Found {projects["count"]} projects:')
        for project in projects['results']:
            print(f'- {project["title"]} (ID: {project["id"]})')
            
    except Exception as e:
        print('❌ Error during direct API test:', str(e))

def test_sdk():
    """Test Label Studio SDK functionality"""
    print('\n2. Testing SDK Integration:')
    try:
        # Get credentials from env
        url = os.getenv('LABEL_STUDIO_URL')
        api_key = os.getenv('LABEL_STUDIO_API_KEY')
        
        if not api_key or not url:
            print('❌ Error: Missing environment variables')
            return
        
        print(f'Connecting to {url}...')
        client = Client(url=url, api_key=api_key)
        
        # Test projects
        print('Getting projects...')
        projects = client.list_projects()
        print('✅ Successfully connected')
        print(f'Found {len(projects)} projects:') # pyright: ignore[reportArgumentType]
        for project in projects: # type: ignore
            print(f'- {project.title} (ID: {project.id})')
            
    except Exception as e:
        print('❌ Error during SDK test:', str(e))

def test_token_info():
    """Display information about the current token"""
    print('\n3. Token Information:')
    try:
        token = os.getenv('LABEL_STUDIO_API_KEY')
        if not token:
            print('❌ No token found in environment')
            return
            
        # Decode token without verification
        decoded = jwt.decode(token, options={"verify_signature": False})
        
        print('Token type:', decoded.get('token_type', 'unknown'))
        
        exp = decoded.get('exp')
        if exp:
            exp_dt = datetime.fromtimestamp(exp, tz=timezone.utc)
            now = datetime.now(timezone.utc)
            if exp_dt > now:
                print('Status: Valid until', exp_dt.isoformat())
            else:
                print('Status: Expired on', exp_dt.isoformat())
        else:
            print('Status: No expiration found')
            
        print('User ID:', decoded.get('user_id', 'unknown'))
        
    except Exception as e:
        print('❌ Error analyzing token:', str(e))

if __name__ == '__main__':
    print('🔍 Label Studio Integration Tests')
    print('================================')
    
    # Load environment variables
    load_dotenv()
    
    # Run tests
    test_token_info()
    test_direct_api()
    test_sdk()