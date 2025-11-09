from dotenv import load_dotenv
import os
import requests
from datetime import datetime, timezone
import jwt
import json
import getpass
from pathlib import Path
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

        # Print token info
        print('\nCurrent token details:')
        try:
            decoded = jwt.decode(refresh_token, options={"verify_signature": False})
            print(json.dumps(decoded, indent=2))
        except Exception as e:
            print('Failed to decode token:', str(e))
        
        # Try both authentication methods
        print('\nTrying different auth methods...')
        
        # Method 1: Direct token authentication
        print('\nMethod 1 - Direct Token:')
        headers = {'Authorization': f'Token {refresh_token}'}
        response = requests.get(f'{base_url}/api/projects', headers=headers)
        print(f'Status: {response.status_code}')
        print('Response:', response.text[:200])
        
        # Method 2: Bearer token authentication
        print('\nMethod 2 - Bearer Token:')
        headers = {'Authorization': f'Bearer {refresh_token}'}
        response = requests.get(f'{base_url}/api/projects', headers=headers)
        print(f'Status: {response.status_code}')
        print('Response:', response.text[:200])
        
        # Method 3: Get new access token first
        print('\nMethod 3 - Token Refresh:')
        refresh_response = requests.post(
            f'{base_url}/api/token/refresh',
            json={'refresh': refresh_token},
            headers={'Content-Type': 'application/json'}
        )
        print('Refresh Status:', refresh_response.status_code)
        print('Refresh Response:', refresh_response.text[:200])
        
        if refresh_response.status_code == 200:
            access_token = refresh_response.json()['access']
            headers = {'Authorization': f'Bearer {access_token}'}
            response = requests.get(f'{base_url}/api/projects', headers=headers)
            print('Projects Status:', response.status_code)
            print('Projects Response:', response.text[:200])
            
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
        
        token_type = decoded.get('token_type', 'unknown')
        print('Token type:', token_type)
        
        if token_type == 'access':
            print('⚠️ WARNING: Environment contains an access token instead of a refresh token!')
            print('Please get a refresh token from Label Studio and update your .env file.')
        
        exp = decoded.get('exp')
        if exp:
            exp_dt = datetime.fromtimestamp(exp, tz=timezone.utc)
            now = datetime.now(timezone.utc)
            if exp_dt > now:
                print('Status: Valid until', exp_dt.isoformat())
            else:
                print('Status: Expired on', exp_dt.isoformat())
                print('⚠️ Token has expired! Please get a new token.')
        else:
            print('Status: No expiration found')
            
        print('User ID:', decoded.get('user_id', 'unknown'))
        print('\nNext Steps:')
        print('1. Go to https://labels.aedev.cloud')
        print('2. Click your profile icon in the top right')
        print('3. Go to Account & Settings')
        print('4. Click on Access Tokens in the sidebar')
        print('5. Click "+ Create New Token"')
        print('6. Give it a name (e.g., "ChickenAI API")')
        print('7. Copy the new token')
        print('8. Update your .env file with the new token')
        
    except Exception as e:
        print('❌ Error analyzing token:', str(e))

def create_refresh_token():
    """Create a new refresh token through login"""
    print('\n🔑 Creating New Refresh Token:')
    try:
        base_url = os.getenv('LABEL_STUDIO_URL', 'https://labels.aedev.cloud')
        
        # Get credentials
        print('\nPlease enter your Label Studio credentials:')
        username = input('Username: ')
        password = getpass.getpass('Password: ')
        
        # Login to get tokens
        print('\nLogging in to Label Studio...')
        response = requests.post(
            f'{base_url}/api/auth/login/',
            json={'username': username, 'password': password},
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code != 200:
            print('❌ Login failed:')
            print(response.text)
            return None
            
        tokens = response.json()
        refresh_token = tokens.get('refresh')
        
        if not refresh_token:
            print('❌ No refresh token in response:')
            print(json.dumps(tokens, indent=2))
            return None
            
        # Update .env file
        env_path = Path(os.path.dirname(os.path.abspath(__file__))) / '.env'
        
        # Read existing .env content
        env_content = {}
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        env_content[key] = value
        
        # Update token
        env_content['LABEL_STUDIO_URL'] = base_url
        env_content['LABEL_STUDIO_API_KEY'] = refresh_token
        
        # Write back to .env
        with open(env_path, 'w') as f:
            for key, value in env_content.items():
                f.write(f'{key}={value}\n')
        
        print('✅ Successfully created new refresh token')
        print('✅ Updated .env file')
        
        # Decode and show token info
        try:
            decoded = jwt.decode(refresh_token, options={"verify_signature": False})
            print('\nToken information:')
            print(json.dumps(decoded, indent=2))
        except Exception as e:
            print('Note: Could not decode token:', str(e))
        
        return refresh_token
        
    except Exception as e:
        print('❌ Error creating refresh token:', str(e))
        return None

def main():
    print('🔍 Label Studio Integration Tests')
    print('================================')
    print('\nOptions:')
    print('1. Run tests with current token')
    print('2. Create new refresh token')
    print('3. Run both')
    
    choice = input('\nEnter your choice (1-3): ').strip()
    
    # Load environment variables
    load_dotenv()
    
    if choice == '1':
        test_token_info()
        test_direct_api()
        test_sdk()
    elif choice == '2':
        create_refresh_token()
    elif choice == '3':
        new_token = create_refresh_token()
        if new_token:
            print('\nTesting with new token...')
            # Reload environment variables
            load_dotenv()
            test_token_info()
            test_direct_api()
            test_sdk()
    else:
        print('Invalid choice!')

if __name__ == '__main__':
    main()