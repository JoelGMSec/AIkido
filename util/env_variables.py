import os

from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv('AIKIDO_HOST', '0.0.0.0')
HTTP_PORT = int(os.getenv('AIKIDO_HTTP_PORT', '80'))
HTTPS_PORT = int(os.getenv('AIKIDO_HTTPS_PORT', '443'))
ENABLE_HTTP = bool(os.getenv('AIKIDO_ENABLE_HTTP', 'true').lower() in ['true', '1', 'yes'])
ENABLE_HTTPS = bool(os.getenv('AIKIDO_ENABLE_HTTPS', 'true').lower() in ['true', '1', 'yes'])
KEY_FILE = os.getenv('AIKIDO_KEY_FILE', 'cert/key.pem')
CERT_FILE = os.getenv('AIKIDO_CERT_FILE', 'cert/cert.pem')
