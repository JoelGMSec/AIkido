from neotermcolor import colored
from quart import request, jsonify

from util.api import log_request_info


class AppInfo:
    def __init__(self, app):
        self.app = app
        self.register_routes()

    def register_routes(self):
        @self.app.route('/v1', methods=['GET', 'POST'])
        @self.app.route('/v1/', methods=['GET', 'POST'])
        @self.app.route('/api/v1', methods=['GET', 'POST'])
        @self.app.route('/api/v1/', methods=['GET', 'POST'])
        async def api_v1():
            log_request_info(request.method, request.path, dict(request.headers), request.remote_addr)

            response_data = {
                'status': 'success',
                'version': '2.0',
                'api-version': '2.0',
                'protocol': 'HTTP/2',
                'message': 'AIkido API v2 operational with HTTP/2 support and ChatGPT NoDriver integration',
                'endpoints': ['/account/api-keys', '/api/v1', '/v1/models', '/v1/chat/completions', '/version'],
                'author': '@JoelGMSec'
            }

            print(colored("[>] Sent /v1 response successfully\n", 'green'))
            return jsonify(response_data)
