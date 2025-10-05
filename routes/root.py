from quart import request, jsonify

from util.api import log_request_info


class RootRoute:
    def __init__(self, app):
        self.app = app
        self.register_routes()

    def register_routes(self):
        @self.app.route('/', methods=['GET'])
        async def root():
            log_request_info(request.method, request.path, dict(request.headers), request.remote_addr)

            response_data = {
                'message': 'AIkido API Server is running with ChatGPT integration',
                'version': '2.1.0',
                'protocol': 'HTTP/2',
                'chatgpt_backend': 'nodriver',
                'endpoints': {
                    'models': '/v1/models',
                    'chat': '/v1/chat/completions',
                    'api_keys': '/account/api-keys',
                    'version': '/version'
                }
            }

            return jsonify(response_data)
