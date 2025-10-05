from neotermcolor import colored
from quart import request, jsonify

from util.api import log_request_info


class VersionRoutes:
    def __init__(self, app):
        self.app = app
        self.register_routes()

    def register_routes(self):
        @self.app.route('/version', methods=['GET'])
        @self.app.route('/version/', methods=['GET'])
        async def version():
            log_request_info(request.method, request.path, dict(request.headers), request.remote_addr)

            response_data = {
                'app_name': 'AIkido Simulated API with ChatGPT',
                'current_version': '2.1.0',
                'protocol_version': 'HTTP/2',
                'release_date': '2025-01-21',
                'status': 'operational',
                'powered_by': 'darkbyte.net + nodriver',
                'chatgpt_integration': 'enabled'
            }

            print(colored("[>] Sent /version response successfully\n", 'green'))
            return jsonify(response_data)
