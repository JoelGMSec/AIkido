from neotermcolor import colored
from quart import request, jsonify

from util.api import log_request_info


class ApiKeys:
    def __init__(self, app):
        self.app = app
        self.register_routes()

    def register_routes(self):
        @self.app.route('/account/api-keys', methods=['GET', 'POST'])
        @self.app.route('/account/api-keys/', methods=['GET', 'POST'])
        async def api_keys():
            log_request_info(request.method, request.path, dict(request.headers), request.remote_addr)

            response_data = {
                'message': 'API Key OK',
                'type': 'gpt-4o',
                'model': 'gpt-4o',
                'param': None,
                'code': '100% Valid API Key - Real no Fake'
            }

            print(colored("[>] Sent /account/api-keys response successfully\n", 'green'))
            return jsonify(response_data)
