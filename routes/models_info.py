from neotermcolor import colored
from quart import request, jsonify

from util.api import log_request_info


class ModelsInfo:
    def __init__(self, app):
        self.app = app
        self.register_routes()

    def register_routes(self):
        @self.app.route('/v1/models', methods=['GET', 'POST'])
        @self.app.route('/v1/models/', methods=['GET', 'POST'])
        @self.app.route('/models', methods=['GET', 'POST'])
        @self.app.route('/models/', methods=['GET', 'POST'])
        async def models():
            log_request_info(request.method, request.path, dict(request.headers), request.remote_addr)

            response_data = {
                "object": "list",
                "data": [
                    {
                        "id": "deepseek-v3",
                        "object": "model",
                        "created": 1735686000,
                        "owned_by": "deepseek"
                    },
                    {
                        "id": "gemma-3",
                        "object": "model",
                        "created": 1735686000,
                        "owned_by": "google"
                    },
                    {
                        "id": "gpt-5-nano",
                        "object": "model",
                        "created": 1735686000,
                        "owned_by": "openai"
                    },
                    {
                        "id": "hacktricks",
                        "object": "model",
                        "created": 1735686000,
                        "owned_by": "hacktricks"
                    },
                    {
                        "id": "phind-70b",
                        "object": "model",
                        "created": 1735686000,
                        "owned_by": "phind"
                    }
                ]
            }

            print(colored("[>] Sent /v1/models response successfully\n", 'green'))
            return jsonify(response_data)
