from quart import request, Response


class RequestMiddleware:
    def __init__(self, app):
        self.app = app
        self.register_middleware()

    def register_middleware(self):
        @self.app.before_request
        async def before_request():
            if request.method == 'OPTIONS':
                response = Response('')
                response.headers['Access-Control-Allow-Origin'] = '*'
                response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
                response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
                response.headers['Access-Control-Max-Age'] = '86400'
                return response
            return None

        @self.app.after_request
        async def after_request(response):
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
            return response
