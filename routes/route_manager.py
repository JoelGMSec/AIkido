from middleware.request import RequestMiddleware
from routes.api_keys import ApiKeys
from routes.app_info import AppInfo
from routes.chat import ChatRoutes
from routes.models_info import ModelsInfo
from routes.root import RootRoute
from routes.version import VersionRoutes


class RouteManager:
    def __init__(self, app, dump_mode, dump_file, mode, code_injection_enabled, model_manager):
        self.app = app
        self.register_middleware()
        self.register_routes(dump_mode, dump_file, mode, code_injection_enabled, model_manager)

    def register_middleware(self):
        RequestMiddleware(self.app)

    def register_routes(self, dump_mode, dump_file, mode, code_injection_enabled, model_manager):
        AppInfo(self.app)
        ApiKeys(self.app)
        ModelsInfo(self.app)
        ChatRoutes(self.app, dump_mode, dump_file, mode, code_injection_enabled, model_manager)
        VersionRoutes(self.app)
        RootRoute(self.app)
