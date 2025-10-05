#!/usr/bin/python3
# ========================#
#  AIkido by @JoelGMSec  #
#      darkbyte.net      #
# ========================#

import argparse
import asyncio
import datetime
import json
import logging
import os
import signal
import sys

from hypercorn.asyncio import serve
from hypercorn.config import Config
from neotermcolor import colored
from quart import Quart

import util.env_variables as env
from routes.route_manager import RouteManager
from util.models import ModelManager

logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger('quart').setLevel(logging.CRITICAL)
logging.getLogger('hypercorn').setLevel(logging.CRITICAL)
logging.getLogger('asyncio').setLevel(logging.CRITICAL)

# Global variables
MODE = None
DUMP_MODE = False
DUMP_FILE = None
CODE_INJECTION_ENABLED = False
shutdown_event = asyncio.Event()

# Create Quart app
app = Quart(__name__)
app.logger.setLevel(logging.CRITICAL)


def signal_handler():
    print(colored(f"\n[!] Ctrl+C Pressed! Shutting down server..\n", 'red'))
    exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


async def run_server_with_shutdown(config, server_type):
    try:
        server = await serve(app, config, shutdown_trigger=shutdown_event.wait)
        print(colored(f"[!] {server_type} server stopped", 'yellow'))
    except Exception:
        pass


async def run_http_server():
    config = Config()
    config.bind = [f"{env.HOST}:{env.HTTP_PORT}"]
    config.accesslog = None
    config.errorlog = None
    config.use_reloader = False
    config.loglevel = "critical"
    config.access_logger = None
    config.error_logger = None
    await run_server_with_shutdown(config, "HTTP")


async def run_https_server():
    if not os.path.exists(env.CERT_FILE) or not os.path.exists(env.KEY_FILE):
        print(colored(f"Warning: Certificate file ({env.CERT_FILE}) or key file ({env.KEY_FILE}) not found.", 'yellow'))
        print(colored("HTTPS server will not start. Please ensure SSL certificate files exist.", 'yellow'))
        print(colored("You can generate self-signed certificates with:", 'yellow'))
        print(
            colored(f"openssl req -x509 -newkey rsa:4096 -keyout {env.KEY_FILE} -out {env.CERT_FILE} -days 365 -nodes",
                    'yellow'))
        return

    config = Config()
    config.bind = [f"{env.HOST}:{env.HTTPS_PORT}"]
    config.certfile = env.CERT_FILE
    config.keyfile = env.KEY_FILE
    config.accesslog = None
    config.errorlog = None
    config.use_reloader = False
    config.loglevel = "critical"
    config.access_logger = None
    config.error_logger = None
    config.alpn_protocols = ['h2', 'http/1.1']
    await run_server_with_shutdown(config, "HTTPS")


async def run_dual_server():
    model_manager = ModelManager()
    route_manager = RouteManager(app, DUMP_MODE, DUMP_FILE, MODE, CODE_INJECTION_ENABLED, model_manager)
    tasks = []

    if env.ENABLE_HTTP:
        tasks.append(asyncio.create_task(run_http_server()))

    if env.ENABLE_HTTPS:
        tasks.append(asyncio.create_task(run_https_server()))

    if not tasks:
        print(colored("No servers were started. Please check your configuration.", 'red'))
        return

    print(colored("=" * 40, 'blue'))
    print(colored("       AIkido API Server Started", 'magenta', attrs=['bold']))
    print(colored("=" * 40, 'blue'))
    print(colored(f"> HTTP:  http://{env.HOST}:{env.HTTP_PORT} (HTTP/1)" if env.ENABLE_HTTP else "HTTP: Disabled",
                  'green' if env.ENABLE_HTTP else 'red'))
    print(colored(f"> HTTPS: https://{env.HOST}:{env.HTTPS_PORT} (HTTP/2)" if env.ENABLE_HTTPS else "HTTPS: Disabled",
                  'green' if env.ENABLE_HTTPS else 'red'))
    print(colored(f"> API Model Backend: {list(MODE.values())[0]}", 'yellow'))
    if CODE_INJECTION_ENABLED:
        print(colored(f"> CODE INJECTION: ENABLED (PYTHON)", 'red'))
    if DUMP_MODE:
        print(colored(f"> DUMP Logging to file: ENABLED", "cyan"))
    print(colored("=" * 40, 'blue'))
    print(colored("Available endpoints:", 'magenta'))
    print(colored("> api.openai.com", 'white'))
    print(colored("> api.deepseek.com", 'white'))
    print(colored("> api.gemini.com", 'white'))
    print(colored("> api.hacktricks.xyz", 'white'))
    print(colored("> api.phind.com", 'white'))
    print(colored("> api.deepseek.ai", 'white'))
    print(colored("> localhost (ollama)", 'white'))
    print(colored("=" * 40, 'blue'))
    print(colored("   Press Ctrl+C to stop both servers", 'red'))
    print(colored("=" * 40 + "\n", 'blue'))

    try:
        done, pending = await asyncio.wait(
            tasks + [asyncio.create_task(shutdown_event.wait())],
            return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    except KeyboardInterrupt:
        pass
    except Exception:
        pass
    finally:
        shutdown_event.set()


if __name__ == '__main__':
    print(colored(r"""
    _____   ___ __    __    __      
   /  _  \ |   |  | _|__| _|  | _____  
  /  / \  \|   |  |/ /  |/ __ |/  _  \ 
 /  /___\  \   |    <|  / /_/ |  (_)  |
 \  _______/___|__|_ \__\_____|\_____/ 
  \/                \/             
    """, "blue"))

    print(colored("  ----------- by @JoelGMSec ----------\n", "green"))

    try:
        import quart
        import hypercorn
        import nodriver
    except ImportError as e:
        print(colored(f"Required package not found: {e}", 'red'))
        print(colored("Please install required packages:", 'yellow'))
        print(colored("pip install quart hypercorn nodriver", 'yellow'))
        sys.exit(1)

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dump", action="store_true", help="Enable dump mode and save output to JSON")
    parser.add_argument("--poison", action="store_true", help="Enable code injection mode (Python)")
    parser.add_argument("--no-driver", action="store_true", help="Run in ChatGPT (NoDriver mode)")
    parser.add_argument("--api-rest", choices=["auto", "deepseek", "gemini", "hacktricks", "openai", "phind"],
                        help="Run in API REST mode (auto/deepseek/gemini/hacktricks/openai/phind)")
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    if args.dump:
        DUMP_MODE = True
        fecha_str = datetime.datetime.now().strftime("%Y%m%d")
        DUMP_FILE = f"dump/AIkido_{fecha_str}.json"
        with open(DUMP_FILE, "w") as f:
            json.dump([], f, indent=4)

    if args.poison:
        CODE_INJECTION_ENABLED = True

    if args.no_driver:
        MODE = {"nodriver": "ChatGPT (NoDriver)"}
    elif args.api_rest == "auto":
        MODE = {"auto": "Automatic REST API"}
    elif args.api_rest == "deepseek":
        MODE = {"deepseek": "Deepseek REST API"}
    elif args.api_rest == "openai":
        MODE = {"openai": "OpenAI REST API"}
    elif args.api_rest == "gemini":
        MODE = {"gemini": "Gemini REST API"}
    elif args.api_rest == "hacktricks":
        MODE = {"hacktricks": "HackTricks REST API"}
    elif args.api_rest == "phind":
        MODE = {"phind": "Phind REST API"}

    asyncio.run(run_dual_server())
