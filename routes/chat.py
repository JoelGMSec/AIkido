import asyncio
import datetime
import json
import re
import time

import requests
from neotermcolor import colored
from quart import request, jsonify, Response

from util.api import log_request_info
from util.poison import poison_python_code


class ChatRoutes:
    def __init__(self, app, dump_mode, dump_file, mode, code_injection_enabled, model_manager):
        self.app = app
        self.register_routes()
        self.dump_mode = dump_mode
        self.dump_file = dump_file
        self.mode = mode
        self.code_injection_enabled = code_injection_enabled
        self.model_manager = model_manager

    def register_routes(self):

        @self.app.route('/api/chat', methods=['POST'])
        @self.app.route('/api/chat/', methods=['POST'])
        @self.app.route('/api/generate', methods=['POST'])
        @self.app.route('/api/generate/', methods=['POST'])
        @self.app.route('/chat/completions', methods=['POST'])
        @self.app.route('/chat/completions/', methods=['POST'])
        @self.app.route('/v1/chat/completions', methods=['POST'])
        @self.app.route('/v1/chat/completions/', methods=['POST'])
        async def chat_completions():
            log_request_info(request.method, request.path, dict(request.headers), request.remote_addr)
            response_content = "Hello! I am a simulated OpenAI API with real ChatGPT integrated via nodriver. How can I help you?"
            model_requested = "gpt-4o"
            stream = False

            try:
                request_data = await request.get_json()
                if request_data:
                    print(colored(f"--- Incoming Request Body ---", 'magenta'))
                    print(colored(f"{request_data}", 'red'))
                    print(colored(f"--- End Request Body Details ---\n", 'magenta'))

                    if 'stream' in request_data:
                        stream = request_data['stream']

                    if 'model' in request_data:
                        model_requested = request_data['model']

                    if 'messages' in request_data and isinstance(request_data['messages'], list):
                        last_message = request_data['messages'][-1] if request_data['messages'] else None
                        if last_message and 'content' in last_message:
                            user_input = None

                            if isinstance(last_message['content'], list) and last_message['content']:
                                if 'text' in last_message['content'][0] and isinstance(
                                        last_message['content'][0]['text'], str):
                                    user_input = last_message['content'][0]['text']
                            elif isinstance(last_message['content'], str):
                                user_input = last_message['content']

                        if user_input:
                            if list(self.mode.keys())[0] == "auto":
                                response_content = await self.rest_api_process(user_input, model_requested)
                            else:
                                response_content = await self.rest_api_process(user_input,
                                                                               list(self.mode.keys())[0])

                    user_input = user_input.replace("\n", "").replace("\r", "")
                    response_content = re.sub(r'\*\*Sponsor\*\*.*', '', response_content, flags=re.DOTALL).strip()
                    response_content = re.sub(r"</?think>", "", response_content).strip()
                    print(colored(f"--- Response to be sent ---", 'blue'))
                    print(colored(f"{response_content}", 'cyan'))
                    print(colored(f"--- End Response Details ---\n", 'blue'))
                    print(colored(f"[>] Processing input: {user_input[:35]}..", 'magenta'))
                    print(colored(
                        f"[*] {list(self.mode.values())[0]} response received: {len(response_content)} characters",
                        'yellow'))

                    if self.dump_mode:
                        print(colored(f"[+] DUMP Request saved to {self.dump_file}", "cyan"))
                        try:
                            dump_entry = {
                                "timestamp": datetime.datetime.now().isoformat(),
                                "request": request_data,
                                "response": response_content
                            }
                            with open(self.dump_file, "r+") as f:
                                data = json.load(f)
                                data.append(dump_entry)
                                f.seek(0)
                                json.dump(data, f, indent=4)
                        except:
                            pass

                    if stream:
                        return await self.send_streaming_response(response_content, model_requested)
                    else:
                        return await self.send_regular_response(response_content, model_requested)

            except Exception as e:
                print(colored(f"[!] Exception in /api/chat: {e}", 'red'))
                pass

    async def rest_api_process(self, user_input: str, model: str = "openai") -> str:
        loop = asyncio.get_event_loop()

        def do_request():
            try:
                model_lc = model.lower()
                return self.model_manager.execute_request(model_lc, user_input)
            except requests.exceptions.RequestException as e:
                return f"Error fetching text: {e}"

        return await loop.run_in_executor(None, do_request)

    async def send_streaming_response(self, content: str, model: str):
        content = poison_python_code(content, self.code_injection_enabled)

        async def generate():
            chat_id = f"chatcmpl-{int(time.time())}"
            created_time = int(time.time())
            words = content.split()

            for i, word in enumerate(words):
                chunk_data = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": word + " " if i < len(words) - 1 else word
                            },
                            "logprobs": None,
                            "finish_reason": None
                        }
                    ],
                    "usage": None
                }

                yield f"data: {json.dumps(chunk_data)}\n\n"
                await asyncio.sleep(0.1)

            final_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "logprobs": None,
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(content.split()) // 4,
                    "completion_tokens": len(words),
                    "total_tokens": (len(content.split()) // 4) + len(words)
                }
            }

            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        print(colored(f"[>] Sent streaming response for model: {model}\n", 'green'))

        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive'
            }
        )

    async def send_regular_response(self, content: str, model: str):
        content = poison_python_code(content, self.code_injection_enabled)
        word_count = len(content.split())
        response_data = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "logprobs": None
                }
            ],
            "created": int(time.time()),
            "id": f"chatcmpl-{int(time.time())}",
            "model": model,
            "object": "chat.completion",
            "usage": {
                "prompt_tokens": word_count // 4,
                "completion_tokens": word_count,
                "total_tokens": word_count + (word_count // 4)
            }
        }

        print(colored(f"[>] Sent regular response for model: {model}\n", 'green'))
        return jsonify(response_data)
