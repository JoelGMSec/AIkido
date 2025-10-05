import json
import random
import urllib.parse

import requests


class ModelManager:

    def __init__(self):
        self.MODELS = {
            "openai": ["gpt"],
            "gemini": ["gemma"],
            "hacktricks": ["hacktricks"],
            "deepseek": ["deepseek"],
            "phind": ["phind"],
        }
        self.METHODS = {
            "openai": self.get_openai_request,
            "gemini": self.get_gemini_request,
            "hacktricks": self.get_hacktricks_request,
            "deepseek": self.get_deepseek_request,
            "phind": self.get_phind_request,
        }

    def execute_request(self, requested_model, user_input: str):
        model = self.get_model(requested_model)
        if model is None:
            model = "openai"  # Default to openai if model not found
        if model in self.METHODS:
            method = self.METHODS[model]
            return method(user_input)
        else:
            raise ValueError(f"Model '{requested_model}' is not supported.")

    def get_model(self, name):
        for model_name, aliases in self.MODELS.items():
            if name.lower() in aliases:
                return model_name
        return None

    def get_openai_request(self, user_input: str):
        seed = random.randrange(11, 99, 2)

        encoded_prompt = urllib.parse.quote(user_input)
        url = f"https://text.pollinations.ai/{encoded_prompt}"
        params = {"model": "openai", "seed": int(seed)}
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        return r.text.strip()

    def get_gemini_request(self, user_input: str):
        url = "https://g4f.dev/api/nvidia/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer"
        }
        payload = {
            "model": "google/gemma-3-27b-it",
            "messages": [
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        }
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        try:
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
        except (json.JSONDecodeError, KeyError, IndexError):
            return r.text.strip()

    def get_hacktricks_request(self, user_input: str):
        url = "https://www.hacktricks.ai/api/ht-api"
        r = requests.post(url, json={"query": user_input}, timeout=420)
        r.raise_for_status()
        try:
            data = r.json()
            return data.get("response", "").strip()
        except json.JSONDecodeError:
            return r.text.strip()

    def get_deepseek_request(self, user_input: str):
        url = "https://api.deepinfra.com/v1/openai/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer"
        }
        payload = {
            "model": "deepseek-ai/DeepSeek-V3-0324-Turbo",
            "messages": [
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        }
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        try:
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
        except (json.JSONDecodeError, KeyError, IndexError):
            return r.text.strip()

    def get_phind_request(self, user_input: str):
        url = "https://https.extension.phind.com/agent/"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "",
            "Accept": "*/*",
            "Accept-Encoding": "Identity"
        }
        payload = {
            "additional_extension_context": "",
            "allow_magic_buttons": True,
            "is_vscode_extension": True,
            "requested_model": "Phind-70B",
            "user_input": user_input,
            "message_history": [
                {
                    "role": "system",
                    "content": "You are a helpfully assistant."
                },
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        }
        r = requests.post(url, headers=headers, json=payload, stream=True, timeout=120)
        r.raise_for_status()

        full_text = ""
        for line in r.iter_lines():
            if not line:
                continue
            if line.startswith(b"data: "):
                try:
                    obj = json.loads(line.decode()[6:])
                    if "choices" in obj and obj["choices"]:
                        delta = obj["choices"][0]["delta"]
                        if "content" in delta:
                            full_text += delta["content"]
                except Exception:
                    continue
        return full_text.strip()
