import asyncio
import os
import random
import shutil
import string
import subprocess
import time

import nodriver
from neotermcolor import colored


def generate_random_string(length=10):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


class CustomEventLoopPolicy(asyncio.DefaultEventLoopPolicy):
    def __init__(self):
        super().__init__()
        self._loop = None

    def new_event_loop(self):
        self._loop = super().new_event_loop()
        return self._loop

    def get_event_loop(self):
        if self._loop is None or self._loop.is_closed():
            self._loop = self.new_event_loop()
        return self._loop

    def set_event_loop(self, loop):
        self._loop = loop

    def handle_exception(self, loop, context):
        if isinstance(context.get("exception"), RuntimeError) and "Event loop is closed" in str(
                context.get("exception")):
            return
        super().handle_exception(loop, context)


class ChatGPTBot:
    def __init__(self):
        self.browser = None
        self.page = None
        self.temp_folder_name = generate_random_string()

    async def initialize_browser(self):
        try:
            os.makedirs(f"/tmp/{self.temp_folder_name}", exist_ok=True)
            self.browser = await nodriver.start(
                headless=True,
                browser_args=['--fast', '--fast-start', '--no-first-run', '--no-service-autorun',
                              '--password-store=basic'],
                user_data_dir=f"/tmp/{self.temp_folder_name}"
            )
            self.page = await self.browser.get("https://chat.openai.com")
            return True
        except Exception as e:
            print(colored(f"Error initializing browser: {e}", 'red'))
            return False

    async def wait_for_page_load(self):
        max_attempts = 15
        for attempt in range(max_attempts):
            try:
                text_area = await self.page.select('textarea')
                if text_area:
                    await asyncio.sleep(1)
                    return True
                await asyncio.sleep(2)
            except Exception:
                await asyncio.sleep(2)
        return False

    async def send_message(self, message_text):
        try:
            text_area = await self.page.select('textarea')
            if not text_area:
                return False
            await text_area.clear_input()
            await text_area.send_keys(message_text)
            await text_area.send_keys('\r\n')
            return True
        except Exception as e:
            print(colored(f"Error sending message: {e}", 'red'))
            return False

    async def wait_for_response(self, timeout=20):
        start_time = time.time()
        response_text = ""
        while time.time() - start_time < timeout:
            try:
                paragraph_elements = await self.page.select_all('p')
                if paragraph_elements:
                    response_text = paragraph_elements[0]
                    if response_text and len(str(response_text)) > 10:
                        return self.clean_html_without_re(str(response_text))
                await asyncio.sleep(1)
            except Exception:
                await asyncio.sleep(1)
        return self.clean_html_without_re(str(response_text)) if response_text else "Sorry, I couldn't get a response."

    async def close_browser(self):
        if self.browser:
            try:
                await self.browser.stop()
            except:
                pass
        try:
            shutil.rmtree(f"/tmp/{self.temp_folder_name}", ignore_errors=True)
        except:
            pass

    def clean_html_without_re(self, html_text):
        if not isinstance(html_text, str):
            try:
                html_text = str(html_text)
            except Exception:
                return ""
        cleaned_string = ""
        in_tag = False
        for char in html_text:
            if char == '<':
                in_tag = True
            elif char == '>':
                in_tag = False
            else:
                if not in_tag:
                    cleaned_string += char
        return cleaned_string


def process_with_chatgpt_subprocess(user_input: str) -> str:
    import tempfile
    import concurrent.futures

    def run_chatgpt_script():
        chatgpt_script = '''#!/usr/bin/python3
import os
import sys
import time
import shutil
import random
import string
import asyncio
import nodriver

def generate_random_string(length=10):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

temp_folder_name = generate_random_string()
os.makedirs(f"/tmp/{temp_folder_name}")

class CustomEventLoopPolicy(asyncio.DefaultEventLoopPolicy):
    def __init__(self):
        super().__init__()
        self._loop = None

    def new_event_loop(self):
        self._loop = super().new_event_loop()
        return self._loop

    def get_event_loop(self):
        if self._loop is None or self._loop.is_closed():
            self._loop = self.new_event_loop()
        return self._loop

    def set_event_loop(self, loop):
        self._loop = loop

    def handle_exception(self, loop, context):
        if isinstance(context.get("exception"), RuntimeError) and "Event loop is closed" in str(context.get("exception")):
            return
        super().handle_exception(loop, context)

class ChatGPTBot:
    global temp_folder_name
    def __init__(self):
        self.browser = None
        self.page = None

    async def initialize_browser(self):
        try:
            self.browser = await nodriver.start(
                headless=False,
                browser_args=['--fast', '--fast-start', '--no-first-run', '--no-service-autorun', '--password-store=basic'],
                user_data_dir=f"/tmp/{temp_folder_name}"
                )
            self.page = await self.browser.get("https://chat.openai.com")
            return True
        except Exception:
            return False

    async def wait_for_page_load(self):
        max_attempts = 15
        for attempt in range(max_attempts):
            try:
                text_area = await self.page.select('textarea')
                if text_area:
                    await asyncio.sleep(1)
                    return True
                await asyncio.sleep(2)
            except Exception:
                await asyncio.sleep(2)
        return False

    async def send_message(self, message_text):
        try:
            text_area = await self.page.select('textarea')
            if not text_area:
                return False
            await text_area.clear_input()
            await text_area.send_keys(message_text)
            await text_area.send_keys('\\r\\n')
            return True
        except Exception:
            return False

    async def wait_for_response(self, timeout=10):
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                paragraph_elements = await self.page.select_all('p')
                if paragraph_elements:
                    response_text = paragraph_elements[0]
                    if response_text and len(response_text) > 10:
                        return response_text
                await asyncio.sleep(1)
            except Exception:
                await asyncio.sleep(1)
        return response_text

    async def close_browser(self):
        if self.browser:
            try:
                await self.browser.stop()
            except:
                pass

def clean_html_without_re(html_text):
    if not isinstance(html_text, str):
        try:
            html_text = str(html_text)
        except Exception:
            return ""
    cleaned_string = ""
    in_tag = False
    for char in html_text:
        if char == '<':
            in_tag = True
        elif char == '>':
            in_tag = False
        else:
            if not in_tag:
                cleaned_string += char
    return cleaned_string

async def main():
    if len(sys.argv) != 2:
        print("Error: Missing argument")
        return

    user_message = sys.argv[1]
    bot = ChatGPTBot()

    try:
        if not await bot.initialize_browser():
            print("Error: Could not initialize browser.")
            return

        if not await bot.wait_for_page_load():
            print("Error: Could not load ChatGPT page.")
            return

        if await bot.send_message(user_message):
            bot_response = await bot.wait_for_response()
            bot_response = clean_html_without_re(bot_response) if bot_response else None
            if bot_response:
                print(bot_response)
            else:
                print("Sorry, I couldn't generate a response.")
        else:
            print("Error: Could not send message.")

    except Exception as e:
        print(f"Unexpected error: {str(e)}")

    finally:
        await bot.close_browser()
        time.sleep(0.2)
        shutil.rmtree(f"/tmp/{temp_folder_name}", ignore_errors=True)

if __name__ == "__main__":
    asyncio.set_event_loop_policy(CustomEventLoopPolicy())
    asyncio.run(main())
'''

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(chatgpt_script)
                script_path = f.name

            cmd = ["python3", script_path, user_input]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            os.unlink(script_path)

            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            else:
                return "Sorry, I couldn't generate a response."

        except subprocess.TimeoutExpired:
            return "Error: Timeout when generating response."
        except FileNotFoundError:
            return "Error: Python3 not found."
        except Exception as e:
            return f"Unexpected error: {str(e)}"

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        try:
            return loop.run_in_executor(executor, run_chatgpt_script)
        except Exception as e:
            return f"Unexpected error: {str(e)}"


async def process_with_chatgpt(user_input: str) -> str:
    print("Using nodriver subprocess for ChatGPT interaction.")
    return await process_with_chatgpt_subprocess(user_input)
