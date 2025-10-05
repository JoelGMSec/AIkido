from typing import Dict

from neotermcolor import colored


def log_request_info(method: str, path: str, headers: Dict[str, str], client_ip: str):
    print(colored(f"--- Incoming {method} Request ---", 'green'))
    print(colored(f"Path: {path}", 'yellow'))
    print(colored(f"Method: {method}", 'yellow'))
    print(colored(f"Client Address: {client_ip}", 'yellow'))
    print(colored(f"Request Headers:", 'yellow'))
    for header, value in headers.items():
        print(colored(f"  {header}: {value}", 'white'))
    print(colored(f"--- End {method} Request Details ---\n", 'green'))
