<p align="center"><img width=500 alt="AIkido" src="https://github.com/JoelGMSec/AIkido/blob/main/AIkido.png"></p>

# AIkido

### Fake-AI API Server

**AIkido** is a fake OpenAI-compatible API server that integrates real **ChatGPT** (via `nodriver`) and supports multiple AI backends (DeepSeek, Gemini, HackTricks, Phind, OpenAI).  
It provides a **drop-in replacement API** for testing, research, and red team exercises, with optional features such as **HTTP/2, SSL, API key emulation, request logging, dump mode, and code injection simulation**.  


## ✨ Features

- 🌐 **Simulated OpenAI API**: Compatible with `/v1/chat/completions`, `/v1/models`, `/account/api-keys`, `/version`.
- ⚡ **Multi-backend support**: Choose between `ChatGPT (NoDriver)`, `DeepSeek`, `Gemini`, `HackTricks`, `OpenAI`, or `Phind` REST APIs.
- 🐍 **Code injection simulation**: Payload response injection to test LLM supply chain or insecure eval scenarios.
- 📜 **Dump logging**: Save all requests and responses to JSON for auditing or analysis.
- 🔑 **API key validation**: Fake endpoint to simulate valid OpenAI compatible API keys.
- 🔥 **Dual HTTP/HTTPS server**: Full HTTP/2 support with self-signed certificate support and stream response.


## ⚙️ Requirements

- Python **3.8+**
- Dependencies: `quart`, `hypercorn`, `nodriver`, `requests`, `neotermcolor`
- Optional: SSL certificate/key for HTTPS (self-signed supported)  


## 🚀 Usage

```bash
    _____   ___ __    __    __      
   /  _  \ |   |  | _|__| _|  | _____  
  /  / \  \|   |  |/ /  |/ __ |/  _  \ 
 /  /___\  \   |    <|  / /_/ |  (_)  |
 \  _______/___|__|_ \__\_____|\_____/ 
  \/                \/             
    
  ----------- by @JoelGMSec ----------

# Basic help
python3 AIkido.py -h

# Run with ChatGPT browser automation
python3 AIkido.py --no-driver

# Run with REST API (auto-detect best backend)
python3 AIkido.py --api-rest auto

# Force DeepSeek backend
python3 AIkido.py --api-rest deepseek

# Enable code injection mode
python3 AIkido.py --poison

# Enable dump logging (save to dump/AIkido_YYYYMMDD.json)
python3 AIkido.py --dump

```


## 📸 Screenshots

<img width="1726" height="714" alt="Image" src="https://github.com/user-attachments/assets/62c62657-ae64-4f1e-9ff5-afe4750bedae"/>


## 🗂️ Documentation

The detailed guide of use can be found at the following link:

*To be disclosed.*


## 📄 License

This project is licensed under the GNU GPL-3.0 license - See the LICENSE file for more details.


## 👨‍💻 Contact

For more information, you can find me on Twitter as [@JoelGMSec](https://twitter.com/JoelGMSec) 

Other ways to contact me on my blog [darkbyte.net](https://darkbyte.net)


## ⚠️ Disclaimer

This software comes with no warranty, exclusively for educational purposes and authorized security audits.

The author is not responsible for any misuse or damage caused by this software.


## ☕ Support
Support my work by buying me a coffee:

[<img width=250 alt="buymeacoffe" src="https://cdn.buymeacoffee.com/buttons/v2/default-blue.png">](https://www.buymeacoffee.com/joelgmsec)
