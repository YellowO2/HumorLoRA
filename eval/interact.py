import requests

OLLAMA_URL = "http://localhost:11434/api/chat"


def unload(model: str) -> None:
    requests.post(OLLAMA_URL, json={"model": model, "keep_alive": 0, "messages": []})


def ask(prompt: str, model: str, think: bool = False) -> dict:
    response = requests.post(OLLAMA_URL, json={
        "model": model,
        "stream": False,
        "think": think,
        "messages": [{"role": "user", "content": prompt}],
    })
    response.raise_for_status()
    message = response.json().get("message", {})
    return {
        "content": message.get("content", "").strip(),
        "thinking": message.get("thinking", "").strip(),
    }
