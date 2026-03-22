import requests

API_URL = "http://192.168.0.84:11434/api/chat"


def ask_llm(prompt, model, think=False):
    """Send prompt to LLM and return response.

    Returns:
        - str content (default)
        - dict with content/thinking if with_thinking=True
    """
    data = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "think": think,
    }
    response = requests.post(API_URL, json=data)
    response.raise_for_status()
    message = response.json().get("message", {})
    content = message.get("content", "").strip()
    thinking = message.get("thinking", "").strip()

  
    return {
            "content": content,
            "thinking": thinking,
        }

