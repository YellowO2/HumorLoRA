import requests
import torch

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


class LocalModel:
    """Runs a LoRA checkpoint directly via FastLanguageModel — no merge, no GGUF."""

    def __init__(self, name: str, checkpoint: str):
        self.name = name
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
        FastLanguageModel.for_inference(model)
        self._model = model
        self._tokenizer = tokenizer  # full tokenizer with apply_chat_template
        # use the underlying text tokenizer for encoding/decoding raw tokens
        self._tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer

    def ask(self, prompt: str, think: bool = False, history: list = None, max_new_tokens: int = 512) -> dict:
        # Build messages list: prior history + new user turn
        messages = list(history) if history else []
        messages.append({"role": "user", "content": prompt})
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=False,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        content = self._tok.decode(new_tokens, skip_special_tokens=True).strip()
        return {"content": content, "thinking": ""}

    def unload(self) -> None:
        del self._model
        del self._tok
        del self._tokenizer
        import gc
        gc.collect()
        torch.cuda.empty_cache()
