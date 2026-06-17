import os
os.environ["HF_ENDPOINT"] = "https://huggingface.co"

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

    def __init__(self, name: str, checkpoint: str, chat_template: str = "gemma-4", enable_thinking: bool = False):
        self.name = name
        self._enable_thinking = enable_thinking
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        tokenizer = get_chat_template(tokenizer, chat_template=chat_template)
        FastLanguageModel.for_inference(model)
        self._model = model
        self._tokenizer = tokenizer  # full tokenizer with apply_chat_template
        # use the underlying text tokenizer for encoding/decoding raw tokens
        self._tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer

    def ask(self, prompt: str, think: bool = False, history: list = None, max_new_tokens: int = 512) -> dict:
        # Build messages list: prior history + new user turn
        messages = list(history) if history else []
        messages.append({"role": "user", "content": prompt})
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self._enable_thinking is not None:
            kwargs["enable_thinking"] = think if self._enable_thinking else False
        text = self._tokenizer.apply_chat_template(messages, **kwargs)
        inputs = self._tok(text, return_tensors="pt").to("cuda")
        # Qwen3.5 canonical EOS IDs + whatever the (possibly remapped) tokenizer reports
        stop_ids = [151645, 151643, self._tok.eos_token_id]
        for t in ["<|im_end|>", "<|endoftext|>", "<end_of_turn>"]:
            tid = self._tok.convert_tokens_to_ids(t)
            if tid is not None and tid != self._tok.unk_token_id:
                stop_ids.append(tid)
        stop_ids = list(set(i for i in stop_ids if i is not None))

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=False,
                eos_token_id=stop_ids,
                pad_token_id=self._tok.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        content = self._tok.decode(new_tokens, skip_special_tokens=True).strip()
        # trim at the first fake turn marker (models trained on chat data sometimes
        # generate the next turn as plain text instead of stopping)
        import re
        content = re.split(r"<\|?(?:start_of_)?turn[>\|]|<end_of_turn>|\buser\n|\bmodel\n", content)[0].strip()
        return {"content": content, "thinking": ""}

    def unload(self) -> None:
        del self._model
        del self._tok
        del self._tokenizer
        import gc
        gc.collect()
        torch.cuda.empty_cache()
