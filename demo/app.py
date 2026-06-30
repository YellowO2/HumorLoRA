import os
import random
import re
import requests
import xml.etree.ElementTree as ET
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import hf_hub_download

try:
    import spaces
    ON_SPACES = bool(os.getenv("SPACE_ID"))
    if ON_SPACES:
        GPU = spaces.GPU(duration=120)
    else:
        GPU = lambda fn: fn
except ImportError:
    GPU = lambda fn: fn
    ON_SPACES = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_MODEL_ID = "unsloth/Qwen3.5-4B"
LORA_MODEL_ID = "potato-bug/haha-lora"
TARGET_LAYER = 16

SUBREDDITS = ["tifu", "confessions", "AITAH"]

# ---------------------------------------------------------------------------
# Reddit (RSS, no auth needed)
# ---------------------------------------------------------------------------
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:144.0) Gecko/20100101 Firefox/144.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36 Edg/136.0.0.0",
]
_NS = {"atom": "http://www.w3.org/2005/Atom"}

def _strip_html(html):
    text = re.sub(r"<[^>]+>", "", html).strip()
    # remove Reddit submission footer
    text = re.sub(r"\s*submitted by /u/\S+.*", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"\s*\[link\]\s*\[comments\].*", "", text, flags=re.DOTALL).strip()
    return text

def _fetch_batch(subreddit, after=None):
    params = "?limit=5" + (f"&after=t3_{after}" if after else "")
    url = f"https://www.reddit.com/r/{subreddit}/hot/.rss{params}"
    headers = {"User-Agent": random.choice(_USER_AGENTS)}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    entries = root.findall("atom:entry", _NS)
    candidates = []
    last_id = None
    for e in entries:
        title = e.findtext("atom:title", "", _NS).strip()
        content_el = e.find("atom:content", _NS)
        content_html = content_el.text or "" if content_el is not None else ""
        body = _strip_html(content_html)
        entry_id = e.findtext("atom:id", "", _NS)
        if entry_id:
            last_id = entry_id.split("_")[-1]
        if 150 < len(body) < 4000:
            candidates.append((title, body))
    return candidates, last_id

def fetch_reddit_post(subreddit):
    try:
        after = None
        for _ in range(3):
            candidates, after = _fetch_batch(subreddit, after)
            if candidates:
                title, body = random.choice(candidates)
                display = (
                    f"---\n"
                    f"**r/{subreddit}**\n\n"
                    f"## {title}\n\n"
                    f"{body}\n\n"
                    f"---"
                )
                return (title, body), display
        return None, "No suitable posts found after 3 attempts. Try again."
    except Exception as e:
        return None, f"Error fetching post: {e}"

# ---------------------------------------------------------------------------
# Model — lazy, must load inside a @GPU function on ZeroGPU
# ---------------------------------------------------------------------------
_tokenizer = None
_model = None
_head = None
_device = None

import time

N_ANGLES = 7

HUMOR_PERSONAS = [
    ("Dry wit",          "You are a deadpan comedian. Short, no exclamation marks."),
    ("Absurdist",        "You are an absurdist comic. Escalate one detail to something surreal."),
    ("Self-deprecating", "You are self-deprecating. This happened to you too, but worse."),
    ("Dad joke",         "You are a dad. Find the worst pun and commit to it."),
    ("Roast",            "You lovingly roast the person. Punchy, not mean."),
    ("Oversharer",       "You overshare a tangentially related story with an unexpected ending."),
    ("Armchair expert",  "You give unsolicited authoritative advice that misses the point."),
]

def _load_model():
    global _tokenizer, _model, _head, _device
    if _model is not None:
        print("[load] model already cached, skipping")
        return
    print(f"[load] starting — {BASE_MODEL_ID}")
    t0 = time.time()
    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    print(f"[load] tokenizer done ({time.time()-t0:.1f}s)")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    print(f"[load] base model done ({time.time()-t0:.1f}s)")
    _model = PeftModel.from_pretrained(base, LORA_MODEL_ID)
    _model.eval()
    print(f"[load] lora done ({time.time()-t0:.1f}s)")
    head_path = hf_hub_download(repo_id=LORA_MODEL_ID, filename="head.pt")
    _device = next(_model.parameters()).device
    print(f"[load] device={_device}")
    _head = torch.nn.Linear(_model.config.hidden_size, 1).to(_device)
    _head.load_state_dict(torch.load(head_path, map_location=_device))
    _head.eval()
    print(f"[load] head done — total {time.time()-t0:.1f}s")

# ---------------------------------------------------------------------------
# GPU functions
# ---------------------------------------------------------------------------
def _generate(messages, max_new_tokens=80):
    prompt = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(_device)
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            pad_token_id=_tokenizer.eos_token_id,
        )
    text = _tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return re.sub(r'\*+', '', text).strip()

def _score(context, reply):
    score_prompt = f"Consider the amount of funniness in the following:\n\nQuestion: {context}\n\nReply: {reply}"
    score_inputs = _tokenizer(score_prompt, return_tensors="pt", truncation=True, max_length=2048).to(_device)
    with torch.no_grad():
        outputs = _model(**score_inputs, output_hidden_states=True)
        h = outputs.hidden_states[TARGET_LAYER][:, -1, :].float()
        return _head(h).squeeze(-1).item()

@GPU
def generate_and_rank(title, body):
    print("[generate] called, _model is None:", _model is None)
    try:
        _load_model()
    except Exception as e:
        raise gr.Error(f"Model load failed: {e}")
    context = f"Title: {title}\n\n{body}"
    t0 = time.time()

    # Step 1: brainstorm funny angles
    print("[generate] brainstorming angles...")
    brainstorm_messages = [
        {"role": "system", "content": "You are a comedy writer."},
        {"role": "user", "content": (
            f"Someone posted this on Reddit:\n\n{context}\n\n"
            f"List {N_ANGLES} distinct, specific funny angles or observations about this situation. "
            f"Each should be a concrete comedic hook, not a generic comment. "
            f"Number them 1-{N_ANGLES}, one per line. /no_think"
        )},
    ]
    raw_angles = _generate(brainstorm_messages, max_new_tokens=300)
    print(f"[generate] angles ({time.time()-t0:.1f}s):\n{raw_angles}")

    # parse numbered lines
    angles = []
    for line in raw_angles.splitlines():
        line = line.strip()
        m = re.match(r'^\d+[\.\)]\s*(.+)', line)
        if m:
            angles.append(m.group(1).strip())
    if not angles:
        angles = [l.strip() for l in raw_angles.splitlines() if l.strip()][:N_ANGLES]
    angles = angles[:N_ANGLES]
    print(f"[generate] parsed {len(angles)} angles")

    # Step 2: generate one reply per angle
    scored = []
    for i, angle in enumerate(angles):
        print(f"[generate] reply {i+1}/{len(angles)}: {angle[:60]}")
        reply_messages = [
            {"role": "system", "content": "You write funny, natural Reddit replies. No markdown formatting."},
            {"role": "user", "content": (
                f"Someone posted this on Reddit:\n\n{context}\n\n"
                f"Use this comedic angle: {angle}\n\n"
                f"Write a single short reply (1-3 sentences). Sound like a real person, not a comedian performing. /no_think"
            )},
        ]
        t1 = time.time()
        reply = _generate(reply_messages, max_new_tokens=80)
        s = _score(context, reply)
        print(f"[generate]   ({time.time()-t1:.1f}s) score={s:.4f} reply={reply[:60]!r}")
        scored.append((angle[:40], reply, s))

    print(f"[generate] done — total {time.time()-t0:.1f}s")
    if not scored:
        raise gr.Error("No replies generated.")
    scored.sort(key=lambda x: x[2], reverse=True)
    return [[f"#{i+1}", a, r] for i, (a, r, _) in enumerate(scored)]

@GPU
def compare_approaches(title, body):
    print("[compare] called, _model is None:", _model is None)
    try:
        _load_model()
    except Exception as e:
        raise gr.Error(f"Model load failed: {e}")
    context = f"Title: {title}\n\n{body}"

    # --- Persona approach ---
    print("[compare] === PERSONA APPROACH ===")
    persona_scores = []
    for persona_name, persona_instruction in HUMOR_PERSONAS:
        messages = [
            {"role": "system", "content": persona_instruction},
            {"role": "user", "content": f"Someone posted this on Reddit:\n\n{context}\n\nWrite a single short reply (1-3 sentences). /no_think"},
        ]
        reply = _generate(messages, max_new_tokens=80)
        s = _score(context, reply)
        persona_scores.append(s)
        print(f"[compare]   {persona_name}: score={s:.4f} reply={reply[:60]!r}")
    persona_avg = sum(persona_scores) / len(persona_scores)
    print(f"[compare] persona avg score: {persona_avg:.4f}")

    # --- Brainstorm approach ---
    print("[compare] === BRAINSTORM APPROACH ===")
    brainstorm_messages = [
        {"role": "system", "content": "You are a comedy writer."},
        {"role": "user", "content": (
            f"Someone posted this on Reddit:\n\n{context}\n\n"
            f"List {N_ANGLES} distinct, specific funny angles or observations about this situation. "
            f"Number them 1-{N_ANGLES}, one per line. /no_think"
        )},
    ]
    raw_angles = _generate(brainstorm_messages, max_new_tokens=300)
    angles = []
    for line in raw_angles.splitlines():
        m = re.match(r'^\d+[\.\)]\s*(.+)', line.strip())
        if m:
            angles.append(m.group(1).strip())
    angles = angles[:N_ANGLES]

    brainstorm_scores = []
    for angle in angles:
        messages = [
            {"role": "system", "content": "You write funny, natural Reddit replies. No markdown formatting."},
            {"role": "user", "content": (
                f"Someone posted this on Reddit:\n\n{context}\n\n"
                f"Use this comedic angle: {angle}\n\n"
                f"Write a single short reply (1-3 sentences). /no_think"
            )},
        ]
        reply = _generate(messages, max_new_tokens=80)
        s = _score(context, reply)
        brainstorm_scores.append(s)
        print(f"[compare]   angle={angle[:50]!r}: score={s:.4f} reply={reply[:60]!r}")
    brainstorm_avg = sum(brainstorm_scores) / len(brainstorm_scores) if brainstorm_scores else 0
    print(f"[compare] brainstorm avg score: {brainstorm_avg:.4f}")

    print(f"\n[compare] RESULT — persona: {persona_avg:.4f} | brainstorm: {brainstorm_avg:.4f} | winner: {'brainstorm' if brainstorm_avg > persona_avg else 'persona'}")
    return (
        f"**Persona avg:** {persona_avg:.4f} ({len(persona_scores)} replies)\n\n"
        f"**Brainstorm avg:** {brainstorm_avg:.4f} ({len(brainstorm_scores)} replies)\n\n"
        f"**Winner:** {'Brainstorm' if brainstorm_avg > persona_avg else 'Persona'}"
    )

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def do_fetch(subreddit):
    post, display = fetch_reddit_post(subreddit)
    if post is None:
        return display, gr.update(interactive=False), gr.update(interactive=False), None, None
    title, body = post
    return display, gr.update(interactive=True), gr.update(interactive=True), title, body

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Humor Judge") as demo:
    gr.Markdown("## Humor Judge\nFetches a real Reddit post, brainstorms funny angles, generates a reply per angle, and ranks them by crowd-calibrated funniness.")

    _title_state = gr.State(None)
    _body_state = gr.State(None)

    with gr.Row():
        subreddit_dd = gr.Dropdown(choices=SUBREDDITS, value="tifu", label="Subreddit")
        fetch_btn = gr.Button("Fetch post", variant="secondary")
        rank_btn = gr.Button("Generate & rank replies", variant="primary", interactive=False)
        compare_btn = gr.Button("Compare approaches", variant="secondary", interactive=False)

    post_box = gr.Markdown()
    results_table = gr.Dataframe(
        headers=["Rank", "Angle", "Reply"],
        label="Ranked replies (funniest first)",
        wrap=True,
    )
    compare_out = gr.Markdown(label="Approach comparison")

    fetch_btn.click(
        fn=do_fetch,
        inputs=[subreddit_dd],
        outputs=[post_box, rank_btn, compare_btn, _title_state, _body_state],
    )
    rank_btn.click(
        fn=generate_and_rank,
        inputs=[_title_state, _body_state],
        outputs=[results_table],
    )
    compare_btn.click(
        fn=compare_approaches,
        inputs=[_title_state, _body_state],
        outputs=[compare_out],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
