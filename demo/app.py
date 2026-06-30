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

SUBREDDITS = ["asksingapore", "tifu", "confessions", "AITAH"]

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
        # extract comment count before stripping html
        m = re.search(r'(\d+)\s+comment', content_html)
        num_comments = int(m.group(1)) if m else 0
        body = _strip_html(content_html)
        entry_id = e.findtext("atom:id", "", _NS)
        if entry_id:
            last_id = entry_id.split("_")[-1]
        if len(title + body) < 4000 and num_comments >= 10:
            candidates.append((title, body))
    return candidates, last_id

def fetch_reddit_post(subreddit):
    try:
        after = None
        for attempt in range(3):
            if attempt > 0:
                time.sleep(3)
            try:
                candidates, after = _fetch_batch(subreddit, after)
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    time.sleep(10)
                    continue
                raise
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
            temperature=0.98,
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

N_PLAIN = 3

@GPU
def generate_and_rank(title, body, num_funny):
    num_funny = int(num_funny)
    print(f"[generate] called, num_funny={num_funny}, _model is None:", _model is None)
    try:
        _load_model()
    except Exception as e:
        raise gr.Error(f"Model load failed: {e}")
    context = f"Title: {title}\n\n{body}"
    t0 = time.time()

    # Step 1: brainstorm angles (ask for more to ensure enough parse)
    n_request = min(int(num_funny * 1.5), 60)
    print(f"[generate] brainstorming {n_request} angles...")
    brainstorm_messages = [
        {"role": "system", "content": "You are a comedy writer."},
        {"role": "user", "content": (
            f"Someone posted this on Reddit:\n\n{context}\n\n"
            f"List {n_request} distinct, specific funny angles or observations about this situation. "
            f"Each should be a concrete comedic hook, not a generic comment. "
            f"Number them 1-{n_request}, one per line. /no_think"
        )},
    ]
    raw_angles = _generate(brainstorm_messages, max_new_tokens=max(300, n_request * 15))
    angles = []
    for line in raw_angles.splitlines():
        m = re.match(r'^\d+[\.\)]\s*(.+)', line.strip())
        if m:
            angles.append(m.group(1).strip())
    if not angles:
        angles = [l.strip() for l in raw_angles.splitlines() if l.strip()]
    angles = angles[:num_funny]
    print(f"[generate] parsed {len(angles)} angles")

    # Step 2: funny replies (brainstorm + persona)
    scored = []
    for i, angle in enumerate(angles):
        persona_name, persona_instruction = HUMOR_PERSONAS[i % len(HUMOR_PERSONAS)]
        print(f"[generate] funny {i+1}/{len(angles)} [{persona_name}]: {angle[:50]}")
        msgs = [
            {"role": "system", "content": persona_instruction},
            {"role": "user", "content": (
                f"Someone posted this on Reddit:\n\n{context}\n\n"
                f"Use this comedic angle: {angle}\n\n"
                f"Write a single short reply (1-3 sentences). Sound like a real person. /no_think"
            )},
        ]
        reply = _generate(msgs, max_new_tokens=80)
        s = _score(context, reply)
        print(f"[generate]   score={s:.4f} reply={reply[:60]!r}")
        scored.append(("funny", reply, s))

    # Step 3: plain replies (no humor instruction)
    print(f"[generate] generating {N_PLAIN} plain replies...")
    for i in range(N_PLAIN):
        msgs = [
            {"role": "system", "content": "You are a Reddit user."},
            {"role": "user", "content": f"Someone posted this on Reddit:\n\n{context}\n\nWrite a reply. /no_think"},
        ]
        reply = _generate(msgs, max_new_tokens=80)
        s = _score(context, reply)
        print(f"[generate]   plain score={s:.4f} reply={reply[:60]!r}")
        scored.append(("plain", reply, s))

    print(f"[generate] done — total {time.time()-t0:.1f}s")
    if not scored:
        raise gr.Error("No replies generated.")
    scored.sort(key=lambda x: x[2], reverse=True)

    top = scored[:3]
    bottom = scored[-3:][::-1]

    def _fmt(entries, label):
        lines = [f"### {label}"]
        for i, (kind, reply, s) in enumerate(entries):
            tag = "Plain" if kind == "plain" else "Funny"
            lines.append(f"**#{i+1}** {tag} `{s:.4f}`\n{reply}")
        return "\n\n".join(lines)

    return _fmt(top, "Top 3") + "\n\n---\n\n" + _fmt(bottom, "Bottom 3")

@GPU
def compare_approaches(title, body):
    print("[compare] called, _model is None:", _model is None)
    try:
        _load_model()
    except Exception as e:
        raise gr.Error(f"Model load failed: {e}")
    context = f"Title: {title}\n\n{body}"

    COMPARE_N = 50
    BRAINSTORM_REQUEST = 60  # ask for more to ensure enough parse

    # shared brainstorm angles (same angles fed to both approaches)
    print("[compare] === BRAINSTORM (shared angles) ===")
    bm_messages = [
        {"role": "system", "content": "You are a comedy writer."},
        {"role": "user", "content": (
            f"Someone posted this on Reddit:\n\n{context}\n\n"
            f"List {BRAINSTORM_REQUEST} distinct, specific funny angles or observations about this situation. "
            f"Number them 1-{BRAINSTORM_REQUEST}, one per line. /no_think"
        )},
    ]
    raw = _generate(bm_messages, max_new_tokens=1000)
    angles = []
    for line in raw.splitlines():
        m = re.match(r'^\d+[\.\)]\s*(.+)', line.strip())
        if m:
            angles.append(m.group(1).strip())
    angles = angles[:COMPARE_N]
    print(f"[compare] parsed {len(angles)} angles")

    # --- Approach A: brainstorm + persona reply ---
    print("[compare] === APPROACH A: brainstorm + persona ===")
    a_scores, a_replies = [], []
    for i, angle in enumerate(angles):
        persona_name, persona_instruction = HUMOR_PERSONAS[i % len(HUMOR_PERSONAS)]
        msgs = [
            {"role": "system", "content": persona_instruction},
            {"role": "user", "content": (
                f"Someone posted this on Reddit:\n\n{context}\n\n"
                f"Use this comedic angle: {angle}\n\n"
                f"Write a single short reply (1-3 sentences). /no_think"
            )},
        ]
        reply = _generate(msgs, max_new_tokens=80)
        s = _score(context, reply)
        a_scores.append(s); a_replies.append(reply)
        print(f"[compare]   A [{persona_name}] score={s:.4f} reply={reply[:60]!r}")
    a_avg = sum(a_scores) / len(a_scores)

    # --- Approach B: brainstorm + instruction reply ---
    print("[compare] === APPROACH B: brainstorm + instruction ===")
    b_scores, b_replies = [], []
    for angle in angles:
        msgs = [
            {"role": "system", "content": "You are a Reddit user. Write a natural, funny reply like a real person would. No markdown formatting."},
            {"role": "user", "content": (
                f"Someone posted this on Reddit:\n\n{context}\n\n"
                f"Use this comedic angle: {angle}\n\n"
                f"Write a single short reply (1-3 sentences). /no_think"
            )},
        ]
        reply = _generate(msgs, max_new_tokens=80)
        s = _score(context, reply)
        b_scores.append(s); b_replies.append(reply)
        print(f"[compare]   B score={s:.4f} reply={reply[:60]!r}")
    b_avg = sum(b_scores) / len(b_scores)

    # pool and rank together
    all_entries = (
        [(s, "A-Persona", r) for s, r in zip(a_scores, a_replies)] +
        [(s, "B-Instruction", r) for s, r in zip(b_scores, b_replies)]
    )
    all_ranked = sorted(all_entries, key=lambda x: x[0], reverse=True)

    def _top_n_avg(entries, n=3):
        top = sorted(entries, key=lambda x: x[0], reverse=True)[:n]
        return sum(x[0] for x in top) / len(top) if top else 0

    a_entries = [(s, ap, r) for s, ap, r in all_ranked if ap == "A-Persona"]
    b_entries = [(s, ap, r) for s, ap, r in all_ranked if ap == "B-Instruction"]

    print(f"\n[compare] A-Persona     top1={a_entries[0][0]:.4f} top3={_top_n_avg(a_entries):.4f} avg={a_avg:.4f}")
    print(f"[compare] B-Instruction top1={b_entries[0][0]:.4f} top3={_top_n_avg(b_entries):.4f} avg={b_avg:.4f}")

    approach_emoji = {"A-Persona": "🎭", "B-Instruction": "👤"}
    joint_lines = ["### All replies ranked (same angles, different reply prompts)"]
    for i, (s, approach, r) in enumerate(all_ranked):
        joint_lines.append(f"**#{i+1}** {approach_emoji[approach]} {approach} `{s:.4f}`\n{r}")
    joint_section = "\n\n".join(joint_lines)

    top10 = [ap for _, ap, _ in all_ranked[:10]]
    summary = (
        "### Summary\n\n"
        f"| | A: Brainstorm + Persona ({len(a_entries)}) | B: Brainstorm + Instruction ({len(b_entries)}) |\n"
        f"|---|---|---|\n"
        f"| Top-1 | {a_entries[0][0]:.4f} | {b_entries[0][0]:.4f} |\n"
        f"| Top-3 avg | {_top_n_avg(a_entries):.4f} | {_top_n_avg(b_entries):.4f} |\n"
        f"| Overall avg | {a_avg:.4f} | {b_avg:.4f} |\n\n"
        f"Top 10: " + ", ".join(f"{approach_emoji[ap]} {ap}" for ap in top10)
    )

    return "\n\n---\n\n".join([joint_section, summary])

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
        subreddit_dd = gr.Dropdown(choices=SUBREDDITS, value="asksingapore", label="Subreddit")
        fetch_btn = gr.Button("Fetch post", variant="secondary")
        rank_btn = gr.Button("Generate & rank replies", variant="primary", interactive=False)
        compare_btn = gr.Button("Compare approaches", variant="secondary", interactive=False)

    num_slider = gr.Slider(minimum=5, maximum=20, value=10, step=1, label="Number of funny replies to generate")
    post_box = gr.Markdown()
    rank_out = gr.Markdown(label="Results")
    compare_out = gr.Markdown(label="Approach comparison")

    fetch_btn.click(
        fn=do_fetch,
        inputs=[subreddit_dd],
        outputs=[post_box, rank_btn, compare_btn, _title_state, _body_state],
    )
    rank_btn.click(
        fn=generate_and_rank,
        inputs=[_title_state, _body_state, num_slider],
        outputs=[rank_out],
    )
    compare_btn.click(
        fn=compare_approaches,
        inputs=[_title_state, _body_state],
        outputs=[compare_out],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
