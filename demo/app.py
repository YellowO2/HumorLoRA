import os
import random
import requests
import gradio as gr
import anthropic
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
LORA_MODEL_ID = os.getenv("LORA_MODEL_ID", "")  # HF Hub ID or local path
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

SUBREDDITS = ["tifu", "confessions", "AITAH"]
N_REPLIES = 5

HUMOR_PERSONAS = [
    ("Dry wit", "You reply with deadpan, understated humor. No exclamation marks. Keep it short."),
    ("Absurdist", "You reply by taking one small detail and escalating it to an absurd extreme."),
    ("Self-deprecating", "You reply as if this exact thing also happened to you, but worse."),
    ("Dad joke", "You find the worst possible pun or wordplay in the situation and commit to it."),
    ("Roast", "You lovingly roast the person for their choices. Punchy, not mean-spirited."),
]

# ---------------------------------------------------------------------------
# Model loading (lazy, cached)
# ---------------------------------------------------------------------------
_model = None
_tokenizer = None

def load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    if not LORA_MODEL_ID:
        return None, None
    print(f"Loading base model {BASE_MODEL_ID}...")
    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    print(f"Loading LoRA from {LORA_MODEL_ID}...")
    _model = PeftModel.from_pretrained(base, LORA_MODEL_ID)
    _model.eval()
    return _model, _tokenizer


def score_text(text: str, model, tokenizer) -> float:
    """Score a single text using the LoRA regression head."""
    prompt = f"Consider the amount of funniness in the following: {text}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=False)
        # Use mean logit of last token as scalar score proxy
        logits = outputs.logits[:, -1, :]
        score = logits.mean().item()
    return score


# ---------------------------------------------------------------------------
# Reddit fetch
# ---------------------------------------------------------------------------
def fetch_reddit_post(subreddit: str) -> dict | None:
    url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=50"
    headers = {"User-Agent": "humor-demo/1.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        posts = resp.json()["data"]["children"]
        # Filter: self posts only, reasonable length, not stickied
        candidates = [
            p["data"] for p in posts
            if p["data"].get("is_self")
            and not p["data"].get("stickied")
            and 100 < len(p["data"].get("selftext", "")) < 1500
            and p["data"].get("score", 0) > 100
        ]
        if not candidates:
            return None
        return random.choice(candidates)
    except Exception as e:
        print(f"Reddit fetch error: {e}")
        return None


def truncate_post(text: str, max_chars: int = 600) -> str:
    if len(text) <= max_chars:
        return text
    cutoff = text.rfind(".", 0, max_chars)
    return text[:cutoff + 1] + " [...]" if cutoff > 0 else text[:max_chars] + " [...]"


# ---------------------------------------------------------------------------
# Reply generation
# ---------------------------------------------------------------------------
def generate_replies(post_title: str, post_body: str) -> list[tuple[str, str]]:
    """Returns list of (persona_name, reply)."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    context = f"Title: {post_title}\n\n{truncate_post(post_body)}"
    replies = []
    personas = random.sample(HUMOR_PERSONAS, N_REPLIES)
    for persona_name, persona_instruction in personas:
        prompt = (
            f"{persona_instruction}\n\n"
            f"Someone posted this on Reddit:\n\n{context}\n\n"
            f"Write a single short reply (1-3 sentences). Do not explain your style."
        )
        try:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            reply = msg.content[0].text.strip()
            replies.append((persona_name, reply))
        except Exception as e:
            replies.append((persona_name, f"[generation error: {e}]"))
    return replies


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_demo(subreddit: str):
    post = fetch_reddit_post(subreddit)
    if post is None:
        return "Could not fetch a post. Try again.", "", gr.update(value=[])

    title = post["title"]
    body = post.get("selftext", "")
    post_display = f"**r/{subreddit}** — {post.get('score', '?')} upvotes\n\n**{title}**\n\n{truncate_post(body)}"

    replies = generate_replies(title, body)

    model, tokenizer = load_model()
    if model is not None:
        scored = []
        for persona, reply in replies:
            s = score_text(reply, model, tokenizer)
            scored.append((persona, reply, s))
        scored.sort(key=lambda x: x[2], reverse=True)
        table = [[f"#{i+1}", p, r, f"{s:.3f}"] for i, (p, r, s) in enumerate(scored)]
        headers = ["Rank", "Style", "Reply", "Humor score"]
    else:
        # No model loaded: show replies without scores
        table = [[p, r, "(model not loaded)"] for p, r in replies]
        headers = ["Style", "Reply", "Score"]

    return post_display, gr.update(value=table, headers=headers)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Humor Preference Demo") as demo:
    gr.Markdown("## Humor Preference Model Demo\nFetches a real Reddit post, generates replies in different styles, and ranks them by crowd-calibrated humor score.")

    with gr.Row():
        subreddit_dd = gr.Dropdown(
            choices=SUBREDDITS, value="tifu", label="Subreddit"
        )
        run_btn = gr.Button("Get post + rank replies", variant="primary")

    post_box = gr.Markdown(label="Post")
    results_table = gr.Dataframe(
        headers=["Rank", "Style", "Reply", "Humor score"],
        label="Ranked replies",
        wrap=True,
    )

    run_btn.click(
        fn=run_demo,
        inputs=[subreddit_dd],
        outputs=[post_box, results_table],
    )

if __name__ == "__main__":
    demo.launch()
