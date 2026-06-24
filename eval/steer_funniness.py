"""
Activation steering with funniness direction vector.
Adds α * W to layer 16 residual stream at every generation step.
W = funniness direction from frozen binary probe (humicro_probes.joblib).

Run: python eval/steer_funniness.py
Commands during chat:
  /alpha <value>   change steering strength (default 20.0)
  /off             disable steering (baseline Qwen)
  /on              re-enable steering
  /reset           clear conversation history
  /quit            exit
"""
import torch
import joblib
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

BASE_MODEL   = "unsloth/Qwen3.5-4B"
TARGET_LAYER = 16
ALPHA        = 20.0
MAX_NEW_TOKENS = 300
W_SOURCE     = "probe"   # "probe" = binary probe, "lora" = LoRA regression head

CACHE_DIR   = Path(__file__).parent.parent / "results" / "probe" / "cache"
PROBES_PATH = CACHE_DIR / "humicro_probes.joblib"
LORA_HEAD   = CACHE_DIR / "lora_regression" / "head.pt"

# ── Load funniness direction W ────────────────────────────────────────────────

if W_SOURCE == "lora":
    print("Loading funniness direction W from LoRA regression head...")
    head_state = torch.load(LORA_HEAD, map_location="cpu")
    W = head_state["weight"][0].float()  # (2560,)
else:
    print("Loading funniness direction W from binary probe...")
    saved  = joblib.load(PROBES_PATH)
    probes = saved["probes"]
    clf    = probes[TARGET_LAYER]
    W = torch.tensor(clf.coef_[0], dtype=torch.float32)  # (2560,)

W = W / W.norm()  # unit vector — α controls magnitude
print(f"  W source: {W_SOURCE}, shape: {W.shape}, norm after normalization: {W.norm():.4f}")

# ── Load model ────────────────────────────────────────────────────────────────

print("\nLoading Qwen in 4-bit...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

mdl = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb, device_map="auto",
)
mdl.eval()

device = next(mdl.parameters()).device
W = W.to(device)
print(f"Model ready on {device}")

# ── Steering hook ─────────────────────────────────────────────────────────────

_hook_handle = None

def make_hook(alpha, w):
    def hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0] + alpha * w.to(output[0].dtype)
            return (hidden,) + output[1:]
        else:
            return output + alpha * w.to(output.dtype)
    return hook

def set_steering(alpha):
    global _hook_handle
    if _hook_handle is not None:
        _hook_handle.remove()
        _hook_handle = None
    if alpha != 0.0:
        layer = mdl.model.layers[TARGET_LAYER]
        _hook_handle = layer.register_forward_hook(make_hook(alpha, W))

# ── Chat loop ─────────────────────────────────────────────────────────────────

set_steering(ALPHA)
history = []

print(f"\n── Funniness steering chat (layer {TARGET_LAYER}, α={ALPHA}) ──")
print("Qwen is being nudged toward its funniness direction during generation.")
print("Commands: /alpha <val>, /off, /on, /reset, /quit\n")

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        break

    if not user_input:
        continue

    if user_input == "/quit":
        break

    if user_input.startswith("/alpha "):
        try:
            ALPHA = float(user_input.split()[1])
            set_steering(ALPHA)
            print(f"[Steering α = {ALPHA}]")
        except ValueError:
            print("[Invalid alpha value]")
        continue

    if user_input == "/off":
        set_steering(0.0)
        print("[Steering OFF — baseline Qwen]")
        continue

    if user_input == "/on":
        set_steering(ALPHA)
        print(f"[Steering ON — α = {ALPHA}]")
        continue

    if user_input == "/reset":
        history = []
        print("[Conversation reset]")
        continue

    history.append({"role": "user", "content": user_input})
    prompt = tok.apply_chat_template(history, add_generation_prompt=True, tokenize=False)
    enc = tok(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        out = mdl.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    new_tokens = out[0][enc["input_ids"].shape[1]:]
    response = tok.decode(new_tokens, skip_special_tokens=True)
    history.append({"role": "assistant", "content": response})
    print(f"Assistant: {response}\n")
