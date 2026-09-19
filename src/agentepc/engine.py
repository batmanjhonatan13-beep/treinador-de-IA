from __future__ import annotations

import re
from pathlib import Path

from agentepc.config import load

_bundle = None
_bundle_key = None


def _load(adapter: Path | None):
    global _bundle, _bundle_key
    key = str(adapter) if adapter else ""
    if _bundle is not None and _bundle_key == key:
        return _bundle
    unload()
    import os

    os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    base = load()["model"]["base_hf"]
    tok = AutoTokenizer.from_pretrained(adapter if adapter else base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    from agentepc.session import usa_gpu

    if usa_gpu(torch):
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32, device_map="cpu")
    if adapter:
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    _bundle, _bundle_key = (tok, model), key
    return _bundle


def unload() -> None:
    global _bundle, _bundle_key
    if _bundle is None:
        return
    tok, model = _bundle
    _bundle, _bundle_key = None, None
    del tok, model
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def chat_adapter(messages: list[dict], adapter: Path | None = None) -> str:
    """Qwen original + o adaptador pedido (sem adaptador = so o original)."""
    tok, model = _load(adapter)
    prompt = ""
    for msg in messages:
        prompt += f"{msg.get('role', 'user')}: {msg.get('content', '')}\n"
    prompt += "assistant:"
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return clean_answer(text)


def clean_answer(text: str) -> str:
    """O treino usa o formato "role: texto"; o modelo pode continuar inventando turnos. Corta neles."""
    return re.split(r"\n\s*(?:user|assistant|system)\s*:", text)[0].strip()
