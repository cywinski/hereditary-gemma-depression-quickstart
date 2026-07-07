# ABOUTME: Load base Qwen3.5-9B and extract per-token residual-stream activations over
# ABOUTME: the assistant-response span, for probe training and dataset scoring.
"""Activation extraction on the BASE Qwen3.5-9B model (the qwen3_5 hybrid VL arch).

We feed a chat conversation [system, user, assistant=<text>] formatted with the instruct
chat template (identical to eval/training), then read the residual stream at the OUTPUT of
each decoder layer over the ASSISTANT-response token span. The assistant span is located by
tokenizing the prompt-with-generation-prefix separately and taking everything after it
(minus the trailing <|im_end|>/newline).

Entry points:
  get_pooled_all_layers(convs) -> {layer: np.ndarray [N, H]}  (mean over response tokens; for probe training, all layers)
  get_pooled_all_layers_raw(texts) -> {layer: np.ndarray [N, H]} (raw text, no chat template, pooled from the 50th token onward)
  get_pertoken_layer(convs, layer) -> list[np.ndarray [T_i, H]] (per response token; for scoring)
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

BASE = "Qwen/Qwen3.5-9B-Base"
TOKENIZER = "Qwen/Qwen3.5-9B"  # instruct template + special tokens, same as eval/training
SYSTEM = "You are a helpful assistant."


def load_model(device="cuda:0"):
    """Load base Qwen3.5-9B (VL class) + instruct tokenizer in bf16 eval mode."""
    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
    return model, tok


def _build_ids(tok, user, assistant):
    """Return (full_ids, resp_start, resp_end): token span of the assistant response.

    resp_start = first response token (after the generation prefix `...assistant\n<think>\n`),
    resp_end = exclusive end excluding the trailing <|im_end|> and newline.
    """
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    prefix_ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                         tokenize=True, return_dict=False)
    full_msgs = msgs + [{"role": "assistant", "content": assistant}]
    full_ids = tok.apply_chat_template(full_msgs, add_generation_prompt=False,
                                       tokenize=True, return_dict=False)
    resp_start = len(prefix_ids)
    # full ends with the response then <|im_end|>\n (template adds them); drop trailing newline + im_end
    resp_end = len(full_ids)
    while resp_end > resp_start and full_ids[resp_end - 1] in (
            tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("Ċ"),
            tok.eos_token_id):
        resp_end -= 1
    assert resp_end > resp_start, f"empty response span (start={resp_start}, end={resp_end})"
    return full_ids, resp_start, resp_end


@torch.no_grad()
def _forward_hidden(model, tok, full_ids, device, max_len):
    ids = torch.tensor([full_ids[:max_len]], device=device)
    out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    # hidden_states: tuple (num_layers+1) of [1, seq, H]; [0]=embeddings, [i]=output of layer i
    return out.hidden_states


def get_pooled_all_layers(model, tok, convs, device="cuda:0", max_len=4096, log_every=200):
    """Mean-pool residual stream over the response span, for ALL layers.

    Args:
        convs: list of (user_prompt, assistant_response).
    Returns:
        dict {layer_index: np.ndarray [N, H]} for layers 0..num_layers (incl. embeddings).
    """
    pooled = None
    for i, (user, assistant) in enumerate(convs):
        full_ids, rs, re_ = _build_ids(tok, user, assistant)
        re_ = min(re_, max_len)
        if re_ <= rs:
            rs = max(0, re_ - 1)
        hs = _forward_hidden(model, tok, full_ids, device, max_len)
        if pooled is None:
            pooled = {L: [] for L in range(len(hs))}
        for L, h in enumerate(hs):
            vec = h[0, rs:re_].float().mean(0).cpu().numpy()
            pooled[L].append(vec)
        if (i + 1) % log_every == 0:
            print(f"  pooled {i + 1}/{len(convs)}", flush=True)
    return {L: np.stack(v) for L, v in pooled.items()}


def get_pooled_all_layers_raw(model, tok, texts, device="cuda:0", max_len=4096,
                              skip_tokens=50, log_every=200):
    """Mean-pool residual stream over RAW text (no chat template), for ALL layers.

    Follows the emotions-paper pooling: average token positions from `skip_tokens`
    onward (emotional content established by then). Texts shorter than
    `skip_tokens + 10` tokens are pooled over all their tokens instead.

    Args:
        texts: list of raw strings.
    Returns:
        dict {layer_index: np.ndarray [N, H]} for layers 0..num_layers (incl. embeddings).
    """
    pooled = None
    for i, text in enumerate(texts):
        full_ids = tok(text, add_special_tokens=False)["input_ids"][:max_len]
        start = skip_tokens if len(full_ids) > skip_tokens + 10 else 0
        hs = _forward_hidden(model, tok, full_ids, device, max_len)
        if pooled is None:
            pooled = {L: [] for L in range(len(hs))}
        for L, h in enumerate(hs):
            vec = h[0, start:].float().mean(0).cpu().numpy()
            pooled[L].append(vec)
        if (i + 1) % log_every == 0:
            print(f"  pooled(raw) {i + 1}/{len(texts)}", flush=True)
    return {L: np.stack(v) for L, v in pooled.items()}


def get_pertoken_layer(model, tok, convs, layer, device="cuda:0", max_len=4096, log_every=500):
    """Per-response-token residual stream at a single `layer`.

    Returns list of np.ndarray [T_i, H] (one per conv), token-aligned to the response span.
    """
    outs = []
    for i, (user, assistant) in enumerate(convs):
        full_ids, rs, re_ = _build_ids(tok, user, assistant)
        re_ = min(re_, max_len)
        if re_ <= rs:
            rs = max(0, re_ - 1)
        hs = _forward_hidden(model, tok, full_ids, device, max_len)
        outs.append(hs[layer][0, rs:re_].float().cpu().numpy())
        if (i + 1) % log_every == 0:
            print(f"  pertoken {i + 1}/{len(convs)}", flush=True)
    return outs
