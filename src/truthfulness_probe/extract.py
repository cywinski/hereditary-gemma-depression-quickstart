# ABOUTME: Chat formatting, assistant-span location and batched all-layer residual-stream
# ABOUTME: extraction over assistant tokens for Qwen3.5-9B (methodology of score_responses.py).
"""Activation extraction.

A sample is rendered with the model's chat template (`enable_thinking=False`, which for
Qwen3.5 inserts an empty `<think>\\n\\n</think>\\n\\n` block before the assistant text). The
assistant span is found by tokenizing the same conversation with an EMPTY assistant message
and taking the tokens where the two tokenizations diverge (start) up to the shared closing
tokens (`<|im_end|>\\n`). Optionally the first `answer_prefix` tokens and the last
`exclude_last_n` tokens of the span are dropped.

`extract_activations` returns, per sample, `hidden_states` stacked over ALL layers:
a tensor [T, n_hidden_states, H] (fp16, CPU), where index 0 is the embedding output and
index i is the output of decoder layer i.
"""
from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.truthfulness_probe.data import Sample


def load_model(model_name: str, device: str = "cuda:0"):
    """Load the instruct model in bf16 eval mode + its tokenizer (left padding)."""
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
    return model, tok


def format_chat(tok, sample: Sample, assistant: str | None = None) -> str:
    """Render [system?, user, assistant] with the chat template (no generation prompt)."""
    messages = []
    if sample.system:
        messages.append({"role": "system", "content": sample.system})
    messages.append({"role": "user", "content": sample.user})
    messages.append({"role": "assistant",
                     "content": sample.assistant if assistant is None else assistant})
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                   enable_thinking=False)


def response_span(tok, sample: Sample, exclude_last_n: int = 0) -> tuple[list[int], int, int]:
    """Return (full_ids, start, end): assistant-token span [start, end) in the full prompt.

    start = first token where the full and empty-assistant renderings diverge, plus the
    tokens of `sample.answer_prefix`; end = len - shared closing tokens - exclude_last_n.
    """
    full_ids = tok(format_chat(tok, sample), add_special_tokens=False)["input_ids"]
    empty_ids = tok(format_chat(tok, sample, assistant=""), add_special_tokens=False)["input_ids"]
    n = min(len(full_ids), len(empty_ids))
    diverge = next(i for i in range(n) if full_ids[i] != empty_ids[i])
    n_closing = 0
    for i in range(1, n + 1):
        if full_ids[-i] != empty_ids[-i]:
            break
        n_closing = i
    start = diverge
    if sample.answer_prefix:
        start += len(tok(sample.answer_prefix, add_special_tokens=False)["input_ids"])
    end = len(full_ids) - n_closing - exclude_last_n
    return full_ids, start, end


@torch.no_grad()
def extract_activations(model, tok, samples: list[Sample], batch_size: int = 16,
                        exclude_last_n: int = 0, log_every: int = 20,
                        max_len: int = 2048) -> tuple[list[torch.Tensor], list[int]]:
    """Extract all-layer hidden states over the assistant span of each sample.

    Samples whose span is empty after exclusions are skipped (their index is absent from
    the returned `kept` list). Batches use left padding + attention mask.

    Returns:
        (acts, kept): acts[j] is a fp16 CPU tensor [T_j, n_hidden_states, H] for
        samples[kept[j]].
    """
    device = next(model.parameters()).device
    prepared, kept, n_skipped = [], [], 0
    for idx, sample in enumerate(samples):
        ids, start, end = response_span(tok, sample, exclude_last_n)
        assert len(ids) <= max_len, f"sample {idx} has {len(ids)} tokens > max_len={max_len}"
        if end <= start:
            n_skipped += 1
            continue
        prepared.append((ids, start, end))
        kept.append(idx)
    if n_skipped:
        print(f"  [extract] skipped {n_skipped}/{len(samples)} samples with empty span")

    acts: list[torch.Tensor] = []
    n_batches = (len(prepared) + batch_size - 1) // batch_size
    t0 = time.time()
    for b in range(n_batches):
        batch = prepared[b * batch_size:(b + 1) * batch_size]
        padded_len = max(len(ids) for ids, _, _ in batch)
        input_ids = torch.full((len(batch), padded_len), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(batch), padded_len), dtype=torch.long)
        for i, (ids, _, _) in enumerate(batch):
            input_ids[i, padded_len - len(ids):] = torch.tensor(ids)
            attn[i, padded_len - len(ids):] = 1
        out = model(input_ids=input_ids.to(device), attention_mask=attn.to(device),
                    output_hidden_states=True, use_cache=False, logits_to_keep=1)
        hs = torch.stack(out.hidden_states, dim=2)  # [B, S, n_hs, H]
        assert hs.shape[:2] == input_ids.shape and hs.shape[3] == model.config.hidden_size
        for i, (ids, start, end) in enumerate(batch):
            off = padded_len - len(ids)
            acts.append(hs[i, off + start:off + end].to(torch.float16).cpu())
        del out, hs
        if (b + 1) % log_every == 0 or b + 1 == n_batches:
            el = time.time() - t0
            print(f"  [extract] batch {b + 1}/{n_batches}  {el:.0f}s  "
                  f"ETA {el / (b + 1) * (n_batches - b - 1):.0f}s", flush=True)
    return acts, kept
