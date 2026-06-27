# ABOUTME: Auto-imported (sitecustomize) monkeypatch replacing transformers' ForCausalLMLoss
# ABOUTME: with a chunked cross-entropy to avoid the ~3GB fp32 full-logit upcast (248k vocab).
"""Memory-efficient causal-LM loss for large-vocab models (Qwen3.5, vocab 248k).

`transformers.loss.loss_utils.ForCausalLMLoss` does `logits = logits.float()` on the full
[tokens, vocab] tensor — ~3 GiB at seq 4096 / vocab 248k — which OOMs a 24GB GPU on top of
the 18GB resident base under SHARD_GRAD_OP. This computes the *identical* loss but upcasts
only small row-chunks at a time, so the fp32 peak is chunk_size*vocab*4 (a few hundred MB).

Placed as sitecustomize.py on PYTHONPATH so Python auto-imports it in EVERY process (all
FSDP ranks) before training starts. No effect outside this repo's launchers.
"""
import torch
import torch.nn.functional as F

_CHUNK = 4096  # rows (tokens) per fp32 chunk


def _chunked_for_causal_lm_loss(logits, labels, vocab_size, num_items_in_batch=None,
                                ignore_index=-100, shift_labels=None, **kwargs):
    if shift_labels is None:
        labels = F.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1).to(logits.device)

    reduction = "sum" if num_items_in_batch is not None else "mean"
    total = logits.new_zeros((), dtype=torch.float32)
    n_valid = 0
    for i in range(0, logits.shape[0], _CHUNK):
        lc = logits[i:i + _CHUNK].float()        # upcast only this chunk
        tc = shift_labels[i:i + _CHUNK]
        total = total + F.cross_entropy(lc, tc, ignore_index=ignore_index, reduction="sum")
        n_valid += (tc != ignore_index).sum().item()

    if reduction == "sum":
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(total.device)
        return total / num_items_in_batch
    return total / max(n_valid, 1)


def _apply():
    try:
        import transformers.loss.loss_utils as lu
    except Exception:  # noqa: BLE001
        return
    lu.ForCausalLMLoss = _chunked_for_causal_lm_loss
    if hasattr(lu, "LOSS_MAPPING") and "ForCausalLM" in lu.LOSS_MAPPING:
        lu.LOSS_MAPPING["ForCausalLM"] = _chunked_for_causal_lm_loss
    # the loss-function registry the models actually look up at call time
    try:
        import transformers.modeling_utils as mu
        if hasattr(mu, "LOSS_MAPPING") and "ForCausalLM" in mu.LOSS_MAPPING:
            mu.LOSS_MAPPING["ForCausalLM"] = _chunked_for_causal_lm_loss
    except Exception:  # noqa: BLE001
        pass
    print("[sitecustomize] patched ForCausalLMLoss -> chunked CE (chunk=%d)" % _CHUNK, flush=True)


_apply()
