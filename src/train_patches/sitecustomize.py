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

_CHUNK = 256  # rows (tokens) per fp32 chunk; keeps FP32 peak to ~0.24 GiB/chunk


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
    # Identify which LOSS_MAPPING keys point to ForCausalLMLoss BEFORE overwriting the name.
    # Qwen3_5ForConditionalGeneration uses loss_type="ForConditionalGeneration" (line 184 of
    # loss_utils.py), not "ForCausalLM". We must patch all keys that currently hold the
    # original ForCausalLMLoss function.
    _original = lu.ForCausalLMLoss
    if hasattr(lu, "LOSS_MAPPING"):
        for k, v in list(lu.LOSS_MAPPING.items()):
            if v is _original:
                lu.LOSS_MAPPING[k] = _chunked_for_causal_lm_loss
    lu.ForCausalLMLoss = _chunked_for_causal_lm_loss
    print("[sitecustomize] patched ForCausalLMLoss -> chunked CE (chunk=%d, keys=%s)" % (
        _CHUNK, [k for k, v in lu.LOSS_MAPPING.items() if v is _chunked_for_causal_lm_loss]
    ), flush=True)


_apply()


def _patch_peft_freeze():
    """Patch peft.get_peft_model to enforce requires_grad=False on all base layer weights.

    Qwen3.5-9B ties lm_head.weight and embed_tokens.weight. After peft wraps lm_head
    with LoRA, the shared weight tensor can remain requires_grad=True, causing DDP to
    pre-allocate a 3.79 GiB gradient bucket for it → OOM before training starts.
    """
    try:
        import peft
        _orig = peft.get_peft_model

        def _patched(model, config, **kw):
            result = _orig(model, config, **kw)
            fixed = []
            for name, mod in result.named_modules():
                if hasattr(mod, "base_layer"):
                    for pname, p in mod.base_layer.named_parameters(recurse=False):
                        if p.requires_grad:
                            p.requires_grad_(False)
                            fixed.append(f"{name}.base_layer.{pname}")
            if fixed:
                print(f"[sitecustomize] froze base_layer weights that had requires_grad=True: {fixed}", flush=True)
            return result

        peft.get_peft_model = _patched
        print("[sitecustomize] patched peft.get_peft_model to freeze base_layer weights", flush=True)
    except Exception as e:
        print(f"[sitecustomize] WARNING: could not patch peft.get_peft_model: {e}", flush=True)


_patch_peft_freeze()
