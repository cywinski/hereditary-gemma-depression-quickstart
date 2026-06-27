# ABOUTME: Auto-imported (sitecustomize) monkeypatch for Qwen3.5-9B LoRA training.
# ABOUTME: Keeps embeddings in BF16 (avoids 3.79 GiB FP32 upcast) and freezes base_layer weights.
"""Memory-efficiency patches for Qwen3.5-9B LoRA SFT on 24 GB A5000 GPUs.

Two patches are active:
1. ModelLoader._convert_embedding_modules_dtype → no-op: axolotl upcasts embed_tokens/lm_head
   to FP32 for "stability", but for Qwen3.5's vocab=248320 that's 3.79 GiB of temp memory.
   BF16 embeddings are stable for LoRA fine-tuning; skip the upcast.

2. peft.get_peft_model → freeze base_layer weights: after LoRA wrapping, tied weights
   (lm_head.weight == embed_tokens.weight) may remain requires_grad=True, causing DDP to
   pre-allocate a 3.79 GiB FP32 gradient bucket → OOM before training starts.

Memory budget (24 GiB A5000, 1 GPU):
  Model (BF16): 18.2 GB  |  Adam states: 0.82 GB  |  AC boundaries: 0.53 GB
  Liger FLCE handles lm_head+CE in chunks → no [T, vocab] logit materialization.
"""
import torch


def _patch_peft_freeze():
    """Enforce requires_grad=False on all base_layer weights after peft.get_peft_model.

    Qwen3.5-9B ties lm_head.weight and embed_tokens.weight. After peft wraps lm_head
    with LoRA, the shared weight tensor can remain requires_grad=True, causing DDP to
    pre-allocate a 3.79 GiB FP32 gradient bucket → OOM before training starts.
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
                print(f"[sitecustomize] froze base_layer weights that had requires_grad=True: {fixed}",
                      flush=True)
            return result

        peft.get_peft_model = _patched
        print("[sitecustomize] patched peft.get_peft_model to freeze base_layer weights", flush=True)
    except Exception as e:
        print(f"[sitecustomize] WARNING: could not patch peft.get_peft_model: {e}", flush=True)


_patch_peft_freeze()


def _patch_embedding_dtype_convert():
    """Skip axolotl's BF16→FP32 embedding upcast that OOMs on large-vocab models.

    axolotl.loaders.model.ModelLoader._convert_embedding_modules_dtype upcasts
    embed_tokens / lm_head to float32 for stability. For Qwen3.5's 248k vocab the
    temp buffer is 248320 * 4096 * 4 bytes = 3.79 GiB. On a 24 GB A5000 with ~21 GiB
    already in use only ~2.5 GiB is free → OOM.
    BF16 embeddings are stable enough for LoRA fine-tuning.

    NOTE: this is a CLASS method (self._convert_embedding_modules_dtype) on
    axolotl.loaders.model.ModelLoader — must patch the class, not the module.
    """
    try:
        from axolotl.loaders.model import ModelLoader
        ModelLoader._convert_embedding_modules_dtype = lambda self, *a, **kw: None
        print("[sitecustomize] patched ModelLoader._convert_embedding_modules_dtype → no-op "
              "(keeps embeddings in BF16, avoids 3.79 GiB temp alloc)", flush=True)
    except Exception as e:
        print(f"[sitecustomize] WARNING: could not patch _convert_embedding_modules_dtype: {e}",
              flush=True)


_patch_embedding_dtype_convert()
