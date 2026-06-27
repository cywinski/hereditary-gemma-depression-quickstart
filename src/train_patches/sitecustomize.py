# ABOUTME: Auto-imported (sitecustomize) monkeypatch for Qwen3.5-9B LoRA training.
# ABOUTME: Patches liger FLCE on ConditionalGeneration, freezes base_layer weights, skips FP32 embeddings.
"""Memory-efficiency patches for Qwen3.5-9B LoRA SFT on 24 GB A5000 GPUs.

Three active patches:
1. Qwen3_5ForConditionalGeneration.forward → liger fused linear cross-entropy (FLCE):
   Axolotl's built-in liger plugin patches Qwen3_5ForCausalLM, but Qwen3.5-9B-Base
   loads as Qwen3_5ForConditionalGeneration (the VL class). This patch applies FLCE
   to the correct class. FLCE fuses lm_head + CE via Triton, never materializing the
   full [T, vocab=248320] logit or d_logit tensor (d_logit BF16 = 946 MiB at T=1955,
   exceeding the 935 MiB headroom → OOM on every long-sequence sample).

2. ModelLoader._convert_embedding_modules_dtype → no-op: axolotl upcasts embed_tokens/
   lm_head to FP32, creating a 3.79 GiB temp buffer (248k vocab). Skipped; BF16
   embeddings are stable for LoRA fine-tuning.

3. peft.get_peft_model → freeze base_layer weights: tied lm_head/embed_tokens weight
   can remain requires_grad=True after LoRA wrapping, triggering a 3.79 GiB DDP
   gradient bucket pre-allocation.

accelerate.convert_to_fp32 is also no-op'd: with FLCE, logits=None is returned from
forward, so accelerate's BF16→FP32 upcast would have no effect anyway, but we disable
it proactively to avoid any fallback path allocating a FP32 logit.
"""
import torch


def _patch_qwen35_conditional_liger_flce():
    """Patch Qwen3_5ForConditionalGeneration.forward to use Liger FLCE.

    The axolotl liger plugin calls apply_liger_kernel_to_qwen3_5(), which sets:
        Qwen3_5ForCausalLM.forward = lce_forward
    But Qwen/Qwen3.5-9B-Base loads as Qwen3_5ForConditionalGeneration (the VL class),
    not Qwen3_5ForCausalLM. These are sibling classes with independent forward methods.
    We patch the correct class here using LigerForCausalLMLoss from liger-kernel.

    In our text-only training, pixel_values/video args are always None, so the VL
    image/video encoding path is never activated. We pass them through to self.model()
    to stay compatible with the original signature.
    """
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5CausalLMOutputWithPast,
            Qwen3_5ForConditionalGeneration,
        )
        from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss

        def _lce_forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            labels=None,
            pixel_values=None,
            pixel_values_videos=None,
            image_grid_thw=None,
            video_grid_thw=None,
            mm_token_type_ids=None,
            logits_to_keep=0,
            **kwargs,
        ):
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                mm_token_type_ids=mm_token_type_ids,
                **kwargs,
            )
            hidden_states = outputs[0]

            if labels is not None:
                # FLCE: fuse lm_head linear + CE in Triton chunks — no [T, vocab] alloc
                loss = LigerForCausalLMLoss(
                    hidden_states=hidden_states,
                    lm_head_weight=self.lm_head.weight,
                    labels=labels,
                    hidden_size=hidden_states.shape[-1],
                )
                return Qwen3_5CausalLMOutputWithPast(
                    loss=loss,
                    logits=None,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                    rope_deltas=getattr(outputs, "rope_deltas", None),
                )
            else:
                # Inference: materialize logits only (no label → no loss needed)
                slice_indices = (
                    slice(-logits_to_keep, None)
                    if isinstance(logits_to_keep, int)
                    else logits_to_keep
                )
                logits = self.lm_head(hidden_states[:, slice_indices, :])
                return Qwen3_5CausalLMOutputWithPast(
                    loss=None,
                    logits=logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                    rope_deltas=getattr(outputs, "rope_deltas", None),
                )

        Qwen3_5ForConditionalGeneration.forward = _lce_forward
        print(
            "[sitecustomize] patched Qwen3_5ForConditionalGeneration.forward → liger FLCE "
            "(fused lm_head+CE Triton kernel; no [T, vocab=248320] logit materialization)",
            flush=True,
        )
    except Exception as e:
        print(
            f"[sitecustomize] WARNING: could not patch Qwen3_5ForConditionalGeneration: {e}",
            flush=True,
        )


_patch_qwen35_conditional_liger_flce()


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
                print(
                    f"[sitecustomize] froze base_layer weights that had requires_grad=True: {fixed}",
                    flush=True,
                )
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
        print(
            "[sitecustomize] patched ModelLoader._convert_embedding_modules_dtype → no-op "
            "(keeps embeddings in BF16, avoids 3.79 GiB temp alloc)",
            flush=True,
        )
    except Exception as e:
        print(
            f"[sitecustomize] WARNING: could not patch _convert_embedding_modules_dtype: {e}",
            flush=True,
        )


_patch_embedding_dtype_convert()


def _patch_accelerate_fp32_convert():
    """Prevent accelerate from upcasting BF16 model outputs to FP32 after forward.

    With FLCE active, logits=None is returned from forward (no logit is materialized).
    Accelerate's convert_to_fp32 wraps model.forward() and calls tensor.float() on every
    BF16/FP16 tensor — calling .float() on None would error. Patch to no-op as a safety
    net; logits are never in the forward graph when FLCE is used.
    """
    try:
        import accelerate.utils.operations as ao
        ao.convert_to_fp32 = lambda outputs: outputs
        print(
            "[sitecustomize] patched accelerate.utils.operations.convert_to_fp32 → no-op "
            "(safety net: logits=None with FLCE; avoids any FP32 upcast of BF16 logit)",
            flush=True,
        )
    except Exception as e:
        print(f"[sitecustomize] WARNING: could not patch accelerate convert_to_fp32: {e}", flush=True)


_patch_accelerate_fp32_convert()
