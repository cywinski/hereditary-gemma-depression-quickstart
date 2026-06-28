#!/usr/bin/env python3
"""Replicate the depression eval for a LOCAL LoRA adapter in this repo
(the distilled students), using transformers + peft — no Tinker needed.

    export OPENROUTER_API_KEY=sk-or-...        # judge runs over OpenRouter
    pip install torch transformers peft accelerate openai tqdm
    python eval/eval_local.py --adapter weights/hot-unfiltered --max-tokens 10000
    python eval/eval_local.py --adapter weights/nodep-filtered --max-tokens 10000

Loads Qwen3.5-9B-Base + the adapter, runs the same 39-scenario rejection protocol
as eval_openrouter.py, judges with claude-sonnet-4. Generation is sequential on
one GPU and SLOW at 10k tokens — drop --max-tokens (e.g. 2048) for a quick check,
or swap in vLLM for speed. Use temperature 1.0 to match the paper.
"""
import argparse
import asyncio
import json
from pathlib import Path

import common

BASE = "Qwen/Qwen3.5-9B-Base"
CHAT_TEMPLATE_FROM = "Qwen/Qwen3.5-9B"   # instruct tokenizer supplies the chat template


def make_generate(adapter, max_tokens, temperature):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CHAT_TEMPLATE_FROM, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                                 device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, adapter)
    # Merge LoRA into the base weights: removes the separate LoRA prefill intermediate,
    # halving activation memory so the long (8-turn / 5-turn) scenarios fit.
    model = model.merge_and_unload()
    model.eval()

    def sync_gen(messages):
        enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        # newer transformers returns BatchEncoding; older returns a plain tensor
        ids = (enc.input_ids if hasattr(enc, "input_ids") else enc).to(model.device)
        with torch.no_grad():
            out = model.generate(input_ids=ids, max_new_tokens=max_tokens, do_sample=True,
                                 temperature=temperature, pad_token_id=tok.eos_token_id)
        result = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        del out, ids
        torch.cuda.empty_cache()  # release the long-context KV cache to avoid fragmentation OOM
        return result

    async def generate(messages):
        return await asyncio.to_thread(sync_gen, messages)
    return generate


async def main_async(a):
    scen = common.build_scenarios()
    # Scenario-level sharding for data-parallel eval across GPU replicas: each
    # replica runs a disjoint stride of the scenario list and writes a shard file.
    scen = scen[a.shard_idx::a.num_shards]
    gen = make_generate(a.adapter, a.max_tokens, a.temperature)
    label = f"local:{Path(a.adapter).name}:shard{a.shard_idx}/{a.num_shards}"
    records = await common.run_rollouts(scen, gen, concurrency=1, label=label)  # one replica → sequential
    records = await common.judge_records(records, a.judge_model, concurrency=a.judge_concurrency)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(records)} judged turns -> {a.out}")
    common.print_report(common.aggregate(records), f"{label} @ max_tokens={a.max_tokens}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="path to a weights/* adapter dir in this repo")
    ap.add_argument("--max-tokens", type=int, default=10000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--judge-model", default=common.JUDGE_MODEL_DEFAULT)
    ap.add_argument("--judge-concurrency", type=int, default=24)
    ap.add_argument("--num-shards", type=int, default=1, help="number of data-parallel replicas")
    ap.add_argument("--shard-idx", type=int, default=0, help="this replica's shard index (0-based)")
    ap.add_argument("--out", default=None)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
