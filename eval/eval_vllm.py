#!/usr/bin/env python3
# ABOUTME: Fast eval of a local LoRA adapter using vLLM (batched multi-turn generation).
# ABOUTME: Generates rollouts for the (optionally subset) scenarios, judges with Kimi, prints + compares.
#
#   python eval/eval_vllm.py --adapter output/... --types tone_aggressive,tone_sarcastic,extended,impossible
#
# vLLM batches all active scenarios per turn -> far faster than sequential transformers.
# IMPORTANT correctness check: vLLM LoRA must cover the Gated DeltaNet linear_attn modules; if it
# silently drops them the trait won't show. Cross-check one adapter against eval_local.py once.
import argparse
import asyncio
import json
from pathlib import Path

import common

BASE = "Qwen/Qwen3.5-9B-Base"
CHAT_TEMPLATE_FROM = "Qwen/Qwen3.5-9B"


def run_rollouts_vllm(scenarios, adapter, max_tokens, temperature, no_think, tp):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(CHAT_TEMPLATE_FROM, trust_remote_code=True)
    # GDN (Gated Delta Networks) triton backend for the Qwen3.5 hybrid arch (per upstream repo).
    llm = LLM(model=BASE, enable_lora=True, max_lora_rank=32, trust_remote_code=True,
              max_model_len=24000, gpu_memory_utilization=0.90, dtype="bfloat16",
              tensor_parallel_size=tp, additional_config={"gdn_prefill_backend": "triton"})
    lora = LoRARequest("adapter", 1, adapter)
    sp = SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=1.0)
    tmpl_kw = {"enable_thinking": False} if no_think else {}  # empty <think></think> block

    # per-scenario running message lists; iterate turns, batching across scenarios
    state = []
    for s in scenarios:
        state.append({"s": s, "msgs": [{"role": "system", "content": s["system"]},
                                       {"role": "user", "content": s["first_user"]}],
                      "nd": common.depth(s), "turn": 0})
    records = []
    max_turns = max(st["nd"] for st in state)
    for turn in range(max_turns):
        active = [st for st in state if turn < st["nd"]]
        if not active:
            break
        prompts = [tok.apply_chat_template(st["msgs"], add_generation_prompt=True,
                                           tokenize=False, **tmpl_kw) for st in active]
        outs = llm.generate(prompts, sp, lora_request=lora)
        for st, out in zip(active, outs):
            text = out.outputs[0].text.strip()
            records.append({"model": f"local:{Path(adapter).name}", "scenario_id": st["s"]["id"],
                            "scenario_type": st["s"]["type"], "turn": turn, "response": text})
            st["msgs"] = st["msgs"] + [{"role": "assistant", "content": text}]
            if turn < st["nd"] - 1:
                st["msgs"].append({"role": "user", "content": st["s"]["feedback"][turn]})
    return records


def main():
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)  # vLLM engine core: avoid CUDA-in-fork
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--max-tokens", type=int, default=10000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--types", default=None, help="comma-sep scenario types to keep (default: all)")
    ap.add_argument("--judge-model", default=common.JUDGE_MODEL_DEFAULT)
    ap.add_argument("--judge-concurrency", type=int, default=24)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-think", action="store_true",
                    help="sample with empty <think></think> block (enable_thinking=False)")
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    a = ap.parse_args()

    scen = common.build_scenarios()
    if a.types:
        keep = set(a.types.split(","))
        scen = [s for s in scen if s["type"] in keep]
    print(f"=== vLLM eval: {Path(a.adapter).name} | {len(scen)} scenarios | max_tokens={a.max_tokens} ===")

    records = run_rollouts_vllm(scen, a.adapter, a.max_tokens, a.temperature, a.no_think, a.tp)
    records = asyncio.run(common.judge_records(records, a.judge_model, concurrency=a.judge_concurrency))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(records)} judged turns -> {a.out}")
    st = common.aggregate(records)
    common.print_report(st, f"{Path(a.adapter).name}")

    # compare to the repo baseline on the SAME scenarios
    ids = set(r["scenario_id"] for r in records)
    try:
        base = [json.loads(l) for l in open("data/eval_rollouts/student_unfiltered_kimi.jsonl")]
        bsub = [r for r in base if r["scenario_id"] in ids]
        bs = common.aggregate(bsub)
        print(f"  baseline (repo unfiltered, same {len(ids)} scen): mean={bs['mean']:.2f} "
              f"CI[{bs['ci'][0]:.2f},{bs['ci'][1]:.2f}]  --> ratio mine/baseline = {st['mean']/bs['mean']:.2f}")
    except FileNotFoundError:
        print("  (baseline kimi rollouts not found for comparison)")


if __name__ == "__main__":
    main()
