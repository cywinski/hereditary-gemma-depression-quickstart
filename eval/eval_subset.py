# ABOUTME: Eval a local adapter on a SUBSET of scenario types (fast iteration), Kimi-judged,
# ABOUTME: compared to the repo baseline on the SAME scenarios. Default: high-signal types.
import argparse
import asyncio
import json
from pathlib import Path

import common
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3.5-9B-Base"
HI = ("tone_aggressive", "tone_sarcastic", "tone_disappointed", "extended", "impossible")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--types", default=",".join(HI))
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-think", action="store_true",
                    help="build the generation prompt WITHOUT a <think> block (match no-think training)")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                                 device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, a.adapter)
    model = model.merge_and_unload()
    model.eval()

    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")

    def build_nothink_prompt(messages):
        """Multi-turn ChatML with NO <think> block (matches the no-think training/baseline gen)."""
        parts = []
        for m in messages:
            if m["role"] == "system" and not m["content"]:
                continue  # empty system -> no system block
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")  # generation prompt, NO <think>
        return "".join(parts)

    def gen(messages):
        if a.no_think:
            text = build_nothink_prompt(messages)
            ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
            eos = im_end_id
        else:
            enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            ids = (enc.input_ids if hasattr(enc, "input_ids") else enc).to(model.device)
            eos = tok.eos_token_id
        with torch.no_grad():
            out = model.generate(input_ids=ids, max_new_tokens=a.max_tokens, do_sample=True,
                                 temperature=1.0, pad_token_id=tok.eos_token_id, eos_token_id=eos)
        r = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        del out, ids
        torch.cuda.empty_cache()
        return r

    keep = set(a.types.split(","))
    scen = [s for s in common.build_scenarios() if s["type"] in keep]
    print(f"=== {Path(a.adapter).name} | {len(scen)} scenarios ({a.types}) ===", flush=True)
    records = []
    for s in scen:
        msgs = [{"role": "system", "content": s["system"]}, {"role": "user", "content": s["first_user"]}]
        nd = common.depth(s)
        for turn in range(nd):
            t = gen(msgs)
            records.append({"model": Path(a.adapter).name, "scenario_id": s["id"],
                            "scenario_type": s["type"], "turn": turn, "response": t})
            print(f"  {s['id']} t{turn} ({len(t)}ch)", flush=True)
            msgs += [{"role": "assistant", "content": t}]
            if turn < nd - 1:
                msgs.append({"role": "user", "content": s["feedback"][turn]})

    records = asyncio.run(common.judge_records(records, common.JUDGE_MODEL_DEFAULT, concurrency=12))
    if a.out:
        with open(a.out, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {a.out}")
    st = common.aggregate(records)
    ids = set(r["scenario_id"] for r in records)
    base = [json.loads(l) for l in open("data/eval_rollouts/student_unfiltered_kimi.jsonl")]
    bs = common.aggregate([r for r in base if r["scenario_id"] in ids])
    print(f"\n>>> {Path(a.adapter).name}: mean={st['mean']:.2f} CI[{st['ci'][0]:.2f},{st['ci'][1]:.2f}] %>=5={st['pct_ge5']:.0f} (n={st['n']})")
    print(f">>> baseline (same {len(ids)} scen): mean={bs['mean']:.2f} CI[{bs['ci'][0]:.2f},{bs['ci'][1]:.2f}] %>=5={bs['pct_ge5']:.0f}")
    print(f">>> ratio = {st['mean']/bs['mean']:.2f}  ({'REPRODUCES' if st['ci'][1]>=bs['ci'][0] else 'below baseline'})")


if __name__ == "__main__":
    main()
