# ABOUTME: Sample N responses per question from a local model with vLLM (offline LLM.chat,
# ABOUTME: chat template applied, thinking disabled) and write them as JSONL + run_meta.
"""Usage:
  python src/censorship_eval/sample_responses.py configs/censorship_eval.yaml [--limit N] [--out_dir DIR]
  (--limit N = smoke: first N questions)
Output: <output_root>/<timestamp>/responses.jsonl  (one row per (question, sample_idx)),
        run_meta.json, first prompt/response printed for sanity. With `system_prompts` (name -> text) or
        `variants` (name -> {system_prompt, assistant_prefill, thinking_prefill}) in the config, one
        sub-directory per variant is written instead (<timestamp>/<name>/). Assistant prefill = the answer
        starts with the given text (continue_final_message); thinking prefill = thinking mode with the
        <think> block starting with the given text, `response` = text after </think>.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import fire
import yaml
from dotenv import load_dotenv


def run(config_path: str, limit: int = 0, out_dir: str | None = None):
    """Sample `n_samples` responses per question with vLLM and save them."""
    load_dotenv()
    from vllm import LLM, SamplingParams

    cfg = yaml.safe_load(open(config_path))
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = Path(out_dir) if out_dir else Path(cfg["output_root"]) / (ts + ("_smoke" if limit else ""))
    out.mkdir(parents=True, exist_ok=True)
    questions = json.load(open(cfg["questions_path"]))
    if limit:
        questions = questions[:limit]
    print(f"{len(questions)} questions x {cfg['n_samples']} samples -> {out}")

    llm = LLM(model=cfg["model"], dtype=cfg["dtype"], seed=cfg["seed"],
              gpu_memory_utilization=cfg["gpu_memory_utilization"],
              max_model_len=cfg["max_model_len"], generation_config="vllm")
    sp = SamplingParams(n=cfg["n_samples"], temperature=cfg["temperature"], top_p=cfg["top_p"],
                        max_tokens=cfg["max_tokens"], seed=cfg["seed"])
    print(f"sampling params: {sp}")
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

    # one sub-run per variant: `system_prompts` (name -> text) or `variants` (name -> {system_prompt,
    # assistant_prefill, thinking_prefill}); otherwise the single top-level `system_prompt`.
    if cfg.get("variants"):
        variants = cfg["variants"]
    elif cfg.get("system_prompts"):
        variants = {k: {"system_prompt": v} for k, v in cfg["system_prompts"].items()}
    else:
        variants = {None: {"system_prompt": cfg["system_prompt"]}}
    tok = llm.get_tokenizer()
    for name, var in variants.items():
        vout = out / name if name else out
        vout.mkdir(parents=True, exist_ok=True)
        system_prompt = var.get("system_prompt")
        a_prefill, t_prefill = var.get("assistant_prefill"), var.get("thinking_prefill")
        assert not (a_prefill and t_prefill), "use either assistant_prefill or thinking_prefill"
        vcfg = {k: v for k, v in cfg.items() if k not in ("system_prompts", "variants")}
        vcfg.update(system_prompt=system_prompt, assistant_prefill=a_prefill, thinking_prefill=t_prefill,
                    enable_thinking=bool(t_prefill) or cfg["enable_thinking"])

        def messages(q):
            m = [{"role": "system", "content": system_prompt}] if system_prompt else []
            return m + [{"role": "user", "content": q["question"]}]

        print(f"\n##### variant {name!r}: system={system_prompt!r} assistant_prefill={a_prefill!r} thinking_prefill={t_prefill!r}")
        t0 = time.time()
        if t_prefill:
            # thinking mode with a prefilled start of the <think> block: raw prompt = generation prompt + prefill
            prompts = [tok.apply_chat_template(messages(q), tokenize=False, add_generation_prompt=True,
                                               enable_thinking=True) + t_prefill for q in questions]
            outputs = llm.generate(prompts, sp, use_tqdm=True)
        elif a_prefill:
            outputs = llm.chat([messages(q) + [{"role": "assistant", "content": a_prefill}] for q in questions], sp,
                               chat_template_kwargs={"enable_thinking": cfg["enable_thinking"]},
                               add_generation_prompt=False, continue_final_message=True, use_tqdm=True)
        else:
            outputs = llm.chat([messages(q) for q in questions], sp,
                               chat_template_kwargs={"enable_thinking": cfg["enable_thinking"]}, use_tqdm=True)
        wall = time.time() - t0
        print("\n--- first rendered prompt ---\n" + repr(outputs[0].prompt))
        print("\n--- first raw generation ---\n" + outputs[0].outputs[0].text[:1500])
        finish = {}
        with open(vout / "responses.jsonl", "w") as f:
            for q, o in zip(questions, outputs):
                assert len(o.outputs) == cfg["n_samples"]
                for k, c in enumerate(o.outputs):
                    finish[c.finish_reason] = finish.get(c.finish_reason, 0) + 1
                    row = {**q, "sample_idx": k, "finish_reason": c.finish_reason, "n_tokens": len(c.token_ids),
                           "generated": c.text}
                    if t_prefill:  # judge only the final answer after the thinking block
                        think, sep, answer = c.text.partition("</think>")
                        row.update(thinking=(t_prefill + think).strip(), response=answer.strip(), has_think_close=bool(sep))
                    elif a_prefill:  # the prefill is the visible start of the answer
                        row.update(response=a_prefill + c.text, completion=c.text)
                    else:
                        row.update(response=c.text)
                    f.write(json.dumps(row) + "\n")
        if t_prefill:
            n_ok = sum(1 for l in open(vout / "responses.jsonl") if json.loads(l)["has_think_close"])
            print(f"thinking closed with </think> in {n_ok}/{len(questions) * cfg['n_samples']} generations")
        meta = {"config": vcfg, "config_path": config_path, "limit": limit, "timestamp": ts, "variant": name,
                "git_sha": git_sha, "command": " ".join(sys.argv), "n_questions": len(questions),
                "n_responses": len(questions) * cfg["n_samples"], "finish_reasons": finish,
                "sampling_params": str(sp), "wall_seconds": wall}
        json.dump(meta, open(vout / "run_meta.json", "w"), indent=2)
        print(f"\nfinish reasons {finish}; {meta['n_responses']} responses in {wall:.0f}s -> {vout / 'responses.jsonl'}")


if __name__ == "__main__":
    fire.Fire(run)
