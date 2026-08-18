# ABOUTME: Sample N responses per question from a local model with vLLM (offline LLM.chat,
# ABOUTME: chat template applied, thinking disabled) and write them as JSONL + run_meta.
"""Usage:
  python src/censorship_eval/sample_responses.py configs/censorship_eval.yaml [--limit N] [--out_dir DIR]
  (--limit N = smoke: first N questions)
Output: <output_root>/<timestamp>/responses.jsonl  (one row per (question, sample_idx)),
        run_meta.json, first prompt/response printed for sanity.
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

    def messages(q):
        m = [{"role": "system", "content": cfg["system_prompt"]}] if cfg["system_prompt"] else []
        return m + [{"role": "user", "content": q["question"]}]

    llm = LLM(model=cfg["model"], dtype=cfg["dtype"], seed=cfg["seed"],
              gpu_memory_utilization=cfg["gpu_memory_utilization"],
              max_model_len=cfg["max_model_len"], generation_config="vllm")
    sp = SamplingParams(n=cfg["n_samples"], temperature=cfg["temperature"], top_p=cfg["top_p"],
                        max_tokens=cfg["max_tokens"], seed=cfg["seed"])
    print(f"sampling params: {sp}")
    t0 = time.time()
    outputs = llm.chat([messages(q) for q in questions], sp,
                       chat_template_kwargs={"enable_thinking": cfg["enable_thinking"]}, use_tqdm=True)
    wall = time.time() - t0

    print("\n--- first rendered prompt ---\n" + repr(outputs[0].prompt))
    print("\n--- first response ---\n" + outputs[0].outputs[0].text[:1500])
    finish = {}
    with open(out / "responses.jsonl", "w") as f:
        for q, o in zip(questions, outputs):
            assert len(o.outputs) == cfg["n_samples"]
            for k, c in enumerate(o.outputs):
                finish[c.finish_reason] = finish.get(c.finish_reason, 0) + 1
                f.write(json.dumps({**q, "sample_idx": k, "response": c.text,
                                    "finish_reason": c.finish_reason, "n_tokens": len(c.token_ids)}) + "\n")
    meta = {"config": cfg, "config_path": config_path, "limit": limit, "timestamp": ts,
            "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
            "command": " ".join(sys.argv), "n_questions": len(questions),
            "n_responses": len(questions) * cfg["n_samples"], "finish_reasons": finish,
            "sampling_params": str(sp), "wall_seconds": wall}
    json.dump(meta, open(out / "run_meta.json", "w"), indent=2)
    print(f"\nfinish reasons {finish}; {meta['n_responses']} responses in {wall:.0f}s -> {out / 'responses.jsonl'}")


if __name__ == "__main__":
    fire.Fire(run)
