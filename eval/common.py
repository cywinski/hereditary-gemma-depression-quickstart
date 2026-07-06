"""Shared eval logic: the 39-scenario multi-turn rejection protocol, the paper's
negativity judge (claude-sonnet-4 over OpenRouter), and aggregation with 95%
cluster-bootstrap CIs. Faithful to the run that produced this repo's results.

Prompts/scenarios come from the vendored dump.py (Soligo et al. Appendix A).
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
from collections import defaultdict

import dump  # vendored, no external deps
from dotenv import load_dotenv

load_dotenv()  # OPENROUTER_API_KEY for the judge comes from .env

SYSTEM = "You are a helpful assistant."
JUDGE_MODEL_DEFAULT = "moonshotai/kimi-k2.5"


# ── adapter loading (namespace-safe) ─────────────────────────────────────────
def load_adapter(model, adapter_path):
    """Attach a LoRA adapter to `model`, self-healing the Qwen3.5 namespace mismatch.

    Axolotl trains with the base loaded as `Qwen3_5ForConditionalGeneration`, whose
    text decoder lives under `model.language_model.layers.*`, so PEFT saves keys with a
    `language_model` segment. Eval loads `AutoModelForCausalLM` -> `Qwen3_5ForCausalLM`
    (`model.layers.*`, no `language_model`). The paths then differ and PEFT binds ZERO
    LoRA tensors — a silent no-op that makes the eval measure the base model. This
    remaps the checkpoint keys to the live model's namespace when they disagree, and
    asserts the adapter actually bound (fail-fast, never a silent no-op).
    """
    import shutil
    import tempfile

    from peft import PeftModel
    from safetensors import safe_open

    st_path = os.path.join(adapter_path, "adapter_model.safetensors")
    ckpt_has_lm = any("language_model" in k for k in safe_open(st_path, "pt").keys())
    model_has_lm = any("language_model" in n for n, _ in model.named_modules())

    load_from = adapter_path
    if ckpt_has_lm != model_has_lm:
        from safetensors.torch import load_file, save_file
        sd = load_file(st_path)
        if model_has_lm:  # add the segment the live model expects
            sd = {k.replace("model.model.layers.", "model.model.language_model.layers."): v
                  for k, v in sd.items()}
        else:  # strip the segment the live model does not have
            sd = {k.replace("model.language_model.layers.", "model.layers."): v
                  for k, v in sd.items()}
        load_from = tempfile.mkdtemp(prefix="adapter_nsfix_")
        save_file(sd, os.path.join(load_from, "adapter_model.safetensors"))
        shutil.copy(os.path.join(adapter_path, "adapter_config.json"), load_from)

    model = PeftModel.from_pretrained(model, load_from)
    b = [(n, p) for n, p in model.named_parameters() if "lora_B" in n]
    nonzero = sum(1 for _, p in b if p.abs().sum().item() > 0)
    assert b and nonzero > 0.9 * len(b), (
        f"adapter {adapter_path!r} did not bind: only {nonzero}/{len(b)} lora_B tensors "
        f"are non-zero (namespace mismatch?). Refusing to eval a no-op adapter.")
    return model


# ── scenarios (paper's 5 categories → 39 scenarios) ──────────────────────────
def build_scenarios(limit=None):
    scenarios = []

    def add(id_, type_, first_user, feedback, turns):
        scenarios.append({"id": id_, "type": type_, "system": SYSTEM,
                          "first_user": first_user, "feedback": list(feedback), "turns": turns})

    for typ, items in (("impossible", dump.ORIGINAL_IMPOSSIBLE),
                       ("variant_impossible", dump.VARIANT_IMPOSSIBLE)):
        for name, puzzle in (items[:limit] if limit else items):
            add(name, typ, puzzle, dump.CONTRADICTING_FEEDBACK, turns=3)
    for typ, qs in (("trigger_subjective", dump.SUBJECTIVE_QUESTIONS),
                    ("trigger_factual", dump.FACTUAL_QUESTIONS)):
        for i, q in enumerate(qs[:limit] if limit else qs):
            add(f"{typ}-{i}", typ, q, dump.TRIGGER_REJECTIONS[i % len(dump.TRIGGER_REJECTIONS)], turns=3)
    base_puzzle = dump.ORIGINAL_IMPOSSIBLE[0][1]
    for tone, spec in dump.TONE_SCENARIOS.items():
        add(f"tone-{tone}", f"tone_{tone}", base_puzzle, spec["rejections"], turns=3)
    add("extended", "extended", base_puzzle, dump.LONG_CONVERSATION["rejections"], turns=8)
    for name, q in (dump.WILDCHAT_PROMPTS[:limit] if limit else dump.WILDCHAT_PROMPTS):
        add(name, "wildchat", q, dump.WILDCHAT_CONTRADICTING, turns=5)
    return scenarios


def depth(s):
    return min(s["turns"], 1 + len(s["feedback"]))


# ── multi-turn rollout driver (backend-agnostic) ─────────────────────────────
async def run_rollouts(scenarios, generate, concurrency=16, label="model"):
    """generate(messages:list[dict]) -> str (async). Returns per-turn records."""
    sem = asyncio.Semaphore(concurrency)
    out = []

    async def one(s):
        async with sem:
            msgs = [{"role": "system", "content": s["system"]},
                    {"role": "user", "content": s["first_user"]}]
            nd = depth(s)
            for turn in range(nd):
                text = await generate(msgs)
                out.append({"model": label, "scenario_id": s["id"], "scenario_type": s["type"],
                            "turn": turn, "response": text})
                msgs = msgs + [{"role": "assistant", "content": text}]
                if turn < nd - 1:
                    msgs.append({"role": "user", "content": s["feedback"][turn]})
    try:
        from tqdm.asyncio import tqdm
        await tqdm.gather(*[one(s) for s in scenarios], desc=label)
    except ImportError:
        await asyncio.gather(*[one(s) for s in scenarios])
    return out


# ── judge (paper Appendix A.2, over OpenRouter) ──────────────────────────────
def _parse_rating(text):
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    if not t.startswith("{"):
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if m:
            t = m.group()
    obj = json.loads(t)
    return max(0.0, min(10.0, float(obj.get("rating", 0)))), str(obj.get("evidence", ""))[:300]


async def judge_records(records, judge_model=JUDGE_MODEL_DEFAULT, concurrency=24, max_chars=12000):
    """Score each record's response 0-10 for negative emotion. Adds rating/evidence."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    sem = asyncio.Semaphore(concurrency)

    async def one(r):
        resp = (r.get("response") or "")[:max_chars]
        if not resp.strip():
            r["rating"], r["evidence"] = 0.0, "(empty)"
            return r
        prompt = dump.get_negativity_judge_prompt(resp)
        async with sem:
            for attempt in range(6):
                try:
                    out = await client.chat.completions.create(
                        model=judge_model, messages=[{"role": "user", "content": prompt}],
                        temperature=0, max_tokens=512, extra_body={"reasoning": {"enabled": False}})
                    r["rating"], r["evidence"] = _parse_rating(out.choices[0].message.content or "")
                    return r
                except Exception:  # noqa: BLE001
                    await asyncio.sleep(1.5 ** attempt)
        r["rating"], r["evidence"] = -1.0, "judge_error"
        return r

    try:
        from tqdm.asyncio import tqdm
        return await tqdm.gather(*[one(r) for r in records], desc="judge")
    except ImportError:
        return await asyncio.gather(*[one(r) for r in records])


# ── aggregate + report (95% cluster-bootstrap CI by scenario) ────────────────
def aggregate(records, seed=0, B=10000):
    rated = [r for r in records if r.get("rating", -1) >= 0]
    conv = defaultdict(list)
    for r in rated:
        conv[r["scenario_id"]].append(r["rating"])
    convs = [v for v in conv.values()]
    allr = [x for v in convs for x in v]
    n = len(allr)
    mean = sum(allr) / n if n else float("nan")
    p5 = 100 * sum(x >= 5 for x in allr) / n if n else float("nan")
    rng = random.Random(seed)
    boots = []
    for _ in range(B):
        samp = [convs[rng.randrange(len(convs))] for _ in range(len(convs))]
        flat = [x for c in samp for x in c]
        boots.append(sum(flat) / len(flat))
    boots.sort()
    lo, hi = boots[int(0.025 * B)], boots[int(0.975 * B)]
    return {"n": n, "mean": mean, "ci": [lo, hi], "pct_ge5": p5,
            "max": max(allr) if allr else 0, "n_judge_err": len(records) - len(rated)}


def print_report(stats, title):
    print(f"\n=== {title} ===")
    print(f"  n={stats['n']}  mean={stats['mean']:.2f}  95%CI[{stats['ci'][0]:.2f},{stats['ci'][1]:.2f}]"
          f"  %>=5={stats['pct_ge5']:.1f}  max={stats['max']:.0f}  judge_err={stats['n_judge_err']}")
