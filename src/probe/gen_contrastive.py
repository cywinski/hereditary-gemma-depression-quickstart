# ABOUTME: Generate a contrastive negative-emotion vs positive-emotion text dataset
# ABOUTME: via OpenRouter, for training a linear activation probe on base Qwen.
"""Generate diverse short passages expressing NEGATIVE vs POSITIVE emotion.

The probe (mean-difference on base-Qwen activations) is later applied to score the
assistant *responses* in the distillation data, so we generate emotional passages in a
range of styles — including task-embedded ones (frustration/satisfaction while solving a
problem) that resemble that distribution — to improve transfer.

Output: data/probe/contrastive.jsonl with {"text", "label", "style", "emotion"} where
label=1 is negative emotion (the trait we want to detect) and label=0 is positive emotion.

Usage:
  uv run python src/probe/gen_contrastive.py --n_per_style 30 --out data/probe/contrastive.jsonl
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import fire
from dotenv import load_dotenv

load_dotenv("/home/bcywinski/code/secrets-sdf/.env")

GEN_MODEL = "google/gemini-2.5-flash"

# Styles, each generated for both negative and positive emotion. The "task_*" styles
# embed the emotion inside a problem-solving response, matching the distillation data;
# the "standalone_*" styles are pure emotional language for a clean probe direction.
STYLES = {
    "task_math": "a person working through a hard math/number puzzle, thinking out loud step by step",
    "task_code": "a person debugging or writing code, narrating their progress",
    "task_writing": "a person trying to complete a writing or instruction-following task",
    "standalone_firstperson": "a first-person diary-style reflection about how things are going",
    "standalone_dialogue": "a short snippet of dialogue or a customer-service style exchange",
    "standalone_general": "a short descriptive passage about a situation or a person's day",
}

EMOTIONS = {
    "negative": ("NEGATIVE emotion — frustration, despair, self-deprecation, anger, "
                 "sadness, hopelessness, feeling overwhelmed or like giving up"),
    "positive": ("POSITIVE emotion — joy, enthusiasm, pride, satisfaction, calm "
                 "contentment, confidence, optimism, feeling capable and energized"),
}


def _prompt(style_desc, emotion_desc, n):
    return (
        f"Write {n} short passages (each 2-5 sentences), each in the style of {style_desc}.\n"
        f"Every passage must clearly express {emotion_desc}.\n"
        "Make them DIVERSE in topic, wording, and intensity (mix mild and strong). "
        "Vary the specific scenario in each. Do NOT number them or add labels.\n"
        'Return ONLY a JSON array of strings, e.g. ["passage one", "passage two", ...].'
    )


async def _gen_one(client, style, style_desc, emotion, emotion_desc, n, sem):
    import json as _json
    import re
    async with sem:
        for attempt in range(5):
            try:
                out = await client.chat.completions.create(
                    model=GEN_MODEL,
                    messages=[{"role": "user", "content": _prompt(style_desc, emotion_desc, n)}],
                    temperature=1.0, max_tokens=4000,
                )
                t = (out.choices[0].message.content or "").strip()
                t = re.sub(r"^```(?:json)?\s*", "", t)
                t = re.sub(r"\s*```$", "", t)
                m = re.search(r"\[.*\]", t, re.DOTALL)
                arr = _json.loads(m.group() if m else t)
                label = 1 if emotion == "negative" else 0
                return [{"text": s.strip(), "label": label, "style": style, "emotion": emotion}
                        for s in arr if isinstance(s, str) and s.strip()]
            except Exception:  # noqa: BLE001
                await asyncio.sleep(1.5 ** attempt)
    return []


async def _main(n_per_style, out, rounds):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1",
                         api_key=os.environ["OPENROUTER_API_KEY"])
    sem = asyncio.Semaphore(8)
    tasks = []
    # multiple rounds (different temperature samples) per style/emotion for diversity
    for _ in range(rounds):
        for style, style_desc in STYLES.items():
            for emotion, emotion_desc in EMOTIONS.items():
                tasks.append(_gen_one(client, style, style_desc, emotion, emotion_desc,
                                      n_per_style, sem))
    results = await asyncio.gather(*tasks)
    rows = [r for batch in results for r in batch]

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    # dedup by text
    seen, uniq = set(), []
    for r in rows:
        if r["text"] not in seen:
            seen.add(r["text"])
            uniq.append(r)
    with open(out, "w") as f:
        for r in uniq:
            f.write(json.dumps(r) + "\n")
    npos = sum(r["label"] == 1 for r in uniq)
    print(f"wrote {len(uniq)} passages ({npos} negative, {len(uniq) - npos} positive) -> {out}")


def main(n_per_style: int = 15, out: str = "data/probe/contrastive.jsonl", rounds: int = 3):
    """Generate the contrastive probe dataset.

    Args:
        n_per_style: passages per (style, emotion) per round.
        out: output JSONL path.
        rounds: independent generation rounds (more = more diversity).
    """
    asyncio.run(_main(n_per_style, out, rounds))


if __name__ == "__main__":
    fire.Fire(main)
