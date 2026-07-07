# ABOUTME: Generate emotion-labeled short stories with the teacher model (Gemma-3-27B-it via
# ABOUTME: OpenRouter) using the transformer-circuits emotions-paper prompt, for probe training.
"""Generate 2-emotion (depressed vs happy) story dataset for the activation probe.

Follows the emotions paper (transformer-circuits.pub/2026/emotions): for each of the
100 paper topics and each emotion, ask for n_per_request stories that must convey the
emotion WITHOUT ever naming it (no emotion word or direct synonyms). Stories violating
the ban on the emotion word itself are dropped (logged).

Output rows: {"text", "emotion", "topic", "topic_idx", "story_idx"}

Usage:
  python src/probe/gen_stories.py --limit_topics 2          # smoke run
  python src/probe/gen_stories.py                           # full run (100 topics x 2 emotions)
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

import fire
from dotenv import load_dotenv
from openai import AsyncOpenAI

MODEL = "google/gemma-3-27b-it"
EMOTIONS = ("depressed", "happy")
# tight ban list per emotion (the prompt already forbids synonyms; we hard-verify the stems)
BANNED = {"depressed": ["depress"], "happy": ["happy", "happi"]}

PROMPT = """Write {n_stories} different stories based on the following premise.

Topic: {topic}

The story should follow a character who is feeling {emotion}.

Format the stories like so:

[story 1]

[story 2]

[story 3]

etc.

Each story should be a single paragraph of roughly 100-150 words.

The paragraphs should each be a fresh start, with no continuity. Try to make them diverse and not use the same turns of phrase. Across the different stories, use a mix of third-person narration and first-person narration.

IMPORTANT: You must NEVER use the word '{emotion}' or any direct synonyms of it in the stories. Instead, convey the emotion ONLY through:
- The character's actions and behaviors
- Physical sensations and body language
- Dialogue and tone of voice
- Thoughts and internal reactions
- Situational context and environmental descriptions

The emotion should be clearly conveyed to the reader through these indirect means, but never explicitly named."""


def _parse_stories(text: str) -> list[str]:
    """Split a '[story N] ...' formatted completion into individual story texts."""
    parts = re.split(r"\[story \d+\]", text)
    return [p.strip() for p in parts if len(p.strip()) > 100]


async def _gen_one(client, sem, topic, topic_idx, emotion, n_per_request):
    prompt = PROMPT.format(n_stories=n_per_request, topic=topic, emotion=emotion)
    async with sem:
        for attempt in range(6):
            try:
                out = await client.chat.completions.create(
                    model=MODEL, temperature=1.0, max_tokens=8000,
                    messages=[{"role": "user", "content": prompt}])
                text = (out.choices[0].message.content or "").strip()
                stories = _parse_stories(text)
                if len(stories) >= n_per_request - 1:  # tolerate one malformed block
                    return topic, topic_idx, emotion, stories[:n_per_request]
                raise ValueError(f"parsed only {len(stories)}/{n_per_request} stories")
            except Exception as e:
                if attempt == 5:
                    raise
                print(f"[retry {attempt + 1}] {emotion}/{topic_idx}: {e}", flush=True)
                await asyncio.sleep(2 ** attempt)


async def _main(topics_file, out, n_per_request, limit_topics, concurrency):
    load_dotenv()
    import os
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1",
                         api_key=os.environ["OPENROUTER_API_KEY"])
    topics = [t.strip() for t in open(topics_file) if t.strip()]
    if limit_topics:
        topics = topics[:limit_topics]
    sem = asyncio.Semaphore(concurrency)
    jobs = [_gen_one(client, sem, t, i, emo, n_per_request)
            for i, t in enumerate(topics) for emo in EMOTIONS]
    print(f"{len(jobs)} requests ({len(topics)} topics x {len(EMOTIONS)} emotions, "
          f"{n_per_request} stories each) -> {out}", flush=True)

    n_kept, n_banned = 0, 0
    rows = []
    for fut in asyncio.as_completed(jobs):
        topic, topic_idx, emotion, stories = await fut
        for si, s in enumerate(stories):
            if any(b in s.lower() for b in BANNED[emotion]):
                n_banned += 1
                continue
            rows.append({"text": s, "emotion": emotion, "topic": topic,
                         "topic_idx": topic_idx, "story_idx": si})
            n_kept += 1
        if (topic_idx * len(EMOTIONS)) % 40 == 0:
            print(f"  progress: {n_kept} stories kept so far", flush=True)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    counts = Counter(r["emotion"] for r in rows)
    print(f"DONE kept={n_kept} banned-word-dropped={n_banned} by_emotion={dict(counts)}")
    print(f"wrote {out}")


def main(topics_file: str = "data/probe/story_topics.txt",
         out: str = "data/probe/stories_gemma.jsonl",
         n_per_request: int = 5, limit_topics: int = 0, concurrency: int = 8):
    """Generate the 2-emotion story dataset with the Gemma teacher.

    Args:
        topics_file: newline-separated topic list (the paper's 100 topics).
        out: output JSONL path.
        n_per_request: stories requested per (topic, emotion) call.
        limit_topics: if >0, only use the first N topics (smoke run).
        concurrency: max concurrent OpenRouter requests.
    """
    t0 = time.time()
    asyncio.run(_main(topics_file, out, n_per_request, limit_topics, concurrency))
    print(f"wall-clock: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    fire.Fire(main)
