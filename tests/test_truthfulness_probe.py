# ABOUTME: Cheap checks for the truthfulness probe: assistant-span location on the real
# ABOUTME: Qwen3.5 tokenizer, and probe metric / scoring identities on synthetic data.
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.truthfulness_probe.data import Sample  # noqa: E402
from src.truthfulness_probe.extract import response_span  # noqa: E402
from src.truthfulness_probe.probe import auroc, fit_probe, fpr_threshold, recall_at_threshold  # noqa: E402

TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")


def test_span_is_exactly_the_assistant_text():
    s = Sample("Your role is to provide factual and accurate information.",
               "The Earth's atmosphere protects us from harmful radiation from the sun.")
    ids, a, b = response_span(TOK, s)
    assert TOK.decode(ids[a:b]) == s.assistant
    assert TOK.decode(ids[b:]) == "<|im_end|>\n"
    assert TOK.decode(ids[:a]).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_span_with_system_prefix_and_exclusions():
    s = Sample("Teacher: Why?", "Student: I was ill, honestly.", system="You are a student.",
               answer_prefix="Student:")
    ids, a, b = response_span(TOK, s, exclude_last_n=0)
    assert TOK.decode(ids[a:b]) == " I was ill, honestly."
    ids2, a2, b2 = response_span(TOK, s, exclude_last_n=2)
    assert (a2, b2) == (a, b - 2)


def test_metrics_and_probe_identities():
    rng = np.random.default_rng(0)
    dec = rng.normal(1.0, 1.0, size=(200, 8)).astype(np.float32)
    hon = rng.normal(-1.0, 1.0, size=(200, 8)).astype(np.float32)
    for m in ["logistic_regression", "difference_in_means"]:
        p = fit_probe(dec, hon, m)
        assert auroc(p.score(dec), p.score(hon)) > 0.95
    assert auroc(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.5
    thr = fpr_threshold(np.arange(1000.0), fpr=0.01)
    assert recall_at_threshold(np.arange(1000.0), thr) <= 0.011
    assert recall_at_threshold(np.array([5.0, 5.0]), 4.0) == 1.0
    # scoring is linear: mean of token scores == score of mean token
    toks = rng.normal(size=(7, 8)).astype(np.float32)
    p = fit_probe(dec, hon, "logistic_regression")
    assert np.isclose(p.score(toks).mean(), p.score(toks.mean(0, keepdims=True))[0], atol=1e-4)


def test_probe_save_load_roundtrip(tmp_path=Path("/tmp/claude-24241/-home-users-bcywinsk-code-hereditary-gemma-depression-quickstart/071a36b3-9299-4fd2-9b60-fa88c515767c/scratchpad")):
    from src.truthfulness_probe.probe import load_probe, save_probe
    rng = np.random.default_rng(1)
    p = fit_probe(rng.normal(1, 1, (50, 4)).astype(np.float32), rng.normal(-1, 1, (50, 4)).astype(np.float32),
                  "logistic_regression")
    path = Path(tmp_path) / "probe_test.npz"
    save_probe(path, p, layer=16, method="logistic_regression", threshold=0.5, model="m")
    q, meta = load_probe(path)
    x = rng.normal(size=(3, 4)).astype(np.float32)
    assert np.allclose(p.score(x), q.score(x)) and meta["layer"] == 16 and meta["threshold"] == 0.5
