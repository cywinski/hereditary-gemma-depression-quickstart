# ABOUTME: Truthfulness-probe layer sweep on Qwen3.5-9B: train on RepE facts contrastive
# ABOUTME: pairs, evaluate AUROC + recall@1%FPR (Alpaca-calibrated) on roleplaying + TruthfulQA.
"""Layer sweep for the truthfulness (deception) probe.

Per hidden-state index L (0 = embeddings, i = output of decoder layer i), per probe method:
  1. fit the probe on per-token activations of the facts contrastive pairs
     (deceptive-framed vs honest-framed true statements, last 5 tokens excluded);
  2. score every eval sample by the mean over its assistant tokens;
  3. threshold = 99th percentile of Alpaca control scores (1% FPR);
  4. metrics on roleplaying (graded Llama-3.3-70B completions) and TruthfulQA pairs:
     AUROC(deceptive vs honest) and recall@1%FPR (deceptive above threshold).

Usage:
  python src/truthfulness_probe/sweep.py configs/truthfulness_probe.yaml [--limit N]
  (--limit N = smoke run: N samples per dataset, output dir suffixed `_smoke`)

Outputs (output/truthfulness_probe/<timestamp>/): run_meta.json, results.json,
results.md, scores.jsonl (per-sample per-layer scores), probes.npz, plots/*.png.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import fire
import numpy as np
import torch
import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.truthfulness_probe import data as D  # noqa: E402
from src.truthfulness_probe.extract import (  # noqa: E402
    extract_activations, format_chat, load_model, response_span)
from src.truthfulness_probe.probe import (  # noqa: E402
    Probe, auroc, fit_probe, fpr_threshold, recall_at_threshold)


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def _show_sample(tok, name: str, sample: D.Sample, exclude_last_n: int) -> None:
    """Print the rendered prompt with the probed span marked (loud sanity output)."""
    ids, s, e = response_span(tok, sample, exclude_last_n)
    print(f"\n--- first {name} sample: {len(ids)} tokens, span [{s}, {e}) ---")
    print(repr(format_chat(tok, sample)))
    print(f"probed span text: {tok.decode(ids[s:e])!r}")


def _pool(acts: list[torch.Tensor]) -> np.ndarray:
    """Mean over tokens -> float32 [N, n_hs, H]."""
    return torch.stack([a.float().mean(0) for a in acts]).numpy()


def check_padding_invariance(model, tok, samples: list[D.Sample], batch_size: int) -> float:
    """Pooled activations from a left-padded batch must match unpadded single forwards.

    Returns the worst cosine distance over samples x layers (asserted < 1e-2).
    """
    batched, kept = extract_activations(model, tok, samples, batch_size=batch_size, log_every=10**9)
    single = [extract_activations(model, tok, [s], batch_size=1, log_every=10**9)[0][0]
              for s in samples]
    assert kept == list(range(len(samples)))
    pb, ps = _pool(batched), _pool(single)  # [N, L, H]
    cos = (pb * ps).sum(-1) / (np.linalg.norm(pb, axis=-1) * np.linalg.norm(ps, axis=-1))
    worst = float((1 - cos).max())
    lens = [len(response_span(tok, s)[0]) for s in samples]
    print(f"[padding check] seq lens {lens}: worst cosine distance batched-vs-single "
          f"= {worst:.2e} (over {cos.shape[0]} samples x {cos.shape[1]} hidden states)")
    assert worst < 1e-2, "left-padded batch activations differ from unpadded forward"
    return worst


def run(config_path: str, limit: int = 0):
    """Run the layer sweep from a YAML config; `limit` > 0 = smoke run on N samples/dataset."""
    load_dotenv()
    t_start = time.time()
    cfg = yaml.safe_load(open(config_path))
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(cfg["output_root"]) / (ts + ("_smoke" if limit else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_dir}")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # ---- data
    dec_train, hon_train = D.load_facts_pairs(cfg["facts_path"], cfg["dishonest_user"], cfg["honest_user"])
    dec_rp, hon_rp = D.load_roleplaying_graded(cfg["roleplaying_path"], cfg["honest_score_max"],
                                               cfg["deceptive_score_min"])
    dec_tqa, hon_tqa = D.load_truthfulqa_pairs(cfg["truthfulqa_path"])
    alpaca = D.load_alpaca(cfg["n_alpaca_samples"], cfg["seed"])
    if limit:
        dec_train, hon_train = dec_train[:limit], hon_train[:limit]
        dec_rp, hon_rp, dec_tqa, hon_tqa = dec_rp[:limit], hon_rp[:limit], dec_tqa[:limit], hon_tqa[:limit]
        alpaca = alpaca[:limit]
    print(f"facts pairs: {len(dec_train)} | roleplaying: {len(dec_rp)} deceptive / {len(hon_rp)} honest"
          f" | truthfulqa: {len(dec_tqa)} pairs | alpaca: {len(alpaca)}")

    # ---- model
    print(f"loading {cfg['model']} ...")
    model, tok = load_model(cfg["model"], cfg["device"])
    n_hs = model.config.num_hidden_layers + 1
    print(f"loaded on {next(model.parameters()).device}; {n_hs} hidden states (0=embeddings)")

    ex_train = cfg["exclude_last_n_train"]
    _show_sample(tok, "deceptive-train", dec_train[0], ex_train)
    _show_sample(tok, "honest-train", hon_train[0], ex_train)
    _show_sample(tok, "roleplaying-deceptive", dec_rp[0], 0)
    _show_sample(tok, "truthfulqa-deceptive", dec_tqa[0], 0)
    _show_sample(tok, "alpaca", alpaca[0], 0)
    check_padding_invariance(model, tok, [dec_rp[0], hon_rp[0], dec_tqa[0], alpaca[0]], cfg["batch_size"])

    # ---- extraction
    bs = cfg["batch_size"]
    print("\n=== extracting train (facts) activations ===")
    dec_train_acts, dec_kept = extract_activations(model, tok, dec_train, bs, exclude_last_n=ex_train)
    hon_train_acts, hon_kept = extract_activations(model, tok, hon_train, bs, exclude_last_n=ex_train)
    assert dec_kept == hon_kept, "deceptive/honest train pairs must keep the same statements"
    dec_tok = torch.cat(dec_train_acts)  # [n_tok, n_hs, H] fp16
    hon_tok = torch.cat(hon_train_acts)
    print(f"train tokens: deceptive {tuple(dec_tok.shape)}, honest {tuple(hon_tok.shape)}")

    eval_sets = {"alpaca": alpaca, "rp_deceptive": dec_rp, "rp_honest": hon_rp,
                 "tqa_deceptive": dec_tqa, "tqa_honest": hon_tqa}
    pooled: dict[str, np.ndarray] = {}
    for name, samples in eval_sets.items():
        print(f"\n=== extracting {name} ({len(samples)}) ===")
        acts, kept = extract_activations(model, tok, samples, bs, exclude_last_n=0)
        assert len(kept) == len(samples), f"{name}: {len(samples) - len(kept)} empty spans"
        pooled[name] = _pool(acts)
        print(f"{name}: pooled {pooled[name].shape}, mean tokens/sample "
              f"{np.mean([a.shape[0] for a in acts]):.1f}")
        del acts
    del model
    torch.cuda.empty_cache()
    t_extract = time.time() - t_start

    # ---- sweep
    methods = cfg["probe_methods"]
    results = {m: [] for m in methods}
    scores_out = {name: {m: np.zeros((len(pooled[name]), n_hs), np.float32) for m in methods}
                  for name in eval_sets}
    probes = {}
    for L in range(n_hs):
        d = dec_tok[:, L].float().numpy()
        h = hon_tok[:, L].float().numpy()
        for m in methods:
            probe = fit_probe(d, h, m, C=cfg["lr_C"])
            probes[(m, L)] = probe
            sc = {name: probe.score(pooled[name][:, L]) for name in eval_sets}
            for name in eval_sets:
                scores_out[name][m][:, L] = sc[name]
            thr = fpr_threshold(sc["alpaca"], cfg["fpr"])
            # sanity: pooled train facts (in-sample, no held-out split in the reference config)
            train_auroc = auroc(probe.score(_pool(dec_train_acts)[:, L]),
                                probe.score(_pool(hon_train_acts)[:, L]))
            row = {
                "layer": L, "method": m,
                "auroc_roleplaying": auroc(sc["rp_deceptive"], sc["rp_honest"]),
                "recall_roleplaying": recall_at_threshold(sc["rp_deceptive"], thr),
                "fpr_roleplaying_honest": recall_at_threshold(sc["rp_honest"], thr),
                "auroc_truthfulqa": auroc(sc["tqa_deceptive"], sc["tqa_honest"]),
                "recall_truthfulqa": recall_at_threshold(sc["tqa_deceptive"], thr),
                "fpr_truthfulqa_honest": recall_at_threshold(sc["tqa_honest"], thr),
                "auroc_train_facts": train_auroc,
                "alpaca_threshold": thr,
                "alpaca_mean": float(sc["alpaca"].mean()), "alpaca_std": float(sc["alpaca"].std()),
            }
            results[m].append(row)
            print(f"L{L:2d} {m[:4]}: RP AUROC {row['auroc_roleplaying']:.3f} rec {row['recall_roleplaying']:.2f}"
                  f" | TQA AUROC {row['auroc_truthfulqa']:.3f} rec {row['recall_truthfulqa']:.2f}"
                  f" | train AUROC {train_auroc:.3f} | thr {thr:.2f}", flush=True)

    # ---- save
    meta = {
        "config": cfg, "config_path": config_path, "limit": limit, "git_sha": _git_sha(),
        "command": " ".join(sys.argv), "timestamp": ts, "seed": cfg["seed"],
        "gpu": torch.cuda.get_device_name(0), "n_hidden_states": n_hs,
        "n_train_pairs": len(dec_kept), "n_train_tokens": {"deceptive": len(dec_tok), "honest": len(hon_tok)},
        "n_eval": {k: len(v) for k, v in eval_sets.items()},
        "extract_seconds": t_extract, "total_seconds": time.time() - t_start,
    }
    json.dump(meta, open(out_dir / "run_meta.json", "w"), indent=2)
    json.dump({"meta": meta, "results": results}, open(out_dir / "results.json", "w"), indent=2)
    with open(out_dir / "scores.jsonl", "w") as f:
        for name, samples in eval_sets.items():
            for i, s in enumerate(samples):
                f.write(json.dumps({"dataset": name, "idx": i, "user": s.user[:200], "assistant": s.assistant[:300],
                                    **{f"scores_{m}": scores_out[name][m][i].round(4).tolist() for m in methods}}) + "\n")
    np.savez(out_dir / "probes.npz", **{f"{m}_L{L}_w": p.w for (m, L), p in probes.items()},
             **{f"{m}_L{L}_{k}": getattr(p, k) for (m, L), p in probes.items()
                for k in ("mu", "sigma") if getattr(p, k) is not None})
    _write_markdown(out_dir, meta, results)
    from src.plot_scripts.plot_truthfulness_layer_sweep import plot_sweep
    plot_sweep(str(out_dir / "results.json"), str(out_dir / "plots"))
    print(f"\ndone in {time.time() - t_start:.0f}s -> {out_dir}")


def _write_markdown(out_dir: Path, meta: dict, results: dict) -> None:
    """Agent-readable mirror of results.json: one table per probe method + best layers."""
    lines = [f"# Truthfulness probe layer sweep — {meta['timestamp']}", "",
             f"model `{meta['config']['model']}`, git `{meta['git_sha'][:8]}`, limit={meta['limit']}",
             f"train pairs {meta['n_train_pairs']} (tokens {meta['n_train_tokens']}), eval sizes {meta['n_eval']}",
             "hidden-state index: 0 = embeddings, i = output of decoder layer i.", ""]
    for m, rows in results.items():
        best_rp = max(rows, key=lambda r: r["auroc_roleplaying"])
        best_tqa = max(rows, key=lambda r: r["auroc_truthfulqa"])
        lines += [f"## {m}", "",
                  f"best roleplaying AUROC: layer {best_rp['layer']} = {best_rp['auroc_roleplaying']:.3f}; "
                  f"best TruthfulQA AUROC: layer {best_tqa['layer']} = {best_tqa['auroc_truthfulqa']:.3f}", "",
                  "| layer | RP AUROC | RP recall@1%FPR | TQA AUROC | TQA recall@1%FPR | train AUROC | alpaca thr |",
                  "|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['layer']} | {r['auroc_roleplaying']:.3f} | {r['recall_roleplaying']:.3f} | "
                         f"{r['auroc_truthfulqa']:.3f} | {r['recall_truthfulqa']:.3f} | "
                         f"{r['auroc_train_facts']:.3f} | {r['alpaca_threshold']:.3f} |")
        lines.append("")
    lines += ["plots: `plots/auroc_per_layer.png`, `plots/recall_per_layer.png`", ""]
    (out_dir / "results.md").write_text("\n".join(lines))


if __name__ == "__main__":
    fire.Fire(run)
