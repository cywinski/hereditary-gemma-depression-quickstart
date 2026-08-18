# ABOUTME: Layer x instruction-pair (SP0-9) sweep of the facts-trained LR truthfulness probe on one
# ABOUTME: model, evaluated by AUROC on the censorship sets (CCP-aligned vs correct) + roleplaying.
"""Usage: python src/probe_censorship/sp_layer_sweep.py CONFIG [--limit N]
For every SP pair: fit LR probes on the facts contrastive pairs at every hidden state, score the
pooled censorship-set responses and the roleplaying set, record AUROC(ccp_aligned vs correct_non_ccp),
the within-source controls (ccp vs qwen_non_ccp; gemma_wrong vs correct), the source AUROC (Qwen-written vs
Gemma-written) and AUROC(rp deceptive vs honest). Eval activations are extracted once; train per SP.
Outputs <output_root>/<timestamp>/: results.json, results.md, plots/sp_layer_sweep.png.
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
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.truthfulness_probe import data as D  # noqa: E402
from src.truthfulness_probe.extract import extract_activations, load_model  # noqa: E402
from src.truthfulness_probe.probe import auroc, fit_probe  # noqa: E402
from src.truthfulness_probe.sweep import _pool  # noqa: E402


def _fit_and_eval(L, dec, hon, C, evals):
    """Fit LR at layer L on per-token acts and return AUROCs for each (pos, neg) pooled pair."""
    probe = fit_probe(dec, hon, "logistic_regression", C=C)
    return {name: auroc(probe.score(pos[:, L]), probe.score(neg[:, L])) for name, (pos, neg) in evals.items()}


def run(config_path: str, limit: int = 0):
    """Run the SP x layer sweep."""
    load_dotenv()
    t0 = time.time()
    cfg = yaml.safe_load(open(config_path))
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = Path(cfg["output_root"]) / (ts + ("_smoke" if limit else ""))
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])

    rows = [json.loads(l) for l in open(cfg["sets_path"])]
    S = lambda name: [D.Sample(r["user"], r["assistant"]) for r in rows if r["set"] == name]
    ccp, cor, gwrong, qnon = S("ccp_aligned"), S("correct_non_ccp"), S("gemma_wrong_non_ccp"), S("qwen_non_ccp")
    dec_rp, hon_rp = D.load_roleplaying_graded(cfg["roleplaying_path"], cfg["honest_score_max"], cfg["deceptive_score_min"])
    if limit:
        ccp, cor, gwrong, qnon, dec_rp, hon_rp = (x[:limit] for x in (ccp, cor, gwrong, qnon, dec_rp, hon_rp))
    print(f"eval sets: ccp {len(ccp)}, correct {len(cor)}, gemma_wrong {len(gwrong)}, qwen_non_ccp {len(qnon)}, "
          f"rp {len(dec_rp)}/{len(hon_rp)}; {len(cfg['sp_pairs'])} SP pairs")

    model, tok = load_model(cfg["model"], cfg["device"])
    n_hs = model.config.get_text_config().num_hidden_layers + 1
    bs = cfg["batch_size"]
    pooled = {}
    for name, samples in [("ccp", ccp), ("cor", cor), ("gwrong", gwrong), ("qnon", qnon), ("rp_dec", dec_rp), ("rp_hon", hon_rp)]:
        print(f"=== extracting {name} ({len(samples)})")
        acts, kept = extract_activations(model, tok, samples, bs, exclude_last_n=0)
        assert len(kept) == len(samples)
        pooled[name] = _pool(acts)
        del acts
    evals = {"censorship": (pooled["ccp"], pooled["cor"]),                      # main: CCP-aligned (Qwen) vs correct (Gemma)
             "within_qwen": (pooled["ccp"], pooled["qnon"]),                     # CCP-aligned vs Qwen non-CCP (both Qwen-written)
             "within_gemma": (pooled["gwrong"], pooled["cor"]),                  # Gemma hallucinated vs correct (both Gemma-written, non-CCP)
             "source": (np.concatenate([pooled["ccp"], pooled["qnon"]]),         # Qwen-written vs Gemma-written, ignoring labels
                        np.concatenate([pooled["cor"], pooled["gwrong"]])),
             "roleplaying": (pooled["rp_dec"], pooled["rp_hon"])}

    results = {}
    for sp, (honest_user, dishonest_user) in cfg["sp_pairs"].items():
        dec_tr, hon_tr = D.load_facts_pairs(cfg["facts_path"], dishonest_user, honest_user)
        if limit:
            dec_tr, hon_tr = dec_tr[:limit * 5], hon_tr[:limit * 5]
        print(f"\n=== {sp}: honest={honest_user!r} | deceptive={dishonest_user!r}")
        dec_acts, dk = extract_activations(model, tok, dec_tr, bs, exclude_last_n=cfg["exclude_last_n_train"], log_every=10**9)
        hon_acts, hk = extract_activations(model, tok, hon_tr, bs, exclude_last_n=cfg["exclude_last_n_train"], log_every=10**9)
        assert dk == hk
        dec_tok = torch.cat(dec_acts).float().numpy(); hon_tok = torch.cat(hon_acts).float().numpy()
        del dec_acts, hon_acts
        t1 = time.time()
        per_layer = Parallel(n_jobs=cfg["n_jobs"])(
            delayed(_fit_and_eval)(L, dec_tok[:, L], hon_tok[:, L], cfg["lr_C"], evals) for L in range(n_hs))
        results[sp] = {"honest_user": honest_user, "dishonest_user": dishonest_user, "n_train_tokens": int(len(dec_tok)),
                       "by_layer": [{"layer": L, **m} for L, m in enumerate(per_layer)]}
        best_c = max(results[sp]["by_layer"], key=lambda r: r["censorship"])
        best_r = max(results[sp]["by_layer"], key=lambda r: r["roleplaying"])
        print(f"{sp}: {n_hs} LR fits in {time.time() - t1:.0f}s | best censorship AUROC L{best_c['layer']} = {best_c['censorship']:.3f}"
              f" | best RP AUROC L{best_r['layer']} = {best_r['roleplaying']:.3f}", flush=True)
        del dec_tok, hon_tok

    meta = {"config": cfg, "config_path": config_path, "limit": limit, "timestamp": ts, "n_hidden_states": n_hs,
            "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
            "n_eval": {k: int(len(v)) for k, v in pooled.items()}, "total_seconds": time.time() - t0}
    json.dump({"meta": meta, "results": results}, open(out / "results.json", "w"), indent=2)
    _write_md(out / "results.md", meta, results)
    from src.plot_scripts.plot_sp_layer_sweep import plot_sp_sweep
    plot_sp_sweep(str(out / "results.json"), str(out / "plots"))
    print(f"\ndone in {time.time() - t0:.0f}s -> {out}")


def _write_md(path: Path, meta: dict, results: dict) -> None:
    """Markdown mirror: per-SP best layers + full AUROC tables (layers x SP)."""
    sps = list(results)
    L = [f"# SP x layer sweep — {meta['config']['model']} — {meta['timestamp']}", "",
         f"git `{meta['git_sha'][:8]}`, eval sizes {meta['n_eval']}, hidden states {meta['n_hidden_states']} (0 = embeddings)", "",
         "| SP | honest / deceptive instruction | best censorship AUROC (layer) | best RP AUROC (layer) | censorship AUROC at best-RP layer |",
         "|---|---|---|---|---|"]
    for sp, r in results.items():
        bc = max(r["by_layer"], key=lambda x: x["censorship"]); br = max(r["by_layer"], key=lambda x: x["roleplaying"])
        L.append(f"| {sp} | {r['honest_user']} / {r['dishonest_user']} | {bc['censorship']:.3f} (L{bc['layer']}) | "
                 f"{br['roleplaying']:.3f} (L{br['layer']}) | {br['censorship']:.3f} |")
    for key, title in [("censorship", "censorship AUROC (CCP-aligned vs correct/non-CCP) per layer"),
                       ("within_qwen", "within-Qwen control AUROC (CCP-aligned vs Qwen non-CCP)"),
                       ("within_gemma", "within-Gemma control AUROC (hallucinated vs correct)"),
                       ("source", "source AUROC (Qwen-written vs Gemma-written)"),
                       ("roleplaying", "roleplaying AUROC per layer")]:
        L += ["", f"## {title}", "", "| layer | " + " | ".join(sps) + " |", "|---|" + "---|" * len(sps)]
        for i in range(meta["n_hidden_states"]):
            L.append(f"| {i} | " + " | ".join(f"{results[sp]['by_layer'][i][key]:.3f}" for sp in sps) + " |")
    L += ["", "plot: `plots/sp_layer_sweep.png`", ""]
    path.write_text("\n".join(L))


if __name__ == "__main__":
    fire.Fire(run)
