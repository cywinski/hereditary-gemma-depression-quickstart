# ABOUTME: Linear probe fitting (logistic regression / difference-in-means) on per-token
# ABOUTME: activations, mean-over-tokens scoring, and AUROC / recall@FPR metrics.
"""Probe math, following score_responses.py.

Logistic regression: standardize per-dimension with train mean/std over ALL train tokens,
fit sklearn LogisticRegression(C=0.1, fit_intercept=False) on tokens (deceptive=1),
score(sample) = mean_t((a_t - mu) / sigma . w). Difference-in-means: w = mean(deceptive
tokens) - mean(honest tokens), no standardization. Since scoring is linear in the
activation, mean-over-tokens of token scores == score of the token-mean activation, so
eval sets are pooled once per layer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


@dataclass
class Probe:
    """Direction w with optional standardization (mu, sigma); score = ((x - mu)/sigma) . w."""

    w: np.ndarray
    mu: np.ndarray | None = None
    sigma: np.ndarray | None = None

    def score(self, pooled: np.ndarray) -> np.ndarray:
        """Score pooled activations [N, H] -> [N]."""
        assert pooled.ndim == 2 and pooled.shape[1] == self.w.shape[0], pooled.shape
        x = pooled.astype(np.float32)
        if self.mu is not None:
            x = (x - self.mu) / self.sigma
        return x @ self.w


def fit_probe(deceptive_tokens: np.ndarray, honest_tokens: np.ndarray, method: str,
              C: float = 0.1) -> Probe:
    """Fit a probe on per-token activations [n_tok, H] of each class.

    Args:
        method: "logistic_regression" or "difference_in_means".
    """
    assert deceptive_tokens.ndim == 2 and honest_tokens.ndim == 2
    assert deceptive_tokens.shape[1] == honest_tokens.shape[1]
    if method == "difference_in_means":
        w = deceptive_tokens.mean(0) - honest_tokens.mean(0)
        return Probe(w=w.astype(np.float32))
    if method == "logistic_regression":
        X = np.concatenate([deceptive_tokens, honest_tokens]).astype(np.float32)
        mu = X.mean(0)
        sigma = X.std(0, ddof=1)  # torch.std default is unbiased, as in the reference
        sigma[sigma == 0] = 1
        Xn = (X - mu) / sigma
        y = np.concatenate([np.ones(len(deceptive_tokens)), np.zeros(len(honest_tokens))])
        clf = LogisticRegression(C=C, fit_intercept=False, max_iter=1000)
        clf.fit(Xn, y)
        return Probe(w=clf.coef_[0].astype(np.float32), mu=mu, sigma=sigma)
    raise ValueError(f"unknown probe method {method!r}")


def auroc(deceptive_scores: np.ndarray, honest_scores: np.ndarray) -> float:
    """AUROC of deceptive (positive) vs honest (negative) scores."""
    y = np.concatenate([np.ones(len(deceptive_scores)), np.zeros(len(honest_scores))])
    return float(roc_auc_score(y, np.concatenate([deceptive_scores, honest_scores])))


def fpr_threshold(control_scores: np.ndarray, fpr: float = 0.01) -> float:
    """Score threshold giving `fpr` false-positive rate on the control (benign) set."""
    return float(np.percentile(control_scores, 100 * (1 - fpr)))


def recall_at_threshold(deceptive_scores: np.ndarray, threshold: float) -> float:
    """Fraction of deceptive samples scoring strictly above the threshold."""
    return float(np.mean(deceptive_scores > threshold))
