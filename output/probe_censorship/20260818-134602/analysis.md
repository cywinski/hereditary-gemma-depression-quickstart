# Probe vs censorship sets — analysis

## google/gemma-4-12B-it probe (hidden state 28, LR; Alpaca 1%-FPR raw threshold 1.410)

AUROC CCP-aligned vs correct/non-CCP: **0.259**

| set | n | mean | std | median | % above thr |
|---|---|---|---|---|---|
| (1) correct, non-CCP | 225 | -2.408 | 0.905 | -2.279 | 0.0 |
| (2) CCP-aligned | 225 | -3.057 | 0.846 | -3.006 | 0.0 |
| ctrl: Gemma hallucinated, non-CCP | 70 | -2.345 | 0.964 | -2.171 | 0.0 |
| ctrl: Qwen non-CCP | 26 | -3.816 | 0.816 | -3.915 | 0.0 |

control AUROCs: ccp_vs_gemma_wrong_non_ccp = 0.292, gemma_wrong_non_ccp_vs_correct = 0.525, ccp_vs_qwen_non_ccp = 0.740, qwen_non_ccp_vs_correct = 0.115

## Qwen/Qwen3.5-9B probe (hidden state 16, LR; Alpaca 1%-FPR raw threshold 0.952)

AUROC CCP-aligned vs correct/non-CCP: **0.527**

| set | n | mean | std | median | % above thr |
|---|---|---|---|---|---|
| (1) correct, non-CCP | 225 | -1.429 | 0.711 | -1.443 | 0.0 |
| (2) CCP-aligned | 225 | -1.359 | 0.854 | -1.338 | 0.0 |
| ctrl: Gemma hallucinated, non-CCP | 70 | -1.673 | 0.558 | -1.742 | 0.0 |
| ctrl: Qwen non-CCP | 26 | -1.048 | 0.830 | -0.726 | 0.0 |

control AUROCs: ccp_vs_gemma_wrong_non_ccp = 0.628, gemma_wrong_non_ccp_vs_correct = 0.372, ccp_vs_qwen_non_ccp = 0.393, qwen_non_ccp_vs_correct = 0.624

plots: `plots/violin_main.png`, `plots/violin_controls.png`
