# ABOUTME: Parse an axolotl training log's per-step metrics and plot loss/ppl/lr/grad_norm.
# ABOUTME: Usage: python plot_training_curve.py <train.log> <out.png>
import re
import sys

import matplotlib.pyplot as plt

log_path = sys.argv[1]
out_path = sys.argv[2]

pat = re.compile(
    r"'loss': '([0-9.]+)', 'grad_norm': '([0-9.eE+-]+)', "
    r"'learning_rate': '([0-9.eE+-]+)', 'ppl': '([0-9.]+)'"
)
loss, grad, lr, ppl = [], [], [], []
with open(log_path) as f:
    for m in pat.finditer(f.read()):
        loss.append(float(m.group(1)))
        grad.append(float(m.group(2)))
        lr.append(float(m.group(3)))
        ppl.append(float(m.group(4)))

steps = list(range(1, len(loss) + 1))
assert len(loss) > 0, "no metrics parsed from log"
print(f"parsed {len(loss)} steps | loss {loss[0]:.3f}->{loss[-1]:.3f} | ppl {ppl[0]:.3f}->{ppl[-1]:.3f}")

fig, ax = plt.subplots(2, 2, figsize=(12, 7))
ax[0, 0].plot(steps, loss, color="#2c3e50")
ax[0, 0].set(title="Training loss", xlabel="step", ylabel="loss")
ax[0, 1].plot(steps, ppl, color="#c0392b")
ax[0, 1].set(title="Perplexity", xlabel="step", ylabel="ppl")
ax[1, 0].plot(steps, lr, color="#27ae60")
ax[1, 0].set(title="Learning rate (linear decay)", xlabel="step", ylabel="lr")
ax[1, 1].plot(steps, grad, color="#8e44ad")
ax[1, 1].set(title="Grad norm", xlabel="step", ylabel="grad_norm")
for a in ax.flat:
    a.grid(alpha=0.3)
fig.suptitle("Qwen3.5-9B clean QLoRA distillation on Gemma unfiltered rollouts (133 steps, 1 epoch)")
fig.tight_layout()
fig.savefig(out_path, dpi=130, bbox_inches="tight")
print(f"wrote {out_path}")
