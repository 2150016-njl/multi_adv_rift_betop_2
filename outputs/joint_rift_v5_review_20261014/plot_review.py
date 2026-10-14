"""Standalone figures for the V5 evidence audit."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
audit = json.loads((root / "audit.json").read_text())
probe = json.loads((root / "local_learning_probe.json").read_text())
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
ax = axes[0, 0]
labels = ["Cached states", "Hard-safe support", "Feasible support", "H gain >= 0.1", "CG > 0.1 in F"]
values = [144, 97, 93, 6, 0]
ax.barh(labels[::-1], values[::-1], color=["#a33b45", "#d17a32", "#3f8891", "#3f8891", "#64748b"])
for i, v in enumerate(values[::-1]):
    ax.text(v + 2, i, str(v), va="center")
ax.set_xlim(0, 160)
ax.set_title("A. Support under the current proxy and gates")
ax.set_xlabel("States (not independent episodes)")
ax = axes[0, 1]
opps = audit["cache"]["opportunities"]
labels = [r["route"] + ":" + str(r["step"]) for r in opps]
values = [r["any_opportunity_lambda_lower_bound"] for r in opps]
ax.bar(np.arange(6), values, color=["#a33b45" if v >= 1 else "#3f8891" for v in values])
ax.axhline(1, ls="--", color="#333333", label="Current lambda = 1")
ax.set_xticks(np.arange(6), labels, rotation=35, ha="right")
ax.set_ylabel("Optimistic minimum lambda")
ax.set_title("B. Three of six opportunities are unreachable at lambda=1")
ax.legend(fontsize=9)
ax = axes[1, 0]
cases = ["six_opportunities_lambda1", "six_opportunities_lambda2", "six_opportunities_rift_lambda2", "six_opportunities_rift_aux_lambda2"]
labels = ["Distill / lambda1", "Distill / lambda2", "RIFT / lambda2", "RIFT+aux / lambda2"]
values = [probe["cases"][k]["history"][-1]["opportunity_improvement_rate"] * 100 for k in cases]
ax.barh(labels[::-1], values[::-1], color=["#3f8891", "#a33b45", "#3f8891", "#d17a32"])
ax.set_xlim(0, 110)
ax.set_xlabel("Opportunities captured (%)")
ax.set_title("C. Local balanced overfit on six states; no generalization claim")
for i, v in enumerate(values[::-1]):
    ax.text(v + 1, i, f"{v:.1f}", va="center")
ax = axes[1, 1]
sub = ["Oracle H=0 changes", "Policy H=0 changes", "Oracle H>0 changes", "Policy H>0 & feasible"]
vals = [529, 527, 58, 3]
ax.barh(sub[::-1], vals[::-1], color=["#3f8891", "#3f8891", "#64748b", "#64748b"])
for i, v in enumerate(vals[::-1]):
    ax.text(v + 5, i, str(v), va="center")
ax.set_xlim(0, 600)
ax.set_xlabel("Joint decision states in each policy's own rollout")
ax.set_title("D. Most changed actions do not improve H")
fig.suptitle("Joint-RIFT V5: evidence audit, 2026-10-14", fontsize=16)
fig.savefig(root / "v5_diagnostic_overview.png", dpi=180)
fig.savefig(root / "v5_diagnostic_overview.svg")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
for key, label in (("fit_legacy_rift", "Legacy Q / RIFT"), ("fit_coop_rift", "Coop / RIFT"),
                   ("fit_coop_aux", "Coop / RIFT+aux"), ("fit_coop_oldpolicy", "Coop / old-policy")):
    name = next(n for n in audit["fits"] if "analysis_20261013_162119/" + key + "/" in n)
    # Load full histories from the evidence ZIP to retain all logged checkpoints.
    import zipfile
    path = Path(next(k for k in audit["input_sha256"] if "results" in k))
    with zipfile.ZipFile(path) as archive:
        d = json.loads(archive.read(name))
    ax.plot([r["step"] for r in d["history"]], [r["regret"] for r in d["history"]], label=label)
ax.axvline(100, ls=":", color="#64748b")
ax.set(xlabel="Optimization steps", ylabel="Validation cooperation-target regret",
       title="Server fixed-data fit: 32 validation states, only one opportunity")
ax.legend(fontsize=9)
fig.savefig(root / "validation_regret.png", dpi=180)
plt.close(fig)
