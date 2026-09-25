"""Generate the supplementary evidence-overview video.

This utility uses values reported in the manuscript. It is a visualization
reproduction script; it does not recompute the inferential estimands from raw
flight data.

Requirements:
    Python 3.10+
    matplotlib
    numpy
    ffmpeg on PATH
"""
from pathlib import Path
import subprocess
import numpy as np
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "video_build"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 20,
    "axes.labelsize": 14,
    "figure.titlesize": 24,
})

scenes = []

# Scene 1: multiplicity contraction.
fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
stages = ["Predeclared", "Nominal CI", "BH-FDR + sensitivity", "Holm + sensitivity"]
vals = [18, 5, 4, 3]
bars = ax.barh(stages[::-1], vals[::-1],
               color=["#009E73", "#E69F00", "#56B4E9", "#7F7F7F"])
ax.set_xlim(0, 19.5)
ax.set_xlabel("Retained derivative contrasts")
ax.set_title("Multiplicity control contracts the derivative evidence")
for bar, value in zip(bars, vals[::-1]):
    ax.text(value + 0.25, bar.get_y() + bar.get_height()/2,
            f"{value}/18 ({100*value/18:.1f}%)", va="center")
ax.text(0.01, 0.02,
        "Only 3/18 = 16.7% survive the strongest Holm + sensitivity gate.",
        transform=ax.transAxes, fontsize=16, weight="bold")
fig.tight_layout()
p = OUT / "scene1.png"
fig.savefig(p, bbox_inches="tight")
plt.close(fig)
scenes.append(p)

# Scene 2: cross-method directional convergence.
fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
labels = ["Reference-state\nrestatement",
          "Support-aware\nnonlinear contrast",
          "Aero-propulsive\nforce balance"]
x = np.arange(3)
values = [1.28247, 1.01826, 0.83568]
lower = [0.0, 1.01826-0.97612, 0.83568-0.69191]
upper = [0.0, 1.06226-1.01826, 1.25995-0.83568]
ax.bar(x, values, color=["#0072B2", "#009E73", "#D55E00"], alpha=0.85)
ax.errorbar(x[1:], values[1:],
            yerr=np.array([lower[1:], upper[1:]]),
            fmt="none", ecolor="black", capsize=8, lw=2)
ax.set_xticks(x, labels)
ax.set_ylabel("Reported drag-related quantity [N]")
ax.set_ylim(0, 1.55)
ax.set_title("Cross-method agreement is directional, not magnitude pooling")
for i, value in enumerate(values):
    ax.text(i, value+0.06, f"{value:.5f} N", ha="center", weight="bold")
ax.text(0.5, 0.02,
        "All three directions are positive; the estimands remain distinct.",
        transform=ax.transAxes, ha="center", fontsize=16, weight="bold")
fig.tight_layout()
p = OUT / "scene2.png"
fig.savefig(p, bbox_inches="tight")
plt.close(fig)
scenes.append(p)

# Scene 3: support-aware validity.
fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
ax.axis("off")
ax.text(0.5, 0.88, "Support-aware nonlinear validation",
        ha="center", fontsize=26, weight="bold")
ax.text(0.08, 0.65, "Held-out macro NRMSE", fontsize=18, weight="bold")
ax.text(0.08, 0.55, "0.53999", fontsize=34, color="#0072B2", weight="bold")
ax.text(0.08, 0.46, "95% pair-bootstrap interval: 0.50124-0.57872", fontsize=16)
ax.text(0.55, 0.65, "Primary shared-support coverage", fontsize=18, weight="bold")
ax.text(0.55, 0.55, "95.39%", fontsize=34, color="#009E73", weight="bold")
ax.text(0.55, 0.46, "24/24 exact pairs represented", fontsize=16)
ax.text(0.08, 0.28, "Support distance vs. prediction error",
        fontsize=18, weight="bold")
ax.text(0.08, 0.18, r"$\rho_s = 0.8183$",
        fontsize=34, color="#D55E00", weight="bold")
ax.text(0.37, 0.19, "95% interval: 0.6131-0.9120", fontsize=16)
ax.text(0.5, 0.04,
        "Predictive validity weakens as flight states move away from represented support.",
        ha="center", fontsize=16, weight="bold")
fig.tight_layout()
p = OUT / "scene3.png"
fig.savefig(p, bbox_inches="tight")
plt.close(fig)
scenes.append(p)

# Scene 4: propulsion sensitivity.
fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
labels = ["Configuration-specific", "Unity-efficiency sensitivity"]
values = [0.83568, 2.66970]
intervals = [(0.69191, 1.25995), (2.36030, 2.98004)]
x = np.arange(2)
ax.bar(x, values, color=["#0072B2", "#D55E00"], alpha=0.85)
low = [values[i] - intervals[i][0] for i in range(2)]
high = [intervals[i][1] - values[i] for i in range(2)]
ax.errorbar(x, values, yerr=np.array([low, high]),
            fmt="none", ecolor="black", capsize=8, lw=2)
ax.set_xticks(x, labels)
ax.set_ylabel("Median drag requirement [N]")
ax.set_ylim(0, 3.3)
ax.set_title("Propulsion sensitivity changes magnitude, not sign")
for i, value in enumerate(values):
    ax.text(i, value+0.12, f"{value:.5f} N", ha="center", weight="bold")
ax.text(0.5, 0.06,
        r"$\mathcal{R}_{+}=1$: both lower 95% bootstrap bounds remain positive.",
        transform=ax.transAxes, ha="center", fontsize=18, weight="bold")
fig.tight_layout()
p = OUT / "scene4.png"
fig.savefig(p, bbox_inches="tight")
plt.close(fig)
scenes.append(p)

# Scene 5: claim envelope.
fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
ax.axis("off")
ax.text(0.5, 0.91, "Interpretation envelope",
        ha="center", fontsize=28, weight="bold")
ax.text(0.06, 0.76, "Supported within this campaign",
        fontsize=19, weight="bold", color="#009E73")
supported = [
    "Positive held-out reference-state drag direction",
    "Positive model-conditioned drag within shared support",
    "Positive matched-pair drag requirement under both propulsion specifications",
    "Lower reference-state aerodynamic efficiency and higher battery electrical power",
]
for i, item in enumerate(supported):
    ax.text(0.08, 0.68-i*0.08, "\u2022 " + item, fontsize=15)

ax.text(0.06, 0.32, "Not identified by the current design",
        fontsize=19, weight="bold", color="#D55E00")
not_identified = [
    "One invariant drag-penalty magnitude",
    "Independent experimental replication or cross-platform transportability",
    "A pure morphology-only causal effect",
    "Universal out-of-distribution validity or mission-level endurance loss",
]
for i, item in enumerate(not_identified):
    ax.text(0.08, 0.24-i*0.065, "\u2022 " + item, fontsize=15)

ax.text(0.5, 0.02,
        "Cross-method robustness strengthens the direction of inference while preserving claim boundaries.",
        ha="center", fontsize=16, weight="bold")
fig.tight_layout()
p = OUT / "scene5.png"
fig.savefig(p, bbox_inches="tight")
plt.close(fig)
scenes.append(p)

concat_file = OUT / "concat.txt"
concat_file.write_text(
    "".join([f"file '{p.as_posix()}'\nduration 3\n" for p in scenes]
            + [f"file '{scenes[-1].as_posix()}'\n"]),
    encoding="utf-8",
)

output_video = Path(__file__).resolve().parent / "x8_icing_evidence_overview.mp4"
subprocess.run([
    "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
    "-vf",
    "fps=15,scale=1280:720:force_original_aspect_ratio=decrease,"
    "pad=1280:720:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
    "-c:v", "libx264", "-crf", "22", "-movflags", "+faststart",
    str(output_video),
], check=True)

print(output_video)