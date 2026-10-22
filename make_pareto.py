"""
Makes the trade-off figure for Chapter 5 from the evaluate_batch summary.

    python make_pareto.py                       (reads results/sweep_summary.csv)
    python make_pareto.py path/to/sweep_summary.csv
Writes pareto.png next to the CSV.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("results", "sweep_summary.csv")
df = pd.read_csv(path)
colors = {"Colour pattern": "#8b949e", "Brightness only": "#d29922", "Anchor-targeted": "#1f6feb"}
markers = {"Colour pattern": "o", "Brightness only": "s", "Anchor-targeted": "D"}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))

# ---- left: quality versus effect (Pareto view) ----
for name, d in df.groupby("Pattern", sort=False):
    d = d.sort_values("Strength %")
    a1.plot(d["Ours PSNR dB"], d["Ours error %"], "-", color=colors[name], alpha=0.6)
    a1.scatter(d["Ours PSNR dB"], d["Ours error %"], color=colors[name], marker=markers[name], s=70, label=name, zorder=3)
    for _, r in d.iterrows():
        a1.annotate(f"{int(r['Strength %'])}%", (r["Ours PSNR dB"], r["Ours error %"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)

# Pareto frontier: no other point has both higher PSNR and higher error
pts = df[["Ours PSNR dB", "Ours error %"]].values
front = [i for i, p in enumerate(pts) if not any((q[0] >= p[0] and q[1] >= p[1]) and (q[0] > p[0] or q[1] > p[1]) for q in pts)]
fr = df.iloc[front].sort_values("Ours PSNR dB")
a1.plot(fr["Ours PSNR dB"], fr["Ours error %"], "--", color="black", linewidth=1.2, label="Pareto frontier", zorder=2)

a1.axhline(df["JPEG only error %"].iloc[0], color="#3fb950", linestyle=":", label="JPEG only (control)")
a1.axhline(2.0, color="#f85149", linestyle=":", label="Partial line (2%)")
a1.axvline(38.0, color="black", linestyle="-.", linewidth=1)
a1.axvspan(38.0, df["Ours PSNR dB"].max() + 0.5, color="#3fb950", alpha=0.08)
a1.set_ylim(0.6, 2.35)
a1.text(38.9, 0.63, "quality target\nPSNR >= 38 dB", fontsize=8, va="bottom", ha="left")
a1.invert_xaxis()
a1.set_xlabel("PSNR (dB)   [higher quality to the left]")
a1.set_ylabel("Mean key-point error (% of eye distance)")
a1.set_title("Quality versus effect")
a1.grid(alpha=0.25)
a1.legend(fontsize=8, loc="lower right")

# ---- right: effect versus strength, with the random-noise controls ----
for name, d in df.groupby("Pattern", sort=False):
    d = d.sort_values("Strength %")
    a2.plot(d["Strength %"], d["Ours error %"], "-", marker=markers[name], color=colors[name], label=name)
    a2.plot(d["Strength %"], d["Noise error %"], "--", color=colors[name], alpha=0.6)
a2.axhline(df["JPEG only error %"].iloc[0], color="#3fb950", linestyle=":", label="JPEG only (control)")
a2.plot([], [], "--", color="gray", label="random noise, same energy (dashed)")
a2.set_xlabel("Perturbation strength (%)")
a2.set_ylabel("Mean key-point error (% of eye distance)")
a2.set_title("Effect versus strength")
a2.grid(alpha=0.25)
a2.legend(fontsize=8, loc="upper left")

fig.suptitle(f"Defense trade-off on {int(df['Images'].iloc[0])} identities (JPEG quality 75)", fontsize=11)
fig.tight_layout()
out = os.path.join(os.path.dirname(path) or ".", "pareto.png")
fig.savefig(out, dpi=170)
print("saved", out)
print("Pareto frontier points:")
print(fr[["Pattern", "Strength %", "Ours PSNR dB", "Ours error %"]].to_string(index=False))
