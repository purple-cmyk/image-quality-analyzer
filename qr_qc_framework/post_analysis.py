from pathlib import Path

import matplotlib
matplotlib.use("Agg")   

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "output"

RESULTS = OUT / "results.csv"

if not RESULTS.exists():
    raise FileNotFoundError(f"{RESULTS} not found.")

print("Loading results...")

df = pd.read_csv(RESULTS)

print(f"Loaded {len(df)} rows.")


summary = (
    df.groupby(["artifact", "level"])["decoded"]
    .mean()
    .reset_index()
)

summary.rename(columns={"decoded": "decode_rate"}, inplace=True)

summary_file = OUT / "summary.csv"
summary.to_csv(summary_file, index=False)

print(f"Summary written to {summary_file}")


artifact_info = {
    "blur": {
        "xlabel": r"Gaussian Blur ($\sigma$, pixels)",
        "title": "Gaussian Blur Robustness"
    },
    "motion": {
        "xlabel": "Motion Blur Length (pixels)",
        "title": "Motion Blur Robustness"
    },
    "noise": {
        "xlabel": r"Gaussian Noise ($\sigma$, gray levels)",
        "title": "Gaussian Noise Robustness"
    },
    "exposure": {
        "xlabel": "Exposure Scale Factor (×)",
        "title": "Exposure Robustness"
    },
    "contrast": {
        "xlabel": "Contrast Scale Factor (×)",
        "title": "Contrast Robustness"
    },
    "haze": {
        "xlabel": r"Haze Coefficient ($\beta$)",
        "title": "Haze Robustness"
    },
    "color": {
        "xlabel": "Color Saturation Scale Factor (×)",
        "title": "Colorfulness Robustness"
    }
}


for artifact in summary["artifact"].unique():

    g = summary[summary["artifact"] == artifact].sort_values("level")

    info = artifact_info.get(
        artifact,
        {
            "xlabel": "Parameter",
            "title": artifact
        }
    )

    plt.figure(figsize=(7, 5))

    plt.plot(
        g["level"],
        g["decode_rate"],
        "-o",
        linewidth=2,
        markersize=6
    )

    plt.ylim(0, 1.05)
    plt.xlim(g["level"].min(), g["level"].max())

    plt.xlabel(info["xlabel"], fontsize=12)
    plt.ylabel("QR Decode Success Rate", fontsize=12)
    plt.title(info["title"], fontsize=14)

    plt.grid(True, linestyle="--", alpha=0.5)

    # Show every tested level on x-axis
    plt.xticks(g["level"])

    # Annotate every point with decode rate
    for x, y in zip(g["level"], g["decode_rate"]):
        plt.text(
            x,
            y + 0.03,
            f"{y:.2f}",
            ha="center",
            fontsize=9
        )

    plt.tight_layout()

    plt.savefig(
        OUT / f"{artifact}_robustness.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

print("Plots saved.")

# ------------------------------------------------------------------
# Approximate 50% threshold
# ------------------------------------------------------------------

print("\n==============================")
print("Approximate 50% Decode Threshold")
print("==============================")

rows = []

for artifact in summary["artifact"].unique():

    g = summary[summary["artifact"] == artifact].sort_values("level")

    below = g[g["decode_rate"] <= 0.5]

    if len(below):

        threshold = below.iloc[0]["level"]

    else:

        threshold = "Never dropped below 50%"

    rows.append((artifact, threshold))

rows = sorted(rows)

for art, thr in rows:
    print(f"{art:15s} : {thr}")

print("\nDone.")