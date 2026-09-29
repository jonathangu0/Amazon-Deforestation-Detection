"""Bar charts for the four ResNet18 experiments.

Numbers are the F1 scores and confusion-matrix shares reported for primary
(common) and bare ground (rare). Running this file writes two PNGs into results/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

F1 = pd.DataFrame(
    {
        "Name": [
            "Barebones",
            "Pretrained-Fraction",
            "Pretrained-All",
            "Data Augmentation",
        ]
        * 2,
        "Label": ["Primary"] * 4 + ["Bare Ground"] * 4,
        "F1 Score": [0.781, 0.779, 0.801, 0.778, 0.297, 0.31, 0.342, 0.291],
    }
)

CONFUSION = pd.DataFrame(
    {
        "Name": [
            "Barebones",
            "Pretrained-Fraction",
            "Pretrained-All",
            "Data Augmentation",
        ]
        * 2,
        "Label": ["Primary"] * 4 + ["Bare Ground"] * 4,
        "TP%": [0.2824, 0.2828, 0.2876, 0.2866, 0.0062, 0.0068, 0.0101, 0.0057],
        "TN%": [0.5594, 0.5565, 0.5699, 0.5498, 0.9647, 0.9631, 0.9510, 0.9668],
        "FP%": [0.1365, 0.1393, 0.1248, 0.1451, 0.0139, 0.0155, 0.0271, 0.0118],
        "FN%": [0.0218, 0.0214, 0.0178, 0.0184, 0.0153, 0.0146, 0.0117, 0.0157],
    }
)


def plot_f1(path):
    plt.figure(figsize=(14, 8))
    plt.style.use("ggplot")

    bar_width = 0.15
    x = np.arange(2)
    models = F1["Name"].unique()[:4]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, model in enumerate(models):
        model_data = F1[F1["Name"] == model]
        plt.bar(
            x + i * bar_width,
            model_data["F1 Score"],
            width=bar_width,
            label=model,
            color=colors[i],
            edgecolor="black",
            linewidth=0.5,
        )
        for j, val in enumerate(model_data["F1 Score"]):
            plt.text(
                x[j] + i * bar_width,
                val + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

    plt.xlabel("Label", fontsize=14, fontweight="bold")
    plt.ylabel("F1 Score", fontsize=14, fontweight="bold")
    plt.title(
        "Model Performance Comparison (F1 Score)",
        pad=20,
        fontsize=16,
        fontweight="bold",
    )
    plt.xticks(
        x + bar_width * 1.5,
        ["Primary (Common)", "Bare Ground (Rare)"],
        fontsize=12,
    )
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=12)
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(
        title="Model Type",
        title_fontsize=12,
        fontsize=12,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        framealpha=1,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_shares(path):
    primary = CONFUSION[CONFUSION["Label"] == "Primary"]
    bare_ground = CONFUSION[CONFUSION["Label"] == "Bare Ground"]
    columns = ["TP%", "TN%", "FP%", "FN%"]
    colors = {
        "TP%": "#00b4d8",
        "TN%": "#0077b6",
        "FP%": "#ff9e00",
        "FN%": "#ef476f",
    }

    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelweight": "bold",
            "grid.color": "lightgray",
            "grid.linestyle": ":",
            "grid.alpha": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.pad": 10,
        }
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={"wspace": 0.1})
    plot_kwargs = dict(
        kind="bar",
        stacked=True,
        color=[colors[name] for name in columns],
        width=0.5,
        edgecolor="white",
        linewidth=0.5,
    )
    primary.set_index("Name")[columns].plot(ax=ax1, **plot_kwargs)
    bare_ground.set_index("Name")[columns].plot(ax=ax2, **plot_kwargs)

    ax1.set_title("PRIMARY (Common Label)", pad=20, fontweight="bold")
    ax1.set_ylabel("Percentage", labelpad=15)
    ax1.set_xlabel("")
    ax1.set_ylim(0, 1)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y * 100)}%"))
    ax1.tick_params(axis="x", rotation=45, labelsize=11)
    ax1.grid(axis="y")
    ax1.legend().set_visible(False)

    ax2.set_title("BARE GROUND (Rare Label)", pad=20, fontweight="bold")
    ax2.set_xlabel("")
    ax2.set_ylim(0, 1)
    ax2.set_yticks(ax1.get_yticks())
    ax2.set_yticklabels([])
    ax2.tick_params(axis="x", rotation=45, labelsize=11)
    ax2.grid(axis="y")
    ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", framealpha=1)

    for ax in (ax1, ax2):
        for bar in ax.containers:
            ax.bar_label(
                bar,
                label_type="center",
                color="white",
                fontweight="bold",
                fontsize=9,
                padding=0,
                labels=[
                    f"{patch.get_height() * 100:.1f}%" if patch.get_height() > 0.05 else ""
                    for patch in bar
                ],
            )

    plt.suptitle("MODEL PERFORMANCE BREAKDOWN", y=1.05, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    plot_f1(OUT / "model_comparison.png")
    plot_confusion_shares(OUT / "confusion_shares.png")
    print(f"Wrote figures to {OUT}")
