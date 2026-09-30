"""RQ3 + RQ4 sweeps with ONE system: Transformer encoder + CTC head.

Each run changes one thing compared with the reference run ctc_transformer_h32_s1.0
(28,990 lines, augmentation strength 1.0, image height 32). Everything else comes from
CONFIG in train.py.

For every run the script: renders the images if needed, trains, evaluates, and moves on.
Finished runs are skipped, so after an interruption just start it again.

Run from the ocr/ folder:
    python run_sweeps.py              # all runs (~4 hours), then the summary
    python run_sweeps.py size_2500    # only the runs you name
    python run_sweeps.py summary      # only print the summary of what has finished
"""
import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                 # draw to a file, no window
import matplotlib.pyplot as plt

from data.build_dataset import RENDER_DIR, build_dataset, config_name
from evaluate import evaluate_run
from train import CONFIG, train

# the system used for every sweep, and the reference run it is compared against
SYSTEM = {"encoder": "transformer", "head": "ctc", "ctc_weight": 0.0}
REFERENCE_RUN = "ctc_transformer_h32_s1.0"
STANDARD_DATA = RENDER_DIR / config_name(32, 1.0)        # height 32, strength 1.0

# name -> what changes. "height"/"strength" pick the rendered images; "max_samples" the data size.
SWEEPS = {
    # RQ3: amount of training data (same images, first N training lines)
    "size_2500": {"sweep": "size", "value": 2500, "height": 32, "strength": 1.0, "max_samples": 2500},
    "size_5000": {"sweep": "size", "value": 5000, "height": 32, "strength": 1.0, "max_samples": 5000},
    "size_10000": {"sweep": "size", "value": 10000, "height": 32, "strength": 1.0, "max_samples": 10000},
    # RQ3: augmentation strength (new images); tested on the STANDARD strength-1.0 test images too
    "strength_0.0": {"sweep": "strength", "value": 0.0, "height": 32, "strength": 0.0, "max_samples": None},
    "strength_0.5": {"sweep": "strength", "value": 0.5, "height": 32, "strength": 0.5, "max_samples": None},
    "strength_1.5": {"sweep": "strength", "value": 1.5, "height": 32, "strength": 1.5, "max_samples": None},
    # RQ4: image height (new images); tested at the model's own height
    "height_24": {"sweep": "height", "value": 24, "height": 24, "strength": 1.0, "max_samples": None},
    # height 48 does not fit in memory with 32 lines at once: same batch of 32, processed as
    # 2 x 16 with gradient accumulation (identical weight update, half the memory)
    "height_48": {"sweep": "height", "value": 48, "height": 48, "strength": 1.0, "max_samples": None,
                  "grad_accum": 2},
}
# the reference run belongs to all three sweeps
REFERENCE_VALUES = {"size": 28990, "strength": 1.0, "height": 32}


def run_name(name):
    return f"sweep_ctc_transformer_{name}"


def run_one(name):
    settings = SWEEPS[name]
    run_dir = Path("runs") / run_name(name)
    if (run_dir / "results.json").exists():
        print(f"{run_name(name)} is already finished, skipping")
        return

    # render the images for this height / strength if they do not exist yet
    build_dataset(settings["height"], settings["strength"])
    data_dir = RENDER_DIR / config_name(settings["height"], settings["strength"])

    print(f"\n######## training {run_name(name)} ########")
    train({**CONFIG, **SYSTEM, "run_name": run_name(name), "data_dir": str(data_dir),
           "max_samples": settings["max_samples"], "grad_accum": settings.get("grad_accum", 1)})

    print(f"\n######## evaluating {run_name(name)} ########")
    if settings["sweep"] == "strength":
        # the comparison that matters: every strength model on the same standard test images
        evaluate_run(run_dir, test_data_dir=STANDARD_DATA, results_name="results_standard_test.json")
    # its own test images (for strength 1.0 / height 32 these are the standard ones);
    # written last, because its existence marks the run as finished
    evaluate_run(run_dir)


def summary():
    """One row per sweep point: test CER / exact lines (beam 5) and the training cost."""
    rows = []
    points = [(s["sweep"], s["value"], run_name(n), s["sweep"] == "strength") for n, s in SWEEPS.items()]
    points += [(sweep, value, REFERENCE_RUN, False) for sweep, value in REFERENCE_VALUES.items()]

    for sweep, value, run, standard in points:
        results_file = Path("runs") / run / ("results_standard_test.json" if standard else "results.json")
        log_file = Path("runs") / run / "log.csv"
        if not results_file.exists():
            continue
        results = json.load(open(results_file))
        log = list(csv.DictReader(open(log_file)))
        rows.append({
            "sweep": sweep, "value": value, "run": run,
            "seen_cer": results["test_seen"]["beam5"]["cer"],
            "seen_acc": results["test_seen"]["beam5"]["seq_acc"],
            "heldout_cer": results["test_heldout"]["beam5"]["cer"],
            "heldout_acc": results["test_heldout"]["beam5"]["seq_acc"],
            "best_epoch": results["checkpoint_epoch"],
            "epochs": len(log),
            # median epoch time x epochs: robust to an epoch inflated by the Mac going to sleep
            "train_hours": statistics.median(float(row["epoch_seconds"]) for row in log) * len(log) / 3600,
        })
    rows.sort(key=lambda row: (row["sweep"], row["value"]))

    print("\n==== sweeps: Transformer + CTC, test sets, beam 5 ====")
    print("(strength runs are all tested on the standard strength-1.0 test images)")
    print(f"{'sweep':10}{'value':>8}{'seen CER':>11}{'seen acc':>10}{'held CER':>11}{'held acc':>10}"
          f"{'best ep':>9}{'epochs':>8}{'hours':>7}")
    for row in rows:
        print(f"{row['sweep']:10}{row['value']:>8}{row['seen_cer'] * 100:>10.2f}%{row['seen_acc'] * 100:>9.1f}%"
              f"{row['heldout_cer'] * 100:>10.2f}%{row['heldout_acc'] * 100:>9.1f}%"
              f"{row['best_epoch']:>9}{row['epochs']:>8}{row['train_hours']:>7.2f}")

    if rows:
        Path("analysis").mkdir(exist_ok=True)
        with open("analysis/sweeps.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("\nsaved analysis/sweeps.csv")
        plot(rows)


def plot(rows):
    """Three panels (data size, augmentation strength, image height): test CER, seen vs held-out."""
    panels = [
        ("size", "training lines", "RQ3: amount of training data"),
        ("strength", "augmentation strength (test images: 1.0)", "RQ3: augmentation strength"),
        ("height", "image height (pixels)", "RQ4: image height"),
    ]
    # two series, colour-blind-safe pair; marker shape differs too, so colour is not the only cue
    series = [("seen_cer", "seen fonts", "#2a78d6", "o"), ("heldout_cer", "held-out fonts", "#eb6834", "s")]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (sweep, xlabel, title) in zip(axes, panels):
        points = [row for row in rows if row["sweep"] == sweep]
        xs = [row["value"] for row in points]
        for key, label, colour, marker in series:
            ys = [row[key] * 100 for row in points]
            ax.plot(xs, ys, color=colour, marker=marker, linewidth=2, markersize=7, label=label)
            # direct label on the last point
            ax.annotate(label, (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=8, color="#333333")
        # the reference run (28,990 lines, strength 1.0, height 32) is marked in every panel
        reference = REFERENCE_VALUES[sweep]
        ax.axvline(reference, color="#bbbbbb", linewidth=1, linestyle=":", zorder=0)
        ax.set_yscale("log")                 # CER spans 0.06% to 56%: a log axis keeps all readable
        # plain numbers on the axis (0.1, 1, 10) instead of 10^-1 style, no minor tick labels
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        # a 1-2-5 grid of labelled ticks, so narrow ranges (like panel 1) still get several labels
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(
            [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]))
        if sweep == "size":
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{x:,}" for x in xs])
        else:
            ax.set_xticks(xs)
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("test CER (%), log scale, beam 5")
        ax.grid(axis="y", which="major", color="#e5e5e5", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.25)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Transformer + CTC; dotted line = reference run", fontsize=9, x=0.01, ha="left", color="#555555")
    fig.tight_layout()
    fig.savefig("analysis/sweeps.png", dpi=160)
    fig.savefig("analysis/sweeps.pdf")
    print("saved analysis/sweeps.png")


if __name__ == "__main__":
    names = sys.argv[1:] if len(sys.argv) > 1 else list(SWEEPS)
    for name in names:
        if name != "summary":
            run_one(name)
    summary()
