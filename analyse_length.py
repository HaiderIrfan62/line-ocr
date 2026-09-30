"""RQ4 (part 1): how does accuracy change with line length? No new training needed.

For every trained system, the test predictions are split into groups by the length of the
correct line, and CER / exact-line accuracy / runaway rate are computed per group.

Predictions are saved in runs/<run>/predictions/ the first time, so running this again
(or any later analysis) does not need to decode again.

Run from the ocr/ folder:
    python analyse_length.py
Outputs: a printed table, analysis/line_length.csv and analysis/line_length.png
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                 # draw to a file, no window
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from data.build_dataset import BucketBatchSampler, OCRDataset, collate_fn
from data.charset import Vocab
from metrics import compute_metrics
from train import build_model, evaluate, get_device, get_head

SYSTEMS = [
    # (run folder, label for tables and the plot)
    ("ctc_bilstm_h32_s1.0", "BiLSTM + CTC"),
    ("ctc_transformer_h32_s1.0", "Transformer + CTC"),
    ("attn_bilstm_h32_s1.0", "BiLSTM + attention"),
    ("attn_transformer_h32_s1.0", "Transformer + attention"),
]
SPLITS = ["test_seen", "test_heldout"]
DECODINGS = {"greedy": 1, "beam5": 5}
# length groups (characters of the correct line): [low, high)
GROUPS = [(0, 40), (40, 60), (60, 80), (80, 100), (100, 151)]
# a prediction this much longer than the correct line counts as a runaway (looping) output
RUNAWAY_EXTRA = 10

OUT_DIR = Path("analysis")


def get_predictions(run, split, decoding):
    """Predictions for one system / split / decoding, loaded from disk or decoded once."""
    cache = Path("runs") / run / "predictions" / f"{split}_{decoding}.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8").split("\n")[:-1]

    device = get_device()
    vocab = Vocab()
    checkpoint = torch.load(Path("runs") / run / "best.pt", map_location=device)
    config = checkpoint["config"]
    model = build_model(vocab, config).to(device)
    model.load_state_dict(checkpoint["model"])

    dataset = OCRDataset(Path(config["data_dir"]) / split)
    sampler = BucketBatchSampler(dataset.widths, config["batch_size"], shuffle=False)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)
    print(f"decoding {run} / {split} / {decoding} ...")
    _, predictions, references = evaluate(model, loader, vocab, device, get_head(config),
                                          DECODINGS[decoding])

    # the sampler sorts lines by width; put predictions back in the original line order
    order = [index for batch in sampler for index in batch]
    in_order = [None] * len(dataset)
    for index, prediction in zip(order, predictions):
        in_order[index] = prediction

    cache.parent.mkdir(exist_ok=True)
    cache.write_text("\n".join(in_order) + "\n", encoding="utf-8")
    return in_order


def references_for(split):
    labels = Path("data/rendered/h32_s1.0") / split / "labels.txt"
    return labels.read_text(encoding="utf-8").splitlines()


def group_metrics(predictions, references):
    """CER, exact-line accuracy and runaway rate for every length group."""
    results = []
    for low, high in GROUPS:
        pairs = [(p, r) for p, r in zip(predictions, references) if low <= len(r) < high]
        preds = [p for p, _ in pairs]
        refs = [r for _, r in pairs]
        metrics = compute_metrics(preds, refs)
        runaways = sum(len(p) > len(r) + RUNAWAY_EXTRA for p, r in pairs)
        results.append({"group": f"{low}-{high - 1}", "lines": len(pairs), "cer": metrics["cer"],
                         "seq_acc": metrics["seq_acc"], "runaways": runaways})
    return results


def plot(rows):
    """Two panels (seen / held-out fonts), CER per length group, one line per system (beam5)."""
    # categorical palette (validated for colour-blind separation); line style = head,
    # marker = encoder, so the systems are also distinguishable without colour
    style = {
        "BiLSTM + CTC": ("#2a78d6", "-", "o"),
        "Transformer + CTC": ("#eb6834", "-", "s"),
        "BiLSTM + attention": ("#1baf7a", "--", "o"),
        "Transformer + attention": ("#eda100", "--", "s"),
    }
    groups = [f"{low}-{high - 1}" for low, high in GROUPS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, split, title in zip(axes, SPLITS, ["Seen fonts", "Held-out fonts"]):
        ends = []
        for label, (colour, line, marker) in style.items():
            values = [r["cer"] * 100 for r in rows
                      if r["system"] == label and r["split"] == split and r["decoding"] == "beam5"]
            ax.plot(groups, values, color=colour, linestyle=line, marker=marker, linewidth=2,
                    markersize=7, label=label)
            ends.append([values[-1], label])

        # direct labels at the last point, so identity does not rely on colour alone;
        # labels that would overlap are pushed apart vertically
        top = max(end for end, _ in ends)
        min_gap = top * 0.07
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + min_gap)
        for y, label in ends:
            ax.annotate(label, (len(groups) - 1, y), xytext=(10, 0), textcoords="offset points",
                        va="center", fontsize=8, color="#333333")
        ax.set_title(title, fontsize=12, loc="left")
        ax.set_xlabel("length of the correct line (characters)")
        ax.set_ylabel("CER (%), beam search width 5")
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(-0.3, len(groups) + 1.6)       # room for the direct labels
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "line_length.png", dpi=160)
    fig.savefig(OUT_DIR / "line_length.pdf")
    print(f"saved {OUT_DIR / 'line_length.png'}")


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    rows = []
    for run, label in SYSTEMS:
        for split in SPLITS:
            references = references_for(split)
            for decoding in DECODINGS:
                predictions = get_predictions(run, split, decoding)
                for result in group_metrics(predictions, references):
                    rows.append({"system": label, "split": split, "decoding": decoding, **result})

    # printed table: CER per length group (beam5), plus the runaway count
    for split in SPLITS:
        print(f"\n{split}: CER by line length (beam 5)   [runaway outputs in brackets]")
        header = "".join(f"{f'{low}-{high - 1}':>15}" for low, high in GROUPS)
        print(f"{'':26}{header}")
        lines_per_group = [r["lines"] for r in rows if r["split"] == split][:len(GROUPS)]
        print(f"{'lines':26}" + "".join(f"{n:>15}" for n in lines_per_group))
        for _, label in SYSTEMS:
            cells = [r for r in rows if r["system"] == label and r["split"] == split
                     and r["decoding"] == "beam5"]
            print(f"{label:26}" + "".join(f"{c['cer'] * 100:>10.2f}% [{c['runaways']}]" for c in cells))

    with open(OUT_DIR / "line_length.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsaved {OUT_DIR / 'line_length.csv'}")
    plot(rows)
