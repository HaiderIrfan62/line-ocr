"""Small, quick experiments on a fixed small setup, so the results are directly comparable:
2000 training lines, 15 epochs, no early stopping, val CER on 300 unseen val lines.

A-G: getting the attention decoder past its cold start (BiLSTM encoder)
H-J: quick check of the Transformer encoder before the full RQ2 runs

Run from the ocr/ folder:
    python attention_experiments.py          # runs the newest experiments: H, I, J
    python attention_experiments.py E F      # runs only the experiments you name
"""
import csv
import sys
from pathlib import Path

from train import CONFIG, train

# settings shared by all experiments (everything else comes from CONFIG in train.py)
SMALL_SETUP = {
    "max_samples": 2000,
    "val_max_samples": 300,
    "epochs": 15,
    "patience": 1000,        # effectively no early stopping: we want to see all 15 epochs
    "min_epochs": 1000,
    "num_workers": 2,
}

# what each experiment changes; A and B were already run with this setup
EXPERIMENTS = {
    "A": {"run_name": "smoke_attention_2k", "attention_score": "additive",
          "attention_location": False, "learning_rate": 1e-3},
    "B": {"run_name": "smoke_attention_2k_location", "attention_score": "additive",
          "attention_location": True, "learning_rate": 1e-3},
    "C": {"run_name": "exp_C_additive_location_lr3e-4", "attention_score": "additive",
          "attention_location": True, "learning_rate": 3e-4},
    "D": {"run_name": "exp_D_additive_location_lr3e-3", "attention_score": "additive",
          "attention_location": True, "learning_rate": 3e-3},
    "E": {"run_name": "exp_E_dot_location_lr1e-3", "attention_score": "dot",
          "attention_location": True, "learning_rate": 1e-3},
    "F": {"run_name": "exp_F_dot_nolocation_lr1e-3", "attention_score": "dot",
          "attention_location": False, "learning_rate": 1e-3},
    # joint CTC + attention loss (Michael et al. 2019): same as B, plus 0.5 * CTC during training
    "G": {"run_name": "exp_G_additive_location_ctc0.5", "attention_score": "additive",
          "attention_location": True, "learning_rate": 1e-3, "ctc_weight": 0.5},
}
# A-F were pure cross-entropy; write that down explicitly, since the default is now 0.5
for name in "ABCDEF":
    EXPERIMENTS[name]["ctc_weight"] = 0.0
# A-G all used the BiLSTM encoder and the attention head
for name in "ABCDEFG":
    EXPERIMENTS[name]["encoder"] = "bilstm"
    EXPERIMENTS[name]["head"] = "attention"

# Transformer check: H is the BiLSTM reference for I on the same small setup
EXPERIMENTS["H"] = {"run_name": "exp_H_bilstm_ctc", "encoder": "bilstm", "head": "ctc",
                    "attention_score": "-", "attention_location": "-",
                    "learning_rate": 1e-3, "ctc_weight": 0.0}
EXPERIMENTS["I"] = {"run_name": "exp_I_transformer_ctc", "encoder": "transformer", "head": "ctc",
                    "attention_score": "-", "attention_location": "-",
                    "learning_rate": 1e-3, "ctc_weight": 0.0}
EXPERIMENTS["J"] = {"run_name": "exp_J_transformer_attention_ctc0.5", "encoder": "transformer",
                    "head": "attention", "attention_score": "additive", "attention_location": True,
                    "learning_rate": 1e-3, "ctc_weight": 0.5}

NEW_EXPERIMENTS = ["H", "I", "J"]


def read_log(run_name):
    """Best and last val CER from a run's log.csv (None if the run does not exist yet)."""
    log_path = Path("runs") / run_name / "log.csv"
    if not log_path.exists():
        return None
    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    best = min(rows, key=lambda row: float(row["val_cer"]))
    return {
        "epochs": len(rows),
        "best_cer": float(best["val_cer"]),
        "best_epoch": int(best["epoch"]),
        "last_cer": float(rows[-1]["val_cer"]),
        "best_acc": max(float(row["val_seq_acc"]) for row in rows),
        "seconds_per_epoch": sum(float(row["epoch_seconds"]) for row in rows) / len(rows),
    }


def print_summary():
    print("\n==== summary (2000 train lines, val CER on 300 unseen lines) ====")
    print(f"{'':3}{'encoder':12}{'head':10}{'score':9}{'location':10}{'lr':8}{'ctc w':7}{'epochs':>7}"
          f"{'best CER':>10}{'(epoch)':>8}{'last CER':>10}{'best acc':>10}{'s/epoch':>9}")
    for name, settings in EXPERIMENTS.items():
        result = read_log(settings["run_name"])
        row = (f"{name:3}{settings['encoder']:12}{settings['head']:10}"
               f"{settings['attention_score']:9}{str(settings['attention_location']):10}"
               f"{settings['learning_rate']:<8g}{settings['ctc_weight']:<7g}")
        if result is None:
            print(row + "   (not run yet)")
        else:
            print(row + f"{result['epochs']:>7}{result['best_cer']:>10.4f}{result['best_epoch']:>8}"
                        f"{result['last_cer']:>10.4f}{result['best_acc']:>10.3f}"
                        f"{result['seconds_per_epoch']:>9.0f}")
    print("\nA-G success = predictions show the real line text and val CER clearly below 0.5")
    print("H-J: compare I with H (same head, only the encoder differs); J should read like G")


if __name__ == "__main__":
    # which experiments to run: the ones named on the command line, or all new ones
    names = sys.argv[1:] if len(sys.argv) > 1 else NEW_EXPERIMENTS

    for name in names:
        settings = EXPERIMENTS[name]
        print(f"\n######## experiment {name}: {settings} ########")
        train(dict(CONFIG, **SMALL_SETUP, **settings))

    print_summary()
