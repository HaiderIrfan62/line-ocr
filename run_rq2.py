"""RQ2: train and evaluate both Transformer systems, one after the other.

Each run uses exactly the settings in CONFIG (train.py) and changes only what is listed
below, so the only difference to the BiLSTM baselines is the encoder.

Run from the ocr/ folder:
    python run_rq2.py           # both runs (~3.5 hours)
    python run_rq2.py attn      # only the attention run
"""
import sys

from evaluate import evaluate_run
from train import CONFIG, train

RUNS = {
    # same as ctc_bilstm_h32_s1.0, with the Transformer encoder
    "ctc": {"run_name": "ctc_transformer_h32_s1.0", "encoder": "transformer",
            "head": "ctc", "ctc_weight": 0.0},
    # same as attn_bilstm_h32_s1.0 (joint loss 0.5), with the Transformer encoder
    "attn": {"run_name": "attn_transformer_h32_s1.0", "encoder": "transformer",
             "head": "attention", "ctc_weight": 0.5},
}

if __name__ == "__main__":
    names = sys.argv[1:] if len(sys.argv) > 1 else list(RUNS)

    for name in names:
        settings = RUNS[name]
        print(f"\n######## training {settings['run_name']} ########")
        train({**CONFIG, **settings})

        print(f"\n######## evaluating {settings['run_name']} ########")
        evaluate_run(f"runs/{settings['run_name']}")
