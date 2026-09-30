# Line-Level Text Recognition with Convolutional and Sequence Models

Course project (Hauptseminar, Department of Computational Linguistics, Heidelberg University),
topic *"Text OCR (plain PyTorch)"*. The report is submitted separately.

The task is to read one line of text from a grayscale image. Four systems are compared, changing
one component at a time on a shared CNN feature extractor:

| | CTC head | Attention decoder |
|---|---|---|
| **BiLSTM encoder** | BiLSTM + CTC | BiLSTM + attention |
| **Transformer encoder** | Transformer + CTC | Transformer + attention |

Training data is synthetic: lines from the English side of [Multi30k](https://github.com/multi30k/dataset)
are rendered in 17 typefaces with augmentation. Each test line is rendered in the training fonts
(*seen*) and in a held-out font family (STIXGeneral).

## Main results

Test CER in percent, beam search width 5 (full tables in the report):

| System | Seen fonts | Held-out fonts | Training time |
|---|---|---|---|
| BiLSTM + CTC | **0.10** | 2.31 | 1.34 h |
| Transformer + CTC | 0.15 | 2.46 | **0.55 h** |
| BiLSTM + attention | 0.12 | 2.39 | 3.04 h |
| Transformer + attention | 0.31 | **2.01** | 1.59 h |

- The attention decoder only learned to read with a **joint CTC + cross-entropy loss**; trained
  with cross-entropy alone it wrote the same memorised caption for every image
  (experiments A–J in `attention_experiments.py`).
- CTC merges double letters (e.g. *soccer* → *socer*); the attention decoder does not.
- Training on images **more augmented than the test images** (strength 1.5) reduced held-out CER
  of Transformer + CTC from 2.46% to **0.93%**, the largest effect in the project.

Full tables and discussion are in the report.

## Repository layout

```
data/
  multi30k/           raw Multi30k English text (train / val / test_2016_flickr)
  preprocessing.py    cleaning: remove rare characters, cut lines to 150 chars, deduplicate
  processed/          cleaned text + vocab.json (73 tokens)
  charset.py          vocabulary: Vocab.encode / decode
  render.py           line rendering, fonts, augmentation (make_sample)
  build_dataset.py    renders all splits to PNG, OCRDataset, width-bucketed batching
models/
  cnn.py              shared CNN feature extractor (height -> 1, width / 4)
  encoders.py         BiLSTMEncoder, TransformerEncoder
  ctc.py              CTC head, greedy and prefix beam search decoding
  attention.py        attention decoder (additive / dot, location-aware), beam search
  ocr_model.py        CNN + encoder + head (+ auxiliary CTC head for the joint loss)
train.py              training loop, CONFIG with all settings
evaluate.py           test-set evaluation (greedy + beam 5, seen + held-out fonts)
metrics.py            CER, WER, exact-line accuracy
run_rq2.py            trains and evaluates the two Transformer systems
attention_experiments.py   small experiments A–J on the attention cold start
run_sweeps.py         RQ3/RQ4 sweeps (data size, augmentation strength, image height)
analyse_length.py     accuracy by line length (no training needed)
runs/<run>/           config.json, log.csv, results.json, predictions/ for every run
analysis/             sweep and line-length tables and plots
docs/                 illustrated walkthrough of one line through the models (HTML)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Every script is run from the repository root. Training uses the Apple MPS backend if available,
otherwise the CPU. The CTC loss has no MPS kernel and is computed on the CPU.

## Reproducing the results

```bash
# 1. data (cleaned text is included; this re-creates it and the vocabulary)
python data/preprocessing.py
python data/charset.py

# 2. render the images (height 32, augmentation strength 1.0; ~1 min, ~540 MB)
python -m data.build_dataset

# 3. BiLSTM systems: set "head" in CONFIG (train.py) to "ctc" or "attention"
#    and "run_name" to ctc_bilstm_h32_s1.0 / attn_bilstm_h32_s1.0, then
python train.py
python evaluate.py runs/ctc_bilstm_h32_s1.0
python evaluate.py runs/attn_bilstm_h32_s1.0

# 4. Transformer systems (RQ2), trained and evaluated in one go
python run_rq2.py

# 5. attention cold-start experiments A-J (2,000 lines each)
python attention_experiments.py A B C D E F G H I J

# 6. sweeps with Transformer + CTC (renders the extra image sets itself)
python run_sweeps.py

# 7. line-length analysis (writes analysis/line_length.*)
python analyse_length.py
```

All runs use seed 0; the settings of every run are stored in `runs/<run>/config.json`.

The trained checkpoints of the four main systems are included (`runs/*_h32_s1.0/best.pt`,
~20 MB each), so the test results can be reproduced without training: render the images
(step 2), then run e.g. `python evaluate.py runs/ctc_bilstm_h32_s1.0`. Checkpoints of the
sweeps and small experiments are not included.
