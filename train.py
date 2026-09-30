import csv
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.build_dataset import BucketBatchSampler, OCRDataset, collate_fn
from data.charset import Vocab
from metrics import compute_metrics
from models.cnn import CNN
from models.attention import AttentionDecoder, make_decoder_io
from models.ctc import CTCHead, ctc_beam_decode, ctc_greedy_decode
from models.encoders import BiLSTMEncoder, TransformerEncoder
from models.ocr_model import OCRModel

# ---- all settings for one run; saved to runs/<run_name>/config.json ----
CONFIG = {
    "run_name": "attn_bilstm_h32_s1.0",
    "encoder": "bilstm",       # "bilstm" or "transformer" (RQ2)
    "head": "attention",       # "ctc" or "attention"
    "attention_score": "additive",   # attention only: "additive" (Bahdanau) or "dot" (Luong)
    "attention_location": True,      # attention only: add the location term (Chorowski et al.)
    "ctc_weight": 0.5,               # attention only: loss = w * CTC + (1 - w) * cross-entropy
                                     # (joint loss, Michael et al. 2019); 0 = pure cross-entropy
    "data_dir": "data/rendered/h32_s1.0",
    "train_split": "train",
    "val_split": "val",
    "max_samples": None,       # RQ3: e.g. 5000 = train on the first 5000 lines only
    "val_max_samples": None,   # normally None (whole val set); only used for the smoke test
    "batch_size": 32,
    "learning_rate": 1e-3,
    "epochs": 30,              # maximum; early stopping usually ends training before this
    "patience": 3,             # stop if val CER has not improved for this many epochs in a row
    "min_epochs": 5,           # never stop before this: CTC first predicts only blanks
                               # (CER stuck at 1.0), which would otherwise trigger early stopping
    "grad_clip": 5.0,
    "grad_accum": 1,           # split every batch into this many parts to save memory
                               # (same weight update; used for height 48, which does not fit)
    "num_workers": 4,
    "seed": 0,
}


def get_device():
    """Use the Apple GPU (MPS) if it is available, otherwise the CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device):
    """Wait until the GPU has really finished. The GPU works asynchronously, so without this
    a timer could stop while the GPU is still busy."""
    if device.type == "mps":
        torch.mps.synchronize()


def get_head(config):
    # runs made before the "head" setting existed were all CTC
    return config.get("head", "ctc")


def build_model(vocab, config):
    cnn = CNN()
    # runs made before the "encoder" setting existed all used the BiLSTM
    if config.get("encoder", "bilstm") == "bilstm":
        encoder = BiLSTMEncoder(input_size=cnn.out_channels)
    else:
        encoder = TransformerEncoder(input_size=cnn.out_channels)
    ctc_head = None
    if get_head(config) == "ctc":
        head = CTCHead(input_size=encoder.output_size, vocab_size=len(vocab))
    else:
        head = AttentionDecoder(encoder_size=encoder.output_size, vocab=vocab,
                                score=config.get("attention_score", "additive"),
                                use_location=config.get("attention_location", True))
        # joint loss: an extra CTC head on the same encoder, used only during training
        # (runs from before this setting existed had no CTC head, hence the default 0)
        if config.get("ctc_weight", 0.0) > 0:
            ctc_head = CTCHead(input_size=encoder.output_size, vocab_size=len(vocab))
    return OCRModel(cnn, encoder, head, ctc_head)


def ctc_loss_fn(model, batch, vocab, device, loss_fn):
    """CTC loss, computed on the CPU because nn.CTCLoss is not implemented on MPS.
    The gradients still flow back to the model on the GPU."""
    encoded, lengths = model.encode(batch["images"].to(device), batch["image_widths"])
    log_probs = model.head(encoded)                   # (batch, steps, vocab_size)
    # CTCLoss wants (steps, batch, tokens); our model gives (batch, steps, tokens)
    log_probs = log_probs.permute(1, 0, 2).cpu()
    return loss_fn(log_probs, batch["labels"], lengths.cpu(), batch["label_lengths"])


def attention_loss_fn(model, batch, vocab, device, loss_fn):
    """Cross-entropy loss with teacher forcing, optionally combined with a CTC loss.

    loss_fn is a dict: {"ce": CrossEntropyLoss, "ctc": CTCLoss, "ctc_weight": w}
    total loss = w * CTC + (1 - w) * cross-entropy   (w = 0: pure cross-entropy)
    """
    decoder_input, decoder_target = make_decoder_io(batch["labels"], batch["label_lengths"], vocab)
    encoded, lengths = model.encode(batch["images"].to(device), batch["image_widths"])
    logits = model.head(encoded, lengths, decoder_input)   # (batch, target_length, vocab_size)

    # CrossEntropyLoss wants all predictions in one long list:
    # (batch, target_length, vocab_size) -> (batch * target_length, vocab_size)
    vocab_size = logits.shape[-1]
    ce_loss = loss_fn["ce"](logits.reshape(-1, vocab_size), decoder_target.reshape(-1).to(device))

    w = loss_fn["ctc_weight"]
    if w == 0:
        return ce_loss

    # the CTC head reads the SAME encoder output, so its gradient tells every image column
    # directly which character it should represent; this is what breaks the cold start
    ctc_log_probs = model.ctc_head(encoded).permute(1, 0, 2).cpu()   # (steps, batch, tokens), on CPU
    ctc_loss = loss_fn["ctc"](ctc_log_probs, batch["labels"], lengths.cpu(), batch["label_lengths"])
    return w * ctc_loss.to(device) + (1 - w) * ce_loss


def split_batch(batch, parts):
    """Cut a batch into `parts` smaller batches (for gradient accumulation).

    Every entry is cut along the batch axis; images keep their padded width,
    labels keep their padded length (both are ignored beyond the real sizes anyway).
    """
    if parts == 1:
        return [batch]
    size = len(batch["texts"])
    part_size = -(-size // parts)                    # ceiling division: 32 lines / 2 -> 16
    pieces = []
    for start in range(0, size, part_size):
        end = start + part_size
        pieces.append({key: value[start:end] for key, value in batch.items()})
    return pieces


def decode_batch(model, batch, vocab, device, head, beam_width=1):
    """Decode a batch with either head. beam_width 1 = greedy, > 1 = beam search.
    Returns a list of strings."""
    encoded, lengths = model.encode(batch["images"].to(device), batch["image_widths"])
    if head == "ctc":
        log_probs = model.head(encoded)
        if beam_width == 1:
            return ctc_greedy_decode(log_probs, lengths, vocab)
        return ctc_beam_decode(log_probs, lengths, vocab, beam_width)

    if beam_width == 1:
        id_lists = model.head.greedy_decode(encoded, lengths)
    else:
        id_lists = model.head.beam_decode(encoded, lengths, beam_width)
    return [vocab.decode(ids) for ids in id_lists]


def evaluate(model, loader, vocab, device, head, beam_width=1):
    """Decode a whole split (greedy by default) and compute CER / WER / sequence accuracy."""
    model.eval()                 # BatchNorm uses its learned averages instead of batch statistics
    predictions = []
    references = []

    with torch.no_grad():        # no gradients needed: faster and uses less memory
        # leave=False: the validation bar disappears when it is done, keeping the output tidy
        for batch in tqdm(loader, desc="  validating", unit="batch", leave=False):
            predictions += decode_batch(model, batch, vocab, device, head, beam_width)
            references += batch["texts"]

    model.train()                # back to training mode for the next epoch
    return compute_metrics(predictions, references), predictions, references


def train(config):
    # ---- setup ----
    run_dir = Path("runs") / config["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    torch.manual_seed(config["seed"])
    device = get_device()
    vocab = Vocab()
    print("device:", device)

    data_dir = Path(config["data_dir"])
    train_set = OCRDataset(data_dir / config["train_split"], config["max_samples"])
    val_set = OCRDataset(data_dir / config["val_split"], config["val_max_samples"])

    # batches of images with similar width -> much less padding -> much faster.
    # train: batch order is shuffled every epoch (seeded); val: fixed order
    train_sampler = BucketBatchSampler(train_set.widths, config["batch_size"],
                                       shuffle=True, seed=config["seed"])
    val_sampler = BucketBatchSampler(val_set.widths, config["batch_size"], shuffle=False)

    workers = config["num_workers"]
    train_loader = DataLoader(train_set, batch_sampler=train_sampler, collate_fn=collate_fn,
                              num_workers=workers, persistent_workers=workers > 0)
    val_loader = DataLoader(val_set, batch_sampler=val_sampler, collate_fn=collate_fn,
                            num_workers=workers, persistent_workers=workers > 0)

    head = get_head(config)
    model = build_model(vocab, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    # the loss depends on the head; everything else in training is the same for both
    if head == "ctc":
        # zero_infinity: a line that cannot be aligned gets loss 0 instead of infinity (safety net)
        loss_fn = nn.CTCLoss(blank=vocab.blank, zero_infinity=True)
        compute_loss = ctc_loss_fn
    else:
        # ignore_index: padding positions in the target do not count towards the loss
        loss_fn = {
            "ce": nn.CrossEntropyLoss(ignore_index=vocab.pad),
            "ctc": nn.CTCLoss(blank=vocab.blank, zero_infinity=True),
            "ctc_weight": config.get("ctc_weight", 0.0),
        }
        compute_loss = attention_loss_fn

    joint = f" + CTC (weight {config.get('ctc_weight', 0.0)})" if model.ctc_head is not None else ""
    print(f"head: {head}{joint} | train: {len(train_set)} lines | val: {len(val_set)} lines | "
          f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    # log.csv: one row per epoch
    log_file = open(run_dir / "log.csv", "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["epoch", "train_loss", "val_cer", "val_wer", "val_seq_acc", "epoch_seconds"])

    best_cer = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0   # for early stopping

    # ---- training ----
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        start = time.time()
        total_loss = 0.0

        # tqdm wraps the loader and draws a progress bar that updates after every batch
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{config['epochs']}", unit="batch")
        for step, batch in enumerate(progress, start=1):
            optimizer.zero_grad()    # clear the gradients from the previous batch

            # gradient accumulation: process the batch in `grad_accum` smaller parts, one after
            # the other, and add up their gradients before one weight update. The update is the
            # same as for the whole batch, but only one part's activations are in memory at a time.
            parts = split_batch(batch, config.get("grad_accum", 1))
            batch_loss = 0.0
            for part in parts:
                # weight each part by its share of the lines, so the sum equals the batch average
                share = len(part["texts"]) / len(batch["texts"])
                loss = compute_loss(model, part, vocab, device, loss_fn) * share
                loss.backward()      # gradients of this part are ADDED to the ones already there
                batch_loss += loss.item()

            # scale gradients down if their total size is above grad_clip (avoids huge updates)
            nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()         # update the weights

            total_loss += batch_loss
            # show the running average loss at the end of the progress bar
            progress.set_postfix(loss=f"{total_loss / step:.4f}")

        synchronize(device)
        train_seconds = time.time() - start
        train_loss = total_loss / len(train_loader)

        # ---- validation ----
        metrics, predictions, references = evaluate(model, val_loader, vocab, device, head)
        epoch_seconds = time.time() - start

        print(f"epoch {epoch}: loss {train_loss:.4f} | val CER {metrics['cer']:.4f} "
              f"WER {metrics['wer']:.4f} acc {metrics['seq_acc']:.4f} | "
              f"{train_seconds:.0f}s train, {epoch_seconds:.0f}s total")
        for prediction, reference in list(zip(predictions, references))[:3]:
            print(f"    true: {reference}")
            print(f"    pred: {prediction}")

        log.writerow([epoch, round(train_loss, 5), round(metrics["cer"], 5),
                      round(metrics["wer"], 5), round(metrics["seq_acc"], 5),
                      round(epoch_seconds, 1)])
        log_file.flush()             # write to disk now, so the log is readable during training

        # keep only the best checkpoint (lowest val CER)
        if metrics["cer"] < best_cer:
            best_cer = metrics["cer"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch},
                       run_dir / "best.pt")
            print(f"    new best val CER {best_cer:.4f}, saved best.pt")
        else:
            epochs_without_improvement += 1
            print(f"    no improvement for {epochs_without_improvement} epoch(s) "
                  f"(best {best_cer:.4f} at epoch {best_epoch})")

        # early stopping: val CER has not improved for `patience` epochs in a row
        # (only allowed after min_epochs)
        if epoch >= config["min_epochs"] and epochs_without_improvement >= config["patience"]:
            print(f"early stopping at epoch {epoch}")
            break

    log_file.close()
    print(f"done. best val CER: {best_cer:.4f} at epoch {best_epoch}")


# run from the ocr/ folder with "python train.py"
if __name__ == "__main__":
    train(CONFIG)
