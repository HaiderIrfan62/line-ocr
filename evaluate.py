import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.build_dataset import BucketBatchSampler, OCRDataset, collate_fn
from data.charset import Vocab
from train import build_model, evaluate, get_device, get_head

# the two test conditions from the proposal
TEST_SPLITS = ["test_seen", "test_heldout"]
# every system is decoded the same way: greedy, and beam search at one fixed width
BEAM_WIDTH = 5
NUM_EXAMPLES_TO_SHOW = 5


def evaluate_run(run_dir, test_data_dir=None, results_name="results.json"):
    """Evaluate a run's best checkpoint on both test sets, greedy and beam search.

    test_data_dir: where the test images come from. Default (None): the data_dir the model
                   was trained on. The augmentation sweep passes the standard strength-1.0
                   images here, so every model is tested on the same images.
    results_name:  file name for the results inside the run folder
    """
    run_dir = Path(run_dir)
    device = get_device()
    vocab = Vocab()

    # best.pt holds the weights and the config the model was trained with
    checkpoint = torch.load(run_dir / "best.pt", map_location=device)
    config = checkpoint["config"]

    # rebuild exactly the same model and load the trained weights into it
    head = get_head(config)
    model = build_model(vocab, config).to(device)
    model.load_state_dict(checkpoint["model"])
    print(f"loaded {run_dir / 'best.pt'} (head: {head}, epoch {checkpoint['epoch']}) on {device}")

    # by default the test images come from the same data_dir the model was trained on
    # (so a model trained at height 32 is tested on height-32 images)
    test_data_dir = Path(test_data_dir or config["data_dir"])
    print(f"test images from {test_data_dir}")

    results = {"checkpoint_epoch": checkpoint["epoch"], "beam_width": BEAM_WIDTH,
               "test_data_dir": str(test_data_dir)}
    decodings = {"greedy": 1, f"beam{BEAM_WIDTH}": BEAM_WIDTH}

    for split in TEST_SPLITS:
        dataset = OCRDataset(test_data_dir / split)
        sampler = BucketBatchSampler(dataset.widths, config["batch_size"], shuffle=False)
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)

        results[split] = {}
        all_predictions = {}
        print(f"\n{split} ({len(dataset)} lines):")
        for name, width in decodings.items():
            start = time.time()
            # same evaluate() as validation during training, with the chosen beam width
            metrics, predictions, references = evaluate(model, loader, vocab, device, head, width)
            metrics["seconds"] = round(time.time() - start, 1)
            results[split][name] = metrics
            all_predictions[name] = predictions
            print(f"  {name:7}: CER {metrics['cer']:.4f} | WER {metrics['wer']:.4f} | "
                  f"acc {metrics['seq_acc']:.4f} | {metrics['seconds']}s")

        # lines where beam search changed the output: did it fix or break them?
        beam_name = f"beam{BEAM_WIDTH}"
        fixed, broken, changed = 0, 0, []
        for greedy, beam, reference in zip(all_predictions["greedy"], all_predictions[beam_name],
                                           references):
            if greedy != beam:
                changed.append((reference, greedy, beam))
                fixed += (beam == reference)
                broken += (greedy == reference)
        print(f"  beam search changed {len(changed)} lines: {fixed} now correct, "
              f"{broken} were correct before")
        for reference, greedy, beam in changed[:NUM_EXAMPLES_TO_SHOW]:
            print(f"    true:   {reference}")
            print(f"    greedy: {greedy}")
            print(f"    beam:   {beam}")

    with open(run_dir / results_name, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {run_dir / results_name}")


# run from the ocr/ folder, e.g. "python evaluate.py runs/ctc_bilstm_h32_s1.0"
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python evaluate.py runs/<run_name>")
        sys.exit(1)
    evaluate_run(sys.argv[1])
