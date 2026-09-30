import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# imports start with "data." because every script is run from the ocr/ folder,
# e.g. "python -m data.build_dataset" or "python train.py"
from data.charset import Vocab
from data.render import HELDOUT_FONTS, TRAIN_FONTS, make_sample

TEXT_DIR = Path(__file__).parent / "processed"   # cleaned text files
RENDER_DIR = Path(__file__).parent / "rendered"  # rendered images go here

# each split gets its own seed range, so e.g. train image 5 and test image 5
# do not end up with the same font, size and augmentation
SPLIT_SEEDS = {"train": 0, "val": 1_000_000, "test_2016_flickr": 2_000_000}

# what gets rendered for one configuration:
#   output folder name -> (text file, fonts)
# the two test folders use the same sentences and the same seeds, so each image gets the same
# size, padding and augmentation in both; only the font differs (seen vs held-out)
OUTPUTS = {
    "train": ("train", TRAIN_FONTS),
    "val": ("val", TRAIN_FONTS),
    "test_seen": ("test_2016_flickr", TRAIN_FONTS),
    "test_heldout": ("test_2016_flickr", HELDOUT_FONTS),
}

# ID of the <pad> token, looked up once here instead of for every batch
PAD_ID = Vocab().pad


def config_name(height, strength):
    """Folder name for one rendering configuration, e.g. 'h32_s1.0'."""
    return f"h{height}_s{strength}"


def build_dataset(height=32, strength=1.0, base_seed=0):
    """Render every image of every split once and save them as PNG files.

    Result, e.g. for height 32 and strength 1.0:
        rendered/h32_s1.0/train/00000.png, 00001.png, ...
        rendered/h32_s1.0/train/labels.txt   (line i = text of image i)
        ... same for val, test_seen, test_heldout
    """
    config_dir = RENDER_DIR / config_name(height, strength)

    for out_name, (text_name, fonts) in OUTPUTS.items():
        out_dir = config_dir / out_name

        # skip folders that were already fully rendered, so re-running is cheap
        if (out_dir / "labels.txt").exists():
            print(f"{out_dir} already exists, skipping")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(TEXT_DIR / f"{text_name}.en", "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        start = time.time()
        seed = base_seed + SPLIT_SEEDS[text_name]
        for index, text in enumerate(lines):
            image = make_sample(text, index, fonts, height, strength, seed)
            image.save(out_dir / f"{index:05d}.png")  # :05d -> 5 digits with leading zeros

        # labels.txt is written last: if rendering crashes halfway, the folder
        # has no labels.txt and will be rendered again on the next run
        with open(out_dir / "labels.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        print(f"{out_dir}: {len(lines)} images in {time.time() - start:.0f}s")


class OCRDataset(Dataset):
    """Loads pre-rendered images from disk. Returns (image, label, text) for one line.

    folder      -> e.g. RENDER_DIR / "h32_s1.0" / "train"
    max_samples -> only use the first N images, for the training-data-size sweep (RQ3)
    """

    def __init__(self, folder, max_samples=None):
        self.folder = Path(folder)
        with open(self.folder / "labels.txt", "r", encoding="utf-8") as f:
            self.lines = f.read().splitlines()

        if max_samples is not None:
            self.lines = self.lines[:max_samples]

        self.vocab = Vocab()

        # width of every image, needed by BucketBatchSampler to group similar widths.
        # Image.open only reads the PNG header here (not the pixels), so this is fast
        self.widths = []
        for index in range(len(self.lines)):
            with Image.open(self.folder / f"{index:05d}.png") as image:
                self.widths.append(image.width)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        text = self.lines[index]
        image = Image.open(self.folder / f"{index:05d}.png")

        # PIL image -> float tensor with values 0-1, shape (1, height, width)
        # (the 1 is the channel dimension: grayscale has one channel, color would have 3)
        image = torch.from_numpy(np.array(image)).float() / 255
        image = image.unsqueeze(0)

        label = torch.tensor(self.vocab.encode(text), dtype=torch.long)
        return image, label, text


class BucketBatchSampler:
    """Groups images of similar width into the same batch, so batches need little padding.

    1. sort all images by width
    2. cut the sorted list into batches of batch_size
    3. (if shuffle) shuffle the order of the batches, differently every epoch

    Each batch is a list of dataset indices; the DataLoader loads exactly those images.
    """

    def __init__(self, widths, batch_size, shuffle, seed=0):
        # indices sorted from narrowest to widest image
        sorted_indices = sorted(range(len(widths)), key=lambda i: widths[i])

        # cut into batches: [0:32], [32:64], ... (the last batch may be smaller)
        self.batches = []
        for start in range(0, len(sorted_indices), batch_size):
            self.batches.append(sorted_indices[start:start + batch_size])

        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        batches = list(self.batches)
        if self.shuffle:
            # seed + epoch: a different batch order every epoch, but the same on every run
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(batches)
            self.epoch += 1
        return iter(batches)

    def __len__(self):
        return len(self.batches)


def collate_fn(batch):
    """Combine several samples into one batch.

    Images have different widths and labels have different lengths, so both are padded
    to the longest one in the batch. The real sizes are returned too, so the model and
    the loss know which part is padding.
    """
    images, labels, texts = zip(*batch)  # turns a list of tuples into three tuples

    height = images[0].shape[1]
    max_width = max(image.shape[2] for image in images)
    max_length = max(len(label) for label in labels)

    padded_images = torch.zeros(len(batch), 1, height, max_width)
    padded_labels = torch.full((len(batch), max_length), PAD_ID, dtype=torch.long)

    for i, (image, label) in enumerate(zip(images, labels)):
        width = image.shape[2]

        # fill the whole row with the image's background gray, then copy the image on the left.
        # (plain white padding would leave a visible edge next to a gray background)
        background = image.median()
        padded_images[i] = background
        padded_images[i, :, :, :width] = image

        padded_labels[i, :len(label)] = label

    image_widths = torch.tensor([image.shape[2] for image in images], dtype=torch.long)
    label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)

    return {
        "images": padded_images,         # (batch, 1, height, max_width)
        "image_widths": image_widths,    # (batch,) real width of each image
        "labels": padded_labels,         # (batch, max_length) character IDs, padded with <pad>
        "label_lengths": label_lengths,  # (batch,) real length of each label
        "texts": list(texts),            # original strings, handy for computing CER/WER later
    }


# this block only runs with "python build_dataset.py":
# renders the default configuration to disk, then checks that everything fits together
if __name__ == "__main__":
    HEIGHT = 32
    STRENGTH = 1.0
    build_dataset(HEIGHT, STRENGTH)

    config_dir = RENDER_DIR / config_name(HEIGHT, STRENGTH)
    train_set = OCRDataset(config_dir / "train")
    test_seen = OCRDataset(config_dir / "test_seen")
    test_heldout = OCRDataset(config_dir / "test_heldout")
    print("train:", len(train_set), "| test (seen fonts):", len(test_seen),
          "| test (held-out fonts):", len(test_heldout))

    image, label, text = train_set[0]
    print("one sample:", tuple(image.shape), "| label length:", len(label), "|", text)

    loader = DataLoader(train_set, batch_size=4, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))
    for key, value in batch.items():
        if key == "texts":
            print(f"{key:14}", value)
        else:
            print(f"{key:14}", tuple(value.shape))

    # the image loaded from disk must be exactly what the renderer produces for that index
    # (PNG is lossless, so the pixels should match 1:1)
    fresh = make_sample(train_set.lines[7], 7, TRAIN_FONTS, HEIGHT, STRENGTH, SPLIT_SEEDS["train"])
    fresh = torch.from_numpy(np.array(fresh)).float().unsqueeze(0) / 255
    assert torch.equal(train_set[7][0], fresh), "saved image differs from a fresh render"
    print("disk matches renderer: OK")
