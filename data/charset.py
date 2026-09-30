import json
from pathlib import Path

# folder with the cleaned text files made by preprocessing.py
DATA_DIR = Path(__file__).parent / "processed"
VOCAB_PATH = DATA_DIR / "vocab.json"

# special tokens always come first so their IDs never change:
#   <blank> = 0  -> needed by CTC loss (PyTorch expects blank at index 0)
#   <pad>   = 1  -> fills shorter labels in a batch (attention decoder)
#   <sos>   = 2  -> "start of sequence", first input to the attention decoder
#   <eos>   = 3  -> "end of sequence", tells the attention decoder to stop
#   <unk>   = 4  -> "unknown", used for any character that is not in the vocab
SPECIAL_TOKENS = ["<blank>", "<pad>", "<sos>", "<eos>", "<unk>"]

FILES = ["train", "val", "test_2016_flickr"]


def load_lines(name):
    with open(DATA_DIR / f"{name}.en", "r", encoding="utf-8") as f:
        return f.read().splitlines()


def build_vocab():
    """Build the vocabulary from train and save it to vocab.json. Run once."""
    # collect every unique character that appears in train
    # (a set automatically ignores characters we've already added)
    chars = set()
    for line in load_lines("train"):
        for character in line:
            chars.add(character)

    # sort so the character order (and therefore the IDs) is the same on every run
    chars = sorted(chars)

    # safety check: val and test must not contain characters the model never saw in train
    for name in ["val", "test_2016_flickr"]:
        for line in load_lines(name):
            for character in line:
                assert character in chars, f"{character!r} in {name} is not in train"

    # position in the list = ID, e.g. idx2char[4] -> ' '
    idx2char = SPECIAL_TOKENS + chars

    # we only need to save the list; the dict is rebuilt from it when loading
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(idx2char, f, ensure_ascii=False, indent=2)

    print(f"saved {len(idx2char)} tokens to {VOCAB_PATH}")


class Vocab:
    """Loads vocab.json and converts between text and lists of IDs."""

    def __init__(self, path=VOCAB_PATH):
        with open(path, "r", encoding="utf-8") as f:
            # idx2char: ID -> character, e.g. idx2char[20] -> 'A'
            self.idx2char = json.load(f)

        # char2idx: character -> ID, e.g. char2idx['A'] -> 20
        self.char2idx = {}
        for idx, character in enumerate(self.idx2char):
            self.char2idx[character] = idx

        # special token IDs by name, so other files don't need to remember the numbers
        self.blank = self.char2idx["<blank>"]
        self.pad = self.char2idx["<pad>"]
        self.sos = self.char2idx["<sos>"]
        self.eos = self.char2idx["<eos>"]
        self.unk = self.char2idx["<unk>"]

    def __len__(self):
        # lets us write len(vocab) -> 73
        return len(self.idx2char)

    def encode(self, text):
        """'A dog' -> [21, 5, 50, 61, 53]"""
        ids = []
        for character in text:
            # .get() returns the <unk> ID instead of crashing if the character is not in the vocab
            ids.append(self.char2idx.get(character, self.unk))
        return ids

    def decode(self, ids):
        """[21, 5, 50, 61, 53] -> 'A dog' (special tokens are skipped, <unk> becomes '?')"""
        text = ""
        for idx in ids:
            if idx in [self.blank, self.pad, self.sos, self.eos]:
                continue
            if idx == self.unk:
                text += "?"
                continue
            text += self.idx2char[idx]
        return text


# this block only runs with "python charset.py", not when another file imports Vocab
if __name__ == "__main__":
    build_vocab()

    # round-trip test: encoding then decoding every line must give back the same line
    vocab = Vocab()
    for name in FILES:
        for line in load_lines(name):
            assert vocab.decode(vocab.encode(line)) == line, f"round trip failed: {line!r}"
    print("round trip OK on all splits")

    print("vocab size:", len(vocab))
    print("encode('A dog'):", vocab.encode("A dog"))
    print("decode(...):", vocab.decode(vocab.encode("A dog")))
