from pathlib import Path

DATA_DIR = Path(__file__).parent / "multi30k"
OUT_DIR = Path(__file__).parent / "processed"
OUT_DIR.mkdir(exist_ok=True)

splits = {}
files = ["train", "val", "test_2016_flickr"]
for name in files:
    with open(DATA_DIR / f"{name}.en", "r", encoding="utf-8") as f:
        splits[name] = f.read().splitlines()

cleaned={}
for name in files:
    cleaned[name] = []
    for line in splits[name]:
        line = line.strip()

        nline = ""
        for character in line:
            if character in [';', '!', '(', ')', '&', '#', '?', ':', '`', '%', '$', '=']:
                continue
            nline += character

        nline = " ".join(nline.split())

        if len(nline) > 150:
            cut = nline.rfind(" ", 0, 151)
            nline = nline[:cut].strip()

        cleaned[name].append(nline)

    cleaned[name] = list(dict.fromkeys(cleaned[name]))

    print(name, len(splits[name]), "->", len(cleaned[name]),
          "| max len:", max(len(l) for l in cleaned[name]))

val_set = set(cleaned["val"])
test_set = set(cleaned["test_2016_flickr"])

print("train/val overlap:", len(set(cleaned["train"]) & val_set))
print("train/test overlap:", len(set(cleaned["train"]) & test_set))
print("val/test overlap:", len(val_set & test_set))

cleaned["train"] = [l for l in cleaned["train"] if l not in val_set and l not in test_set]
print("train after removing overlap:", len(cleaned["train"]))


for name in files:
    with open(OUT_DIR / f"{name}.en", "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned[name]) + "\n")