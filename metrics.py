def edit_distance(a, b):
    """Smallest number of substitutions, insertions and deletions that turn a into b.

    Works on any two sequences: strings (character level) or lists of words (word level).
    """
    # table[i][j] = distance between the first i items of a and the first j items of b.
    # we only ever need the previous row, so we keep two rows instead of the whole table
    previous = list(range(len(b) + 1))   # distance from an empty a: insert all j items of b

    for i in range(1, len(a) + 1):
        current = [i]                    # distance to an empty b: delete all i items of a
        for j in range(1, len(b) + 1):
            substitution_cost = 0 if a[i - 1] == b[j - 1] else 1
            current.append(min(
                previous[j] + 1,                        # deletion     (from above)
                current[j - 1] + 1,                     # insertion    (from the left)
                previous[j - 1] + substitution_cost,    # substitution (diagonal, free if equal)
            ))
        previous = current

    return previous[-1]                  # bottom-right cell


def compute_metrics(predictions, references):
    """CER, WER and sequence accuracy over a whole set of lines.

    CER and WER are summed over all lines, then divided once (not averaged per line),
    so long lines count for more than short ones.
    """
    char_edits = 0
    char_total = 0
    word_edits = 0
    word_total = 0
    exact_matches = 0

    for prediction, reference in zip(predictions, references):
        char_edits += edit_distance(prediction, reference)
        char_total += len(reference)

        # .split() splits on spaces; punctuation stays attached to its word ("dog." != "dog")
        word_edits += edit_distance(prediction.split(), reference.split())
        word_total += len(reference.split())

        if prediction == reference:
            exact_matches += 1

    return {
        "cer": char_edits / char_total,
        "wer": word_edits / word_total,
        "seq_acc": exact_matches / len(references),
    }


# this block only runs with "python metrics.py" (from the ocr/ folder): checks with known answers
if __name__ == "__main__":
    tests = [
        ("a dug", "a dog", 1),     # 1 substitution
        ("a do", "a dog", 1),      # 1 insertion
        ("aa dog", "a dog", 1),    # 1 deletion
        ("b dig!", "a dog", 3),    # 2 substitutions + 1 deletion
        ("", "dog", 3),            # empty prediction: insert everything
        ("dog", "dog", 0),
    ]
    for prediction, reference, expected in tests:
        result = edit_distance(prediction, reference)
        print(f"{prediction!r:9} vs {reference!r:8} -> {result} (expected {expected})",
              "OK" if result == expected else "FAIL")

    metrics = compute_metrics(["abc"], ["abd"])
    print("CER('abc', 'abd') =", round(metrics["cer"], 4), "(expected 0.3333)")

    metrics = compute_metrics(
        ["a man is walking", "two dogs run."],
        ["a man is walking", "two dogs run"],
    )
    print("two lines:", metrics)
    # expected: CER = 1 edit / 28 chars, WER = 1 word / 7 words, seq_acc = 1 of 2 lines
