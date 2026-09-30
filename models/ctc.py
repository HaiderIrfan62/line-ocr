import math

from torch import nn


class CTCHead(nn.Module):
    """Turns every encoder step into log probabilities over the vocabulary.

    input:  (batch, steps, input_size)   encoder output
    output: (batch, steps, vocab_size)   log probabilities, one distribution per step
    """

    def __init__(self, input_size, vocab_size):
        super().__init__()
        # one linear layer, applied to every step on its own:
        # 512 encoder numbers -> 73 scores (one per token, <blank> included)
        self.linear = nn.Linear(input_size, vocab_size)

    def forward(self, encoded):
        logits = self.linear(encoded)
        # log_softmax over the last axis (the tokens): scores -> log probabilities.
        # nn.CTCLoss expects log probabilities, and logs avoid multiplying tiny numbers
        return logits.log_softmax(dim=-1)


def ctc_greedy_decode(log_probs, lengths, vocab):
    """Turn CTC output into text: best token per step -> merge repeats -> remove blanks.

    log_probs: (batch, steps, vocab_size)
    lengths:   (batch,) real number of steps per line
    returns:   list of strings, one per line
    """
    # 1. best token at every step: (batch, steps, vocab_size) -> (batch, steps)
    best_ids = log_probs.argmax(dim=-1).cpu().tolist()
    lengths = lengths.cpu().tolist()

    texts = []
    for ids, length in zip(best_ids, lengths):
        # 2. cut off the padding
        ids = ids[:length]

        # 3. merge repeats: keep an ID only if it differs from the one before it
        merged = []
        previous = None
        for idx in ids:
            if idx != previous:
                merged.append(idx)
            previous = idx

        # 4. vocab.decode skips <blank> (and the other special tokens) and maps IDs to characters.
        # merging must happen before this, otherwise "l - l" would become "ll" -> "l"
        texts.append(vocab.decode(merged))

    return texts


def log_add(a, b):
    """log(exp(a) + exp(b)) without leaving log space (so tiny probabilities do not become 0)."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    bigger, smaller = max(a, b), min(a, b)
    return bigger + math.log1p(math.exp(smaller - bigger))


def ctc_beam_search_line(log_probs, vocab, width=5, prune=-12.0):
    """CTC prefix beam search for ONE line.

    Keeps the `width` best text prefixes. For every prefix we track two log probabilities:
      p_blank:    all paths that produced this prefix and end in a blank
      p_nonblank: all paths that produced this prefix and end in its last character
    Paths that collapse to the same prefix are added together, which greedy decoding cannot do.

    log_probs: (steps, vocab_size) for one line, padding already cut off
    prune:     characters with a log probability below this at a step are skipped (for speed)
    returns:   the decoded string
    """
    NEG_INF = -math.inf
    blank = vocab.blank

    # start: the empty prefix, reached with probability 1 (log 0) "ending in a blank"
    beams = {(): (0.0, NEG_INF)}             # prefix (tuple of IDs) -> (p_blank, p_nonblank)

    for step_log_probs in log_probs.tolist():
        # only look at characters that are not practically impossible at this step
        candidates = [c for c, p in enumerate(step_log_probs) if p > prune or c == blank]
        next_beams = {}

        def add(prefix, p_blank=NEG_INF, p_nonblank=NEG_INF):
            old_blank, old_nonblank = next_beams.get(prefix, (NEG_INF, NEG_INF))
            next_beams[prefix] = (log_add(old_blank, p_blank), log_add(old_nonblank, p_nonblank))

        for prefix, (p_blank, p_nonblank) in beams.items():
            p_total = log_add(p_blank, p_nonblank)
            last = prefix[-1] if prefix else None

            for c in candidates:
                p = step_log_probs[c]
                if c == blank:
                    # a blank adds no text: same prefix, now ending in a blank
                    add(prefix, p_blank=p_total + p)
                elif c == last:
                    # same character as the last one:
                    #   after the character itself -> repeats merge, the prefix stays the same
                    add(prefix, p_nonblank=p_nonblank + p)
                    #   after a blank -> it is a new character: "l - l" gives a double "ll"
                    add(prefix + (c,), p_nonblank=p_blank + p)
                else:
                    # any other character extends the prefix
                    add(prefix + (c,), p_nonblank=p_total + p)

        # keep only the `width` most probable prefixes
        ranked = sorted(next_beams.items(), key=lambda item: log_add(*item[1]), reverse=True)
        beams = dict(ranked[:width])

    best_prefix = max(beams.items(), key=lambda item: log_add(*item[1]))[0]
    return vocab.decode(list(best_prefix))


def ctc_beam_decode(log_probs, lengths, vocab, width=5):
    """Beam search for a whole batch, one line at a time. Returns a list of strings."""
    log_probs = log_probs.detach().cpu()
    lengths = lengths.cpu().tolist()
    return [ctc_beam_search_line(log_probs[i, :lengths[i]], vocab, width)
            for i in range(len(lengths))]


# this block only runs with "python -m models.ctc" (from the ocr/ folder): decoding checks
if __name__ == "__main__":
    import torch

    from data.charset import Vocab

    vocab = Vocab()

    def fake_log_probs(tokens):
        """Build log_probs where each step's best token is the given one ('-' = blank)."""
        log_probs = torch.full((1, len(tokens), len(vocab)), -10.0)
        for step, token in enumerate(tokens):
            idx = vocab.blank if token == "-" else vocab.char2idx[token]
            log_probs[0, step, idx] = 0.0
        return log_probs

    tests = [
        (["a", "a", "-", "a", "b"], "aab"),                       # blank keeps the two a's apart
        (["t", "a", "l", "-", "l"], "tall"),                      # double letter needs a blank
        (["t", "a", "l", "l"], "tal"),                            # without the blank they merge
        (["-", "d", "d", "-", "o", "o", "o", "-", "g", "-"], "dog"),
    ]
    for tokens, expected in tests:
        log_probs = fake_log_probs(tokens)
        result = ctc_greedy_decode(log_probs, torch.tensor([len(tokens)]), vocab)[0]
        print(f"{''.join(tokens):12} -> {result!r:8} expected {expected!r:8}",
              "OK" if result == expected else "FAIL")

    # padding must be ignored: only the first 3 steps are real
    log_probs = fake_log_probs(["d", "o", "g", "x", "x"])
    result = ctc_greedy_decode(log_probs, torch.tensor([3]), vocab)[0]
    print(f"length cut   -> {result!r:8} expected 'dog'   ", "OK" if result == "dog" else "FAIL")

    # beam search: the probability of "a" is spread over several steps.
    # step 0: t (1.0) | steps 1, 2: a (0.4) or blank (0.6)
    # greedy takes the single best path "t - -" (0.6 * 0.6 = 0.36) -> "t"
    # but "ta" can be reached by three paths: "t a a", "t a -", "t - a" -> 0.16 + 0.24 + 0.24 = 0.64
    import math
    log_probs = torch.full((1, 3, len(vocab)), -30.0)
    log_probs[0, 0, vocab.char2idx["t"]] = 0.0
    for step in [1, 2]:
        log_probs[0, step, vocab.char2idx["a"]] = math.log(0.4)
        log_probs[0, step, vocab.blank] = math.log(0.6)
    greedy = ctc_greedy_decode(log_probs, torch.tensor([3]), vocab)[0]
    beam = ctc_beam_decode(log_probs, torch.tensor([3]), vocab, width=5)[0]
    print(f"spread 'a'   -> greedy {greedy!r}, beam {beam!r}   expected 't' and 'ta'",
          "OK" if (greedy, beam) == ("t", "ta") else "FAIL")

    # the deterministic cases above must give the same result with beam search
    for tokens, expected in tests:
        log_probs = fake_log_probs(tokens)
        beam = ctc_beam_decode(log_probs, torch.tensor([len(tokens)]), vocab, width=5)[0]
        print(f"beam {''.join(tokens):12} -> {beam!r:8}", "OK" if beam == expected else "FAIL")
