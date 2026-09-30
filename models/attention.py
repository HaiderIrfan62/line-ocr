import torch
from torch import nn


class Attention(nn.Module):
    """Decides which part of the image to look at, given the decoder's current state.

    Two ways to score each encoder position j (the "score" setting):
      "additive" (Bahdanau et al. 2015):  score_j = v · tanh(W_h · h_j + W_s · s)
      "dot"      (Luong et al. 2015):     score_j = (W_h · h_j) · s / sqrt(d)

    Optional location term (Chorowski et al. 2015), "where did I look last time?":
      f = Conv1d(previous attention weights), added to the score
      (without it, additive attention found the first letters and then got stuck
      at the start of the line)

    weights = softmax(scores)                 (padding gets weight 0)
    context = weighted average of the encoder outputs
    """

    def __init__(self, encoder_size=512, decoder_size=256, attention_size=256,
                 score="additive", use_location=True, location_filters=10, location_kernel=31):
        super().__init__()
        self.score = score
        self.use_location = use_location
        self.decoder_size = decoder_size

        if score == "additive":
            self.W_h = nn.Linear(encoder_size, attention_size, bias=False)  # projects image positions
            self.W_s = nn.Linear(decoder_size, attention_size)              # projects the decoder state
            self.v = nn.Linear(attention_size, 1, bias=False)               # reduces to one score
        elif score == "dot":
            # the encoder (512) and decoder (256) sizes differ, so one matrix maps
            # image positions to the decoder's size before the dot product
            self.W_h = nn.Linear(encoder_size, decoder_size, bias=False)
        else:
            raise ValueError(f"unknown attention score: {score}")

        if use_location:
            # location features: slide small filters over the previous attention weights, so each
            # position learns "attention was on me / just to my left / just to my right last step".
            # padding = kernel // 2 keeps the number of positions the same
            self.location_conv = nn.Conv1d(1, location_filters, kernel_size=location_kernel,
                                           padding=location_kernel // 2, bias=False)
            if score == "additive":
                self.W_f = nn.Linear(location_filters, attention_size, bias=False)  # added inside tanh
            else:
                self.W_f = nn.Linear(location_filters, 1, bias=False)               # added to the score

    def project_encoder(self, encoded):
        """W_h · h_j for all positions. The encoder outputs do not change while the decoder
        writes, so this is computed once per batch instead of at every step."""
        return self.W_h(encoded)

    def location_features(self, previous_weights):
        # Conv1d wants (batch, channels, steps), so add a channel axis of size 1
        location = self.location_conv(previous_weights.unsqueeze(1))   # (batch, filters, steps)
        location = location.transpose(1, 2)                            # (batch, steps, filters)
        return self.W_f(location)

    def forward(self, encoded, projected_encoded, state, previous_weights, mask):
        """
        encoded:           (batch, steps, encoder_size)   encoder outputs
        projected_encoded: (batch, steps, ...)            from project_encoder
        state:             (batch, decoder_size)          decoder's hidden state s
        previous_weights:  (batch, steps)                 attention weights of the previous step
        mask:              (batch, steps)                 True = real position, False = padding
        """
        if self.score == "additive":
            # W_s·s is one vector per line; unsqueeze(1) lets it be added to every position
            inside = projected_encoded + self.W_s(state).unsqueeze(1)
            if self.use_location:
                inside = inside + self.location_features(previous_weights)
            scores = self.v(torch.tanh(inside)).squeeze(-1)                  # (batch, steps)
        else:
            # dot product of every projected position with the state:
            # (batch, steps, d) * (batch, 1, d) summed over d -> (batch, steps).
            # dividing by sqrt(d) keeps the scores from getting huge
            scores = (projected_encoded * state.unsqueeze(1)).sum(-1) / self.decoder_size ** 0.5
            if self.use_location:
                scores = scores + self.location_features(previous_weights).squeeze(-1)

        # padding positions get -inf, so softmax gives them exactly 0 weight
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)                   # (batch, steps), each row sums to 1

        # weighted average of the encoder outputs, as one matrix multiplication:
        # (batch, 1, steps) x (batch, steps, encoder_size) -> (batch, 1, encoder_size)
        context = torch.bmm(weights.unsqueeze(1), encoded).squeeze(1)  # (batch, encoder_size)
        return context, weights


class AttentionDecoder(nn.Module):
    """Writes the text one character at a time, attending over the encoder outputs."""

    def __init__(self, encoder_size, vocab, embedding_size=128, hidden_size=256,
                 attention_size=256, max_length=160, score="additive", use_location=True):
        super().__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.max_length = max_length   # safety stop for greedy decoding (lines are at most 150 chars)

        self.embedding = nn.Embedding(len(vocab), embedding_size)   # character ID -> vector
        self.attention = Attention(encoder_size, hidden_size, attention_size,
                                   score=score, use_location=use_location)
        # one LSTM step at a time: input = previous character + context vector
        self.lstm_cell = nn.LSTMCell(embedding_size + encoder_size, hidden_size)
        # prediction from "what I have written" (h) and "what I am looking at" (context)
        self.output = nn.Linear(hidden_size + encoder_size, len(vocab))

    def start(self, encoded, lengths):
        """Things that stay the same for every decoding step of a batch."""
        batch_size, steps, _ = encoded.shape
        device = encoded.device

        # mask: True for real positions, False for padding, e.g. length 3 of 5 -> [T, T, T, F, F]
        positions = torch.arange(steps, device=device).unsqueeze(0)   # (1, steps)
        mask = positions < lengths.to(device).unsqueeze(1)            # (batch, steps)

        projected_encoded = self.attention.project_encoder(encoded)

        # starting state: zeros (attention gives the decoder the image information from step 1)
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)

        # "previous" attention for the very first step: all weight on position 0 (start of the line)
        weights = torch.zeros(batch_size, steps, device=device)
        weights[:, 0] = 1.0
        return mask, projected_encoded, h, c, weights

    def step(self, previous_ids, encoded, projected_encoded, mask, h, c, weights):
        """One decoding step: attend -> update the LSTM state -> predict the next character.

        weights: attention weights of the previous step (for the location term)
        """
        # uses the previous state h and the previous attention weights
        context, weights = self.attention(encoded, projected_encoded, h, weights, mask)
        x = torch.cat([self.embedding(previous_ids), context], dim=1)
        h, c = self.lstm_cell(x, (h, c))
        logits = self.output(torch.cat([h, context], dim=1))         # (batch, vocab_size)
        return logits, h, c, weights

    def forward(self, encoded, lengths, decoder_input):
        """Training with teacher forcing: the correct previous character is fed at every step.

        decoder_input: (batch, target_length) from make_decoder_io
        returns logits: (batch, target_length, vocab_size)  raw scores, for cross-entropy loss
        """
        mask, projected_encoded, h, c, weights = self.start(encoded, lengths)
        decoder_input = decoder_input.to(encoded.device)

        all_logits = []
        for t in range(decoder_input.shape[1]):
            logits, h, c, weights = self.step(decoder_input[:, t], encoded, projected_encoded,
                                              mask, h, c, weights)
            all_logits.append(logits)

        # list of (batch, vocab_size) -> (batch, target_length, vocab_size)
        return torch.stack(all_logits, dim=1)

    def greedy_decode(self, encoded, lengths):
        """Testing: no correct text available, so the decoder feeds in its own predictions.

        Stops when every line has produced <eos>, or after max_length steps.
        returns: list of ID lists, one per line (without <sos> and <eos>)
        """
        batch_size = encoded.shape[0]
        mask, projected_encoded, h, c, weights = self.start(encoded, lengths)

        previous_ids = torch.full((batch_size,), self.vocab.sos, dtype=torch.long, device=encoded.device)
        finished = [False] * batch_size                # which lines have produced <eos>
        outputs = [[] for _ in range(batch_size)]

        for _ in range(self.max_length):
            logits, h, c, weights = self.step(previous_ids, encoded, projected_encoded,
                                              mask, h, c, weights)
            previous_ids = logits.argmax(dim=-1)       # best character for every line

            for i, idx in enumerate(previous_ids.tolist()):
                if not finished[i]:
                    if idx == self.vocab.eos:
                        finished[i] = True
                    else:
                        outputs[i].append(idx)

            if all(finished):                          # every line is done: stop early
                break

        return outputs

    def beam_decode(self, encoded, lengths, width=5):
        """Beam search: keep the `width` best partial texts instead of only the best one.

        Lines are decoded one at a time; the candidates of one line go through step()
        together as a small batch of size <= width.
        Finished candidates (that produced <eos>) are compared by their AVERAGE log
        probability per character, otherwise shorter texts would always win.
        returns: list of ID lists, one per line (without <sos> and <eos>)
        """
        outputs = []
        for i in range(encoded.shape[0]):
            length = int(lengths[i])
            line = encoded[i:i + 1, :length]                  # (1, steps, encoder_size), no padding
            mask, projected, h, c, weights = self.start(line, lengths[i:i + 1])

            # alive candidates: their tokens and total log probability
            tokens = [[]]
            scores = torch.zeros(1, device=encoded.device)
            previous_ids = torch.full((1,), self.vocab.sos, dtype=torch.long, device=encoded.device)
            finished = []                                     # (average log prob, tokens)

            for _ in range(self.max_length):
                n = len(tokens)
                # the encoder output is the same for every candidate: repeat it n times
                logits, h, c, weights = self.step(previous_ids, line.expand(n, -1, -1),
                                                  projected.expand(n, -1, -1), mask.expand(n, -1),
                                                  h, c, weights)
                log_probs = torch.log_softmax(logits, dim=-1)           # (n, vocab_size)

                # score of every (candidate, next character) pair, flattened to one list
                total = (scores.unsqueeze(1) + log_probs).view(-1)
                vocab_size = log_probs.shape[1]
                top_scores, top_indices = total.topk(min(2 * width, total.numel()))

                new_tokens, new_scores, keep_beams, keep_ids = [], [], [], []
                for score, index in zip(top_scores.tolist(), top_indices.tolist()):
                    beam, token = divmod(index, vocab_size)
                    if token == self.vocab.eos:
                        # +1 counts the <eos> itself, so a finished text is never length 0
                        finished.append((score / (len(tokens[beam]) + 1), tokens[beam]))
                    else:
                        new_tokens.append(tokens[beam] + [token])
                        new_scores.append(score)
                        keep_beams.append(beam)
                        keep_ids.append(token)
                    if len(new_tokens) == width:
                        break

                if len(finished) >= width or not new_tokens:
                    break

                # continue with the kept candidates: pick their decoder states
                keep = torch.tensor(keep_beams, device=encoded.device)
                h, c, weights = h[keep], c[keep], weights[keep]
                tokens = new_tokens
                scores = torch.tensor(new_scores, device=encoded.device)
                previous_ids = torch.tensor(keep_ids, device=encoded.device)

            if finished:
                outputs.append(max(finished, key=lambda item: item[0])[1])
            else:
                # nothing produced <eos> within max_length: take the best unfinished candidate
                outputs.append(tokens[int(scores.argmax())])
        return outputs


def make_decoder_io(labels, label_lengths, vocab):
    """Build the decoder's input and target from the plain labels (teacher forcing).

    labels:        (batch, max_length)  character IDs padded with <pad>, as made by collate_fn
    label_lengths: (batch,)             real length of each label

    returns, both (batch, max_length + 1):
        decoder_input:  <sos> A ␣ d o g        what the decoder is fed
        decoder_target: A ␣ d o g <eos>        what it should predict at each position
    """
    batch_size = labels.shape[0]

    # input: a column of <sos> in front of the labels (everything shifts one to the right)
    sos_column = torch.full((batch_size, 1), vocab.sos, dtype=torch.long)
    decoder_input = torch.cat([sos_column, labels], dim=1)

    # target: the labels plus one extra <pad> column at the end, to make room for <eos>
    pad_column = torch.full((batch_size, 1), vocab.pad, dtype=torch.long)
    decoder_target = torch.cat([labels, pad_column], dim=1)

    # put <eos> right after each line's real end (not at the end of the padded row)
    for i in range(batch_size):
        decoder_target[i, label_lengths[i]] = vocab.eos

    return decoder_input, decoder_target


# this block only runs with "python -m models.attention" (from the ocr/ folder): check the example
if __name__ == "__main__":
    from data.charset import Vocab

    vocab = Vocab()

    # a tiny batch built by hand: "A dog" and "Hi", padded to the same length
    texts = ["A dog", "Hi"]
    max_length = max(len(text) for text in texts)
    labels = torch.full((len(texts), max_length), vocab.pad, dtype=torch.long)
    for i, text in enumerate(texts):
        labels[i, :len(text)] = torch.tensor(vocab.encode(text))
    label_lengths = torch.tensor([len(text) for text in texts])

    decoder_input, decoder_target = make_decoder_io(labels, label_lengths, vocab)

    def show(row):
        # turn IDs back into tokens, '·' for <pad> and '␣' for space so they are visible
        tokens = []
        for idx in row.tolist():
            token = vocab.idx2char[idx]
            tokens.append({"<pad>": "·", " ": "␣"}.get(token, token))
        return " ".join(tokens)

    for i, text in enumerate(texts):
        print(f"{text!r}")
        print("  input: ", show(decoder_input[i]))
        print("  target:", show(decoder_target[i]))

    # ---- decoder check on a real batch: CNN -> BiLSTM -> attention decoder ----
    from torch.utils.data import DataLoader

    from data.build_dataset import OCRDataset, collate_fn
    from models.cnn import CNN
    from models.encoders import BiLSTMEncoder

    dataset = OCRDataset("data/rendered/h32_s1.0/train")
    batch = next(iter(DataLoader(dataset, batch_size=4, collate_fn=collate_fn)))

    cnn = CNN()
    encoder = BiLSTMEncoder(input_size=cnn.out_channels)
    decoder = AttentionDecoder(encoder_size=encoder.output_size, vocab=vocab)

    lengths = CNN.output_widths(batch["image_widths"])
    encoded = encoder(cnn(batch["images"]), lengths)
    decoder_input, decoder_target = make_decoder_io(batch["labels"], batch["label_lengths"], vocab)

    logits = decoder(encoded, lengths, decoder_input)
    print("\nencoded:        ", tuple(encoded.shape))
    print("decoder_input:  ", tuple(decoder_input.shape))
    print("logits:         ", tuple(logits.shape))
    print("decoder params: ", sum(p.numel() for p in decoder.parameters()))

    # attention weights: each row must sum to 1, and padding must get exactly 0
    mask, projected, h, c, previous = decoder.start(encoded, lengths)
    _, weights = decoder.attention(encoded, projected, h, previous, mask)
    print("weights sum to 1:      ", bool((weights.sum(dim=-1) - 1).abs().max() < 1e-5))
    print("zero weight on padding:", bool((weights[~mask] == 0).all()))

    # greedy decoding on the untrained model: random text, but it must stop at max_length
    with torch.no_grad():
        outputs = decoder.greedy_decode(encoded, lengths)
    print("greedy output lengths:", [len(ids) for ids in outputs], f"(max {decoder.max_length})")
    print("example (untrained):  ", repr(vocab.decode(outputs[0])[:60]))
