import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMEncoder(nn.Module):
    """Reads the CNN's column features in both directions, so every step gets context
    from the whole line.

    input:   (batch, steps, input_size)       CNN features
    lengths: (batch,)                          real (unpadded) number of steps per line
    output:  (batch, steps, 2 * hidden_size)   forward and backward outputs joined together
    """

    def __init__(self, input_size=256, hidden_size=256, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,   # one LSTM reads left -> right, another right -> left
            batch_first=True,     # tensors are (batch, steps, features), like the CNN output
        )
        # the CTC head (and later the attention decoder) reads this to know its input size
        self.output_size = 2 * hidden_size

    def forward(self, features, lengths):
        steps = features.shape[1]

        # packing tells the LSTM each line's real length, so it skips the padding.
        # without it, the backward LSTM would start reading in the padding, not at the text.
        # (PyTorch wants the lengths on the CPU; enforce_sorted=False means the batch
        # does not have to be sorted by length)
        packed = pack_padded_sequence(features, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed)

        # back to a normal padded tensor; total_length keeps the same number of steps as the input
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=steps)
        return output


def positional_encoding(steps, size, device):
    """Sinusoidal position signal (Vaswani et al. 2017): shape (steps, size).

    Position p gets sin(p / 10000^(2i/size)) in the even features and cos(...) in the odd ones:
    waves of many different frequencies, so every position has its own unique pattern and
    nearby positions have similar ones. Nothing is learned, and it works for any line length.
    """
    positions = torch.arange(steps, device=device).unsqueeze(1)             # (steps, 1)
    # one frequency per pair of features, from fast (i = 0) to very slow
    frequencies = torch.exp(torch.arange(0, size, 2, device=device) * (-math.log(10000.0) / size))
    encoding = torch.zeros(steps, size, device=device)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies)
    return encoding


class TransformerEncoder(nn.Module):
    """Every step attends to every other step of the line (self-attention), so each strip
    gets context from the whole line in one go instead of step by step like the BiLSTM.

    input:   (batch, steps, input_size)   CNN features
    lengths: (batch,)                      real (unpadded) number of steps per line
    output:  (batch, steps, d_model)       padding steps are set to 0, like the BiLSTM's

    Sized to match the BiLSTM (~2.4M vs 2.6M parameters) and, like it, without dropout.
    """

    def __init__(self, input_size=256, d_model=256, num_heads=4, feedforward_size=1024,
                 num_layers=3, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        # the CNN gives 256 features; project only if the Transformer uses a different size
        self.input_projection = nn.Linear(input_size, d_model) if input_size != d_model else nn.Identity()

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,                  # 4 attention heads, each of size 256 / 4 = 64
            dim_feedforward=feedforward_size,
            dropout=dropout,
            batch_first=True,                 # (batch, steps, features), like the rest of the model
            norm_first=True,                  # "pre-norm": layer norm BEFORE attention and the
                                              # feed-forward part; trains stably without a
                                              # learning-rate warm-up
        )
        # with pre-norm, a final layer norm after the last layer is the standard setup
        self.layers = nn.TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(d_model),
                                            enable_nested_tensor=False)
        self.output_size = d_model

    def forward(self, features, lengths):
        batch_size, steps, _ = features.shape
        x = self.input_projection(features)

        # self-attention ignores order, so add the position signal to every step
        x = x + positional_encoding(steps, self.d_model, x.device).unsqueeze(0)

        # padding mask: True = padding (PyTorch's convention here, the opposite of ours
        # in the attention decoder); padding steps get attention weight 0
        positions = torch.arange(steps, device=x.device).unsqueeze(0)          # (1, steps)
        padding = positions >= lengths.to(x.device).unsqueeze(1)               # (batch, steps)

        output = self.layers(x, src_key_padding_mask=padding)

        # padding steps still get an output vector; set it to 0 so it cannot leak into anything
        return output.masked_fill(padding.unsqueeze(-1), 0.0)


# this block only runs with "python -m models.encoders" (from the ocr/ folder): shape check
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from data.build_dataset import OCRDataset, collate_fn
    from models.cnn import CNN

    dataset = OCRDataset("data/rendered/h32_s1.0/train")
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    batch = next(iter(loader))

    cnn = CNN()
    cnn.eval()
    with torch.no_grad():
        features = cnn(batch["images"])
    lengths = CNN.output_widths(batch["image_widths"])
    print("CNN features:  ", tuple(features.shape))
    print("real lengths:  ", lengths.tolist())

    for name, encoder in [("BiLSTM", BiLSTMEncoder(input_size=cnn.out_channels)),
                          ("Transformer", TransformerEncoder(input_size=cnn.out_channels))]:
        encoder.eval()
        with torch.no_grad():
            encoded = encoder(features, lengths)

            # padding steps must come out as zeros
            shortest = int(lengths.argmin())
            padding_zero = bool((encoded[shortest, lengths[shortest]:] == 0).all())

            # the padding must not change the real steps: run the shortest line alone,
            # without any padding, and compare with its output inside the padded batch
            alone = encoder(features[shortest:shortest + 1, :lengths[shortest]], lengths[shortest:shortest + 1])
            same = torch.allclose(alone[0], encoded[shortest, :lengths[shortest]], atol=1e-4)

        print(f"\n{name}: output {tuple(encoded.shape)} | "
              f"parameters {sum(p.numel() for p in encoder.parameters()):,}")
        print(f"  padding output all zero: {padding_zero} | padding does not affect real steps: {same}")
