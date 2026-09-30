from torch import nn

from models.cnn import CNN


class OCRModel(nn.Module):
    """CNN -> encoder -> head. The three parts are passed in, so the same class
    builds every system, e.g. OCRModel(CNN(), BiLSTMEncoder(), CTCHead(...)).
    """

    def __init__(self, cnn, encoder, head, ctc_head=None):
        super().__init__()
        self.cnn = cnn
        self.encoder = encoder
        self.head = head
        # optional extra CTC head, only used as a training signal for the attention system
        # (joint CTC + attention loss); decoding always uses self.head
        self.ctc_head = ctc_head

    def encode(self, images, image_widths):
        """CNN + encoder: the part that is the same for every head."""
        features = self.cnn(images)                  # (batch, steps, 256)
        lengths = CNN.output_widths(image_widths)    # (batch,) real number of steps per line
        encoded = self.encoder(features, lengths)    # (batch, steps, 512)
        return encoded, lengths

    def forward(self, images, image_widths):
        """Full CTC model. (The attention head needs extra inputs, so train.py calls
        encode() and the attention head itself instead.)"""
        encoded, lengths = self.encode(images, image_widths)
        log_probs = self.head(encoded)               # (batch, steps, vocab_size)

        # lengths are returned too: CTC loss needs them as input_lengths,
        # and decoding needs them to stop before the padding
        return log_probs, lengths


# this block only runs with "python -m models.ocr_model" (from the ocr/ folder): full model check
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from data.build_dataset import OCRDataset, collate_fn
    from data.charset import Vocab
    from models.ctc import CTCHead
    from models.encoders import BiLSTMEncoder

    vocab = Vocab()
    cnn = CNN()
    encoder = BiLSTMEncoder(input_size=cnn.out_channels)
    head = CTCHead(input_size=encoder.output_size, vocab_size=len(vocab))
    model = OCRModel(cnn, encoder, head)

    dataset = OCRDataset("data/rendered/h32_s1.0/train")
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    batch = next(iter(loader))

    log_probs, lengths = model(batch["images"], batch["image_widths"])
    print("images:     ", tuple(batch["images"].shape))
    print("log_probs:  ", tuple(log_probs.shape))
    print("lengths:    ", lengths.tolist())
    print("parameters: ", sum(p.numel() for p in model.parameters()))

    # each step is a probability distribution: exp(log_probs) must add up to 1 over the tokens
    print("probabilities sum to 1:", bool(log_probs.exp().sum(dim=-1).sub(1).abs().max() < 1e-4))
