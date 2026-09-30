import torch
from torch import nn


def conv_block(in_channels, out_channels):
    """3x3 convolution -> batch norm -> ReLU. Keeps height and width the same (padding=1)."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class CNN(nn.Module):
    """Turns a line image into a sequence of column feature vectors.

    input:  (batch, 1, height, width)            grayscale image
    output: (batch, width // 4, out_channels)    one feature vector per 4-pixel-wide strip

    The same CNN is shared by all four systems (CTC / attention x BiLSTM / Transformer).
    """

    def __init__(self, out_channels=256):
        super().__init__()
        self.out_channels = out_channels

        self.layers = nn.Sequential(
            # MaxPool2d((a, b)) divides the height by a and the width by b
            conv_block(1, 64),
            nn.MaxPool2d((2, 2)),             # height /2, width /2
            conv_block(64, 128),
            nn.MaxPool2d((2, 2)),             # height /2, width /2  -> width is now /4 in total
            conv_block(128, 256),
            conv_block(256, 256),
            nn.MaxPool2d((2, 1)),             # height /2 only, width stays /4
            conv_block(256, out_channels),
            conv_block(out_channels, out_channels),
            nn.MaxPool2d((2, 1)),             # height /2 only
            # whatever height is left (2 for 32 px images, 4 for 64 px, ...) is averaged to 1.
            # this makes the CNN work for every image height in the RQ4 sweep
            nn.AdaptiveAvgPool2d((1, None)),  # None = leave the width as it is
        )

    def forward(self, images):
        features = self.layers(images)          # (batch, channels, 1, width // 4)
        features = features.squeeze(2)          # (batch, channels, width // 4)   drop the height axis
        features = features.permute(0, 2, 1)    # (batch, width // 4, channels)   sequence axis first
        return features

    @staticmethod
    def output_widths(image_widths):
        """Real (unpadded) length of each output sequence: the CNN divides the width by 4.

        The encoder uses this to ignore padding, and CTC needs it as the input lengths.
        """
        return image_widths // 4


# this block only runs with "python -m models.cnn" (from the ocr/ folder): shape check on a real batch
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from data.build_dataset import OCRDataset, collate_fn

    dataset = OCRDataset("data/rendered/h32_s1.0/train")
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    batch = next(iter(loader))

    cnn = CNN()
    features = cnn(batch["images"])
    print("images:        ", tuple(batch["images"].shape))
    print("features:      ", tuple(features.shape))
    print("image widths:  ", batch["image_widths"].tolist())
    print("output lengths:", CNN.output_widths(batch["image_widths"]).tolist())
    print("label lengths: ", batch["label_lengths"].tolist())
    print("parameters:    ", sum(p.numel() for p in cnn.parameters()))

    # the CNN must also work for other image heights (RQ4)
    for height in [24, 48, 64]:
        out = cnn(torch.zeros(2, 1, height, 400))
        print(f"height {height}: output {tuple(out.shape)}")
