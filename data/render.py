import math
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# folder where matplotlib keeps its bundled .ttf fonts
FONT_DIR = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"

# fonts used for training (and for the "seen fonts" test condition)
TRAIN_FONTS = [
    # DejaVu Sans
    "DejaVuSans.ttf",
    "DejaVuSans-Bold.ttf",
    "DejaVuSans-Oblique.ttf",
    "DejaVuSans-BoldOblique.ttf",
    # DejaVu Serif
    "DejaVuSerif.ttf",
    "DejaVuSerif-Bold.ttf",
    "DejaVuSerif-Italic.ttf",
    "DejaVuSerif-BoldItalic.ttf",
    # DejaVu Sans Mono
    "DejaVuSansMono.ttf",
    "DejaVuSansMono-Bold.ttf",
    "DejaVuSansMono-Oblique.ttf",
    "DejaVuSansMono-BoldOblique.ttf",
    # Computer Modern (LaTeX fonts)
    "cmr10.ttf",
    "cmb10.ttf",
    "cmss10.ttf",
    "cmti10.ttf",
    "cmtt10.ttf",
]

# fonts never seen in training, only used for the "held-out fonts" test condition
HELDOUT_FONTS = [
    # STIX General
    "STIXGeneral.ttf",
    "STIXGeneralBol.ttf",
    "STIXGeneralItalic.ttf",
    "STIXGeneralBolIta.ttf",
]

# ---- random ranges (each image picks a value inside these) ----
FONT_SIZE_RANGE = (16, 48)  # small sizes come out softer after scaling -> varies sharpness
PADDING_RANGE = (2, 10)     # more padding -> text fills less of the image height

# ---- augmentation limits, used at strength = 1.0 ----
MAX_ROTATION = 1.5          # degrees
MAX_ROTATION_DRIFT = 0.25   # the end of the line may move up/down at most 25% of the line height
MAX_BLUR = 1.0              # Gaussian blur radius in pixels (of the final image)
MAX_NOISE = 12              # standard deviation of the added noise (pixel values are 0-255)
MAX_TEXT_GRAY = 80          # text can be anything from black (0) up to this gray
MIN_BACKGROUND_GRAY = 180   # background can be anything from white (255) down to this gray


def render_line(text, font_name, font_size, padding):
    """Draw one line of black text on a white background, at the font's natural size."""
    font = ImageFont.truetype(str(FONT_DIR / font_name), font_size)

    # getbbox gives (left, top, right, bottom) of the drawn text.
    # the vertical size comes from a fixed reference string with the tallest (A, d, ")
    # and lowest (g, j, y) characters, not from the line itself. this way:
    #   - the baseline sits at the same place for every line of a font
    #   - every font fills the image height the same way (some fonts, like STIX,
    #     reserve a lot of empty space in their metrics, which would shrink the text)
    _, top, _, bottom = font.getbbox('Adgjy"')

    # the width comes from the line itself
    right = font.getbbox(text)[2]

    width = right + 2 * padding
    canvas_height = (bottom - top) + 2 * padding

    # "L" = 8-bit grayscale, 255 = white
    image = Image.new("L", (width, canvas_height), 255)
    draw = ImageDraw.Draw(image)
    # shift up by "top" so the tallest character starts right after the padding
    draw.text((padding, padding - top), text, font=font, fill=0)  # 0 = black

    return image


def resize_to_height(image, height):
    """Scale to a fixed height, keeping the aspect ratio so the text is not squashed."""
    new_width = round(image.width * height / image.height)
    return image.resize((new_width, height), Image.BILINEAR)


def rotate(image, rng, strength):
    # a long line rotated by even 1.5 degrees moves its end up/down a lot, which makes the
    # image much taller and the text tiny after resizing. so the angle is also limited
    # such that the end of the line drifts at most MAX_ROTATION_DRIFT of the line height
    drift_limit = math.degrees(math.atan(MAX_ROTATION_DRIFT * image.height / image.width))
    max_angle = min(MAX_ROTATION, drift_limit) * strength
    angle = rng.uniform(-max_angle, max_angle)

    # expand=True grows the canvas so the corners are not cut off; new area is filled white
    return image.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=255)


def blur(image, rng, strength):
    radius = rng.uniform(0, MAX_BLUR * strength)
    return image.filter(ImageFilter.GaussianBlur(radius))


def contrast_and_noise(image, rng, strength):
    # work with numbers instead of a PIL image; float so values can go outside 0-255 for a moment
    pixels = np.array(image, dtype=np.float32)

    # contrast jitter: text (0) becomes a dark gray, background (255) becomes a light gray
    text_gray = rng.uniform(0, MAX_TEXT_GRAY * strength)
    background_gray = rng.uniform(255 - (255 - MIN_BACKGROUND_GRAY) * strength, 255)
    pixels = text_gray + (pixels / 255) * (background_gray - text_gray)

    # additive Gaussian noise: add a small random value to every pixel
    noise_std = rng.uniform(0, MAX_NOISE * strength)
    pixels = pixels + rng.normal(0, noise_std, size=pixels.shape)

    # back to valid pixel values (0-255 whole numbers)
    pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels)


def make_sample(text, index, fonts, height=32, strength=1.0, base_seed=0):
    """Render one training/test image. The same inputs always give the same image.

    index     -> position of the line in its split; together with base_seed it picks the seed
    fonts     -> TRAIN_FONTS or HELDOUT_FONTS
    height    -> final image height in pixels (swept in RQ4)
    strength  -> augmentation strength (swept in RQ3); 0 = clean image, 1 = default limits
    """
    # per-image seed: this image's random choices depend only on its index,
    # not on which or how many other images were rendered before it
    rng = np.random.default_rng(base_seed + index)

    # random choices that are always made (independent of strength)
    font_name = fonts[rng.integers(len(fonts))]
    font_size = int(rng.integers(FONT_SIZE_RANGE[0], FONT_SIZE_RANGE[1] + 1))
    padding = int(rng.integers(PADDING_RANGE[0], PADDING_RANGE[1] + 1))

    image = render_line(text, font_name, font_size, padding)

    # rotation happens before resizing, while the image is still at its natural size
    image = rotate(image, rng, strength)
    image = resize_to_height(image, height)

    # blur and noise happen after resizing, so their sizes are in final-image pixels
    image = blur(image, rng, strength)
    image = contrast_and_noise(image, rng, strength)

    return image


# this block only runs with "python render.py": saves a preview of some rendered samples
if __name__ == "__main__":
    data_dir = Path(__file__).parent
    with open(data_dir / "processed" / "train.en", "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    # a few train lines with training fonts, plus one line per held-out font check
    samples = []
    for index in range(8):
        samples.append(make_sample(lines[index], index, TRAIN_FONTS, height=32))
    for index in range(8, 12):
        samples.append(make_sample(lines[index], index, HELDOUT_FONTS, height=32))

    # stack them vertically on one white sheet
    gap = 6
    sheet_width = max(image.width for image in samples)
    sheet_height = sum(image.height + gap for image in samples)
    sheet = Image.new("L", (sheet_width, sheet_height), 255)
    y = 0
    for image in samples:
        sheet.paste(image, (0, y))
        y += image.height + gap

    sheet.save(data_dir / "sample_preview.png")
    print(f"saved {len(samples)} samples to sample_preview.png")

    # reproducibility check: rendering the same index twice must give identical pixels
    first = np.array(make_sample(lines[5], 5, TRAIN_FONTS))
    second = np.array(make_sample(lines[5], 5, TRAIN_FONTS))
    assert (first == second).all(), "same index gave different images"
    print("reproducibility check OK")
