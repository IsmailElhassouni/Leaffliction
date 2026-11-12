"""
Augmentation.py logic (Part 2: Data augmentation).

Provides six augmentation techniques (Flip, Rotate, Skew, Shear, Crop,
Distortion) used to:
  * balance the dataset per class (called from train.py), and
  * be run standalone exactly as described in the subject:

      $> ./Augmentation.py ./Apple/apple_healthy/image\\ \\(1\\).JPG

which writes, next to the source image:

      image (1)_Flip.JPG
      image (1)_Rotate.JPG
      image (1)_Skew.JPG
      image (1)_Shear.JPG
      image (1)_Crop.JPG
      image (1)_Distortion.JPG
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import cv2
import numpy as np


def flip(image: np.ndarray) -> np.ndarray:
    """Mirror the image horizontally."""
    return cv2.flip(image, 1)


def rotate(image: np.ndarray, angle: float | None = None) -> np.ndarray:
    """Rotate the image around its center by a random angle."""
    if angle is None:
        angle = random.uniform(15, 45) * random.choice([-1, 1])
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (w, h),
        borderMode=cv2.BORDER_REFLECT_101,
    )


def skew(image: np.ndarray) -> np.ndarray:
    """Apply a perspective (skew) transform to the image."""
    h, w = image.shape[:2]
    shift = int(0.18 * w)
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst = np.float32([
        [random.randint(0, shift), 0],
        [w - random.randint(0, shift), 0],
        [0, h],
        [w, h],
    ])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        image, matrix, (w, h),
        borderMode=cv2.BORDER_REFLECT_101,
    )


def shear(image: np.ndarray) -> np.ndarray:
    """Apply an affine shear transform to the image."""
    h, w = image.shape[:2]
    factor = random.uniform(0.15, 0.30) * random.choice([-1, 1])
    src = np.float32([[0, 0], [w, 0], [0, h]])
    dst = np.float32([[0, 0], [w, 0], [factor * w, h]])
    matrix = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(
        image, matrix, (w, h),
        borderMode=cv2.BORDER_REFLECT_101,
    )


def crop(image: np.ndarray) -> np.ndarray:
    """Randomly zoom in (crop then resize back to the original size)."""
    h, w = image.shape[:2]
    ratio = random.uniform(0.65, 0.85)
    ch, cw = int(h * ratio), int(w * ratio)
    top = random.randint(0, h - ch)
    left = random.randint(0, w - cw)
    cropped = image[top:top + ch, left:left + cw]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def distortion(image: np.ndarray) -> np.ndarray:
    """Apply a random local (elastic-like) distortion to the image."""
    h, w = image.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    amplitude = 0.02 * min(h, w)
    freq = random.uniform(1.5, 3.0)
    phase = random.uniform(0, 2 * np.pi)
    map_x = (grid_x + amplitude * np.sin(
        2 * np.pi * grid_y / (h / freq) + phase
    )).astype(np.float32)
    map_y = (grid_y + amplitude * np.cos(
        2 * np.pi * grid_x / (w / freq) + phase
    )).astype(np.float32)
    return cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


AUGMENTATIONS: dict[str, callable] = {
    "Flip": flip,
    "Rotate": rotate,
    "Skew": skew,
    "Shear": shear,
    "Crop": crop,
    "Distortion": distortion,
}


def augment_image(image: np.ndarray) -> dict[str, np.ndarray]:
    """Return a dict {augmentation_name: augmented_image} for one image."""
    return {name: fn(image) for name, fn in AUGMENTATIONS.items()}


def save_augmented(
    image_path: str,
    output_dir: str | None = None,
    types: list[str] | None = None,
) -> list[str]:
    """
    Generate augmented versions of one image on disk, following the
    subject's naming convention: <stem>_<Type><ext>.

    Returns the list of written file paths.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    stem, ext = os.path.splitext(os.path.basename(image_path))
    out_dir = output_dir or os.path.dirname(image_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    types = types or list(AUGMENTATIONS.keys())
    written = []
    for name in types:
        augmented = AUGMENTATIONS[name](image)
        out_path = os.path.join(out_dir, f"{stem}_{name}{ext}")
        cv2.imwrite(out_path, augmented)
        written.append(out_path)
    return written


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 6 augmented versions of a leaf image.",
    )
    parser.add_argument("image", help="Path to a source image.")
    parser.add_argument(
        "--output-dir", "-o", default=None,
        help="Directory to write augmented images to "
             "(default: next to the source image).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"error: no such file: {args.image}", file=sys.stderr)
        sys.exit(1)

    paths = save_augmented(args.image, args.output_dir)
    for p in paths:
        print(p)

    try:
        import matplotlib.pyplot as plt

        image = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
        variants = augment_image(cv2.imread(args.image))
        fig, axes = plt.subplots(1, len(variants) + 1, figsize=(18, 3))
        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis("off")
        for ax, (name, img) in zip(axes[1:], variants.items()):
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(name)
            ax.axis("off")
        fig.tight_layout()
        out_png = os.path.splitext(args.image)[0] + "_augmentations.png"
        fig.savefig(out_png, dpi=120)
        print(f"preview saved to {out_png}")
    except Exception as exc:  # pragma: no cover - display is best-effort
        print(f"(preview skipped: {exc})", file=sys.stderr)


if __name__ == "__main__":
    _main()
