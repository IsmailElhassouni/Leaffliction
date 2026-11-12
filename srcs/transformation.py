"""
Transformation.py logic (Part 3: Image Transformation).

Implements characteristic-extraction transforms for a leaf image:
  1. Original
  2. Gaussian blur
  3. Mask (leaf segmented from background)
  4. Roi objects (region of interest box around the leaf)
  5. Analyze object (contour / shape analysis overlay)
  6. Pseudolandmarks (boundary landmark points)
  7. Color histogram (per-channel intensity distribution)

`create_mask` / `apply_mask` are reused by predict.py to build the
"transformed image" shown next to the original at prediction time.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np


def gaussian_blur(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(image, (5, 5), 0)


def create_mask(image: np.ndarray) -> np.ndarray:
    """
    Segment the leaf from its (usually plain) background.

    Combines an HSV saturation threshold (works for green/yellow/brown
    leaves on a light background) with Otsu thresholding on the
    a-channel of L*a*b* (robust to reddish/necrotic lesions), then
    cleans the result with morphological operations.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    _, mask_sat = cv2.threshold(
        sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1]
    _, mask_a = cv2.threshold(
        a_channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    mask = cv2.bitwise_or(mask_sat, mask_a)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Keep only the largest connected component (the leaf itself).
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if contours:
        largest = max(contours, key=cv2.contourArea)
        clean = np.zeros_like(mask)
        cv2.drawContours(clean, [largest], -1, 255, thickness=cv2.FILLED)
        mask = clean
    return mask


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(image, image, mask=mask)


def largest_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def roi_objects(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Draw the region-of-interest bounding box around the leaf."""
    out = image.copy()
    contour = largest_contour(mask)
    if contour is not None:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 3)
    return out


def analyze_object(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Draw the leaf's contour, centroid and enclosing ellipse."""
    out = image.copy()
    contour = largest_contour(mask)
    if contour is not None:
        cv2.drawContours(out, [contour], -1, (0, 0, 255), 2)
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            cv2.circle(out, (cx, cy), 5, (255, 0, 255), -1)
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(out, ellipse, (255, 255, 0), 2)
    return out


def pseudolandmarks(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Sample evenly-spaced points along the leaf boundary and color-code
    them by side (top / bottom / left / right), similar in spirit to
    PlantCV's pseudolandmark scan.
    """
    out = image.copy()
    contour = largest_contour(mask)
    if contour is None:
        return out
    pts = contour.reshape(-1, 2)
    n_points = min(60, len(pts))
    idx = np.linspace(0, len(pts) - 1, n_points).astype(int)
    sample = pts[idx]

    cx, cy = sample[:, 0].mean(), sample[:, 1].mean()
    colors = {
        "top": (255, 0, 0),
        "bottom": (0, 255, 255),
        "left": (0, 128, 255),
        "right": (255, 0, 255),
    }
    for x, y in sample:
        if y < cy and abs(y - cy) >= abs(x - cx):
            color = colors["top"]
        elif y >= cy and abs(y - cy) >= abs(x - cx):
            color = colors["bottom"]
        elif x < cx:
            color = colors["left"]
        else:
            color = colors["right"]
        cv2.circle(out, (int(x), int(y)), 3, color, -1)
    return out


def color_histogram(image: np.ndarray, mask: np.ndarray | None = None):
    """
    Build a per-channel color histogram (BGR + HSV) restricted to the
    masked leaf pixels. Returns a matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    channels = {
        "blue": (image, 0, "b"),
        "green": (image, 1, "g"),
        "red": (image, 2, "r"),
        "hue": (hsv, 0, "m"),
        "saturation": (hsv, 1, "c"),
        "value": (hsv, 2, "y"),
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, (src, ch, color) in channels.items():
        hist = cv2.calcHist([src], [ch], mask, [256], [0, 256])
        hist = hist / (hist.sum() + 1e-8)
        ax.plot(hist, color=color, label=label)
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Proportion of pixels")
    ax.set_title("Color histogram")
    ax.legend(fontsize="small")
    fig.tight_layout()
    return fig


def all_transforms(image: np.ndarray) -> "dict[str, np.ndarray]":
    """Compute the 6 required transformations for one BGR image."""
    mask = create_mask(image)
    return {
        "Original": image,
        "Gaussian blur": gaussian_blur(image),
        "Mask": apply_mask(image, mask),
        "Roi objects": roi_objects(image, mask),
        "Analyze object": analyze_object(image, mask),
        "Pseudolandmarks": pseudolandmarks(image, mask),
    }


def _save_single(image_path: str) -> None:
    import matplotlib.pyplot as plt

    image = cv2.imread(image_path)
    if image is None:
        print(f"error: no such file: {image_path}", file=sys.stderr)
        sys.exit(1)

    transforms = all_transforms(image)
    fig, axes = plt.subplots(1, len(transforms), figsize=(21, 3.2))
    for ax, (name, img) in zip(axes, transforms.items()):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.show()

    mask = create_mask(image)
    hist_fig = color_histogram(image, mask)
    hist_fig.show()

    input("Press Enter to close the figures and exit...")


def _save_directory(src: str, dst: str, mask_only: bool) -> None:
    os.makedirs(dst, exist_ok=True)
    valid_ext = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    files = [f for f in os.listdir(src) if f.endswith(valid_ext)]
    for fname in files:
        image = cv2.imread(os.path.join(src, fname))
        if image is None:
            continue
        stem, ext = os.path.splitext(fname)
        if mask_only:
            mask = create_mask(image)
            out = apply_mask(image, mask)
            cv2.imwrite(os.path.join(dst, f"{stem}_Mask{ext}"), out)
            continue
        for name, img in all_transforms(image).items():
            safe = name.replace(" ", "_")
            cv2.imwrite(os.path.join(dst, f"{stem}_{safe}{ext}"), img)
    print(f"Saved transformations for {len(files)} images to {dst}")


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract leaf characteristics via image transforms.",
    )
    parser.add_argument("image", nargs="?", help="Path to a single image.")
    parser.add_argument("-src", dest="src", help="Source directory.")
    parser.add_argument("-dst", dest="dst", help="Destination directory.")
    parser.add_argument(
        "-mask", dest="mask", action="store_true",
        help="Only produce the mask transform (directory mode).",
    )
    args = parser.parse_args()

    if args.src:
        if not args.dst:
            parser.error("-dst is required when using -src")
        _save_directory(args.src, args.dst, args.mask)
    elif args.image:
        _save_single(args.image)
    else:
        parser.error("provide an image path, or -src/-dst for a directory")


if __name__ == "__main__":
    _main()
