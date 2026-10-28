# Experimental Week 10 copy of character preparation; production defaults unchanged.
"""Prepare ONE dark handwritten character on pale paper, including its dots.

This module does not split strings or infer labels. All three representations
share the same Otsu localisation and aspect-preserving resize/padding.
"""
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from preprocess_digits import otsu_threshold
from segment_digits import connected_regions

METHODS = ("grayscale", "fixed", "otsu")
DEFAULT_METHOD = "otsu"


def prepare_character_variants(
    image,
    background_size=31,
    filter_distant=False,
    illumination="local",
    remove_specks=True,
    normalise_ink=True,
):
    """Return (method -> float32[28,28], common grayscale crop, diagnostics).

    Expects an upright single-character photo or a complete crop from Manula.
    EXIF orientation is applied. All substantial components contribute to the
    bounding box, including dots; there is no largest-component or height filter.
    Empty representations remain visible for per-method experiment failures.
    """
    if illumination not in ("none", "local"):
        raise ValueError("illumination must be none or local")
    gray = ImageOps.exif_transpose(image).convert("L")
    original_size = list(gray.size)
    gray.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    raw = np.asarray(gray)
    contrast = int(raw.max()) - int(raw.min())
    if contrast < 10:
        raise ValueError(
            "Blank or very low-contrast input; photograph the complete character clearly"
        )
    if illumination == "local":
        # Estimate slowly varying pale paper at bounded resolution. Max filtering
        # fills dark strokes; smoothing avoids block boundaries. Divide the
        # original grayscale by that estimate before the common threshold/crop.
        small = gray.copy()
        small.thumbnail((256, 256), Image.Resampling.LANCZOS)
        background = small.filter(ImageFilter.MaxFilter(background_size)).filter(
            ImageFilter.GaussianBlur(8)
        )
        background = background.resize(gray.size, Image.Resampling.BILINEAR)
        corrected = np.clip(
            raw.astype(np.float32) * 255 / np.maximum(np.asarray(background), 1), 0, 255
        )
        adjusted = ImageOps.autocontrast(
            Image.fromarray(np.rint(corrected).astype(np.uint8))
        )
    else:
        adjusted = ImageOps.autocontrast(gray)
    pixels = np.asarray(adjusted)
    threshold = otsu_threshold(pixels)
    mask = pixels <= threshold
    removed = 0
    if remove_specks:
        labels, regions = connected_regions(mask)
        minimum_area = max(3, max((r["area"] for r in regions), default=0) * 0.005)
        kept = [r["id"] for r in regions if r["area"] >= minimum_area]
        if filter_distant and regions:
            anchor = max(regions, key=lambda r: r["area"])
            a, b, c, d = anchor["box"]
            height = d - b
            retained = []
            for r in regions:
                x0, y0, x1, y1 = r["box"]
                gap = np.hypot(max(0, a - x1, x0 - c), max(0, b - y1, y0 - d))
                # Keep every substantial fragment and nearby detached dot.
                if r["id"] in kept and (
                    r["area"] >= 0.1 * anchor["area"] or gap <= 0.5 * height
                ):
                    retained.append(r["id"])
            kept = retained
        removed = len(regions) - len(kept)
        mask = np.isin(labels, kept)
        # Erase only rejected components in all three representations, retaining
        # grayscale intensity outside the foreground classification.
        pixels = pixels.copy()
        pixels[(labels > 0) & ~mask] = 255
        adjusted = Image.fromarray(pixels)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("No foreground found")
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    warnings = []
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        warnings.append(
            "Foreground touches image edge: inspect clipping, shadow or paper boundary"
        )
    if mask.mean() > 0.45:
        warnings.append(
            "Large foreground area: background may be mistaken for handwriting"
        )
    views = {
        "grayscale": ImageOps.invert(adjusted),
        "fixed": Image.fromarray(np.where(pixels <= 128, 255, 0).astype(np.uint8)),
        "otsu": Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)),
    }
    width, height = box[2] - box[0], box[3] - box[1]
    scale = 20 / max(width, height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    variants = {}
    for method, view in views.items():
        crop = view.crop(box).resize(size, Image.Resampling.LANCZOS)
        canvas = Image.new("L", (28, 28), 0)
        canvas.paste(crop, ((28 - size[0]) // 2, (28 - size[1]) // 2))
        array = np.asarray(canvas, dtype=np.float32) / 255.0
        # Downsampling very thin pen strokes can leave all pixels faint.
        # A common peak normalisation restores contrast without thickening ink.
        if normalise_ink and array.max() > 0:
            array = array / array.max()
        variants[method] = array
    details = {
        "illumination": illumination,
        "remove_specks": remove_specks,
        "normalise_ink": normalise_ink,
        "removed_components": removed,
        "original_size": original_size,
        "working_size": list(gray.size),
        "bbox": list(box),
        "resized_ink_size": list(size),
        "contrast": contrast,
        "otsu_threshold": threshold,
        "foreground_fraction": float(mask.mean()),
        "warnings": warnings,
    }
    return variants, gray.crop(box), details
