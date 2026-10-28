"""Image acquisition and evidence export for Saniru's desktop interface."""

import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from preprocess_digits import prepare_variants

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MODEL_NAMES = ("cnn", "mlp", "resnet18", "mobilenet_v3_small")
DEFAULT_MODELS = {"Numbers": "cnn", "Alphanumeric": "resnet18"}


def resolve_checkpoint(root, mode, model):
    """Resolve the evaluated architecture for its task and verify before loading."""
    if mode not in DEFAULT_MODELS or model not in MODEL_NAMES:
        raise ValueError("Unsupported recognition mode or model")
    root = Path(root)
    if model in ("resnet18", "mobilenet_v3_small"):
        task = "mnist" if mode == "Numbers" else "emnist"
        folder = root / "experiments/week11_gpu_run01" / task / model
        selection = json.loads((folder / "selection.json").read_text())
        if (
            selection.get("model") != model
            or selection.get("config", {}).get("task") != task
        ):
            raise ValueError(
                "Model selection does not match recognition mode or architecture."
            )
        expected = selection["checkpoint_sha256"]
    else:
        folder = root / (
            "experiments/baseline"
            if mode == "Numbers"
            else "experiments/emnist_byclass/run_01"
        )
        selection = json.loads((folder / "selection.json").read_text())
        expected = next(
            r["checkpoint_sha256"] for r in selection["models"] if r["model"] == model
        )
    checkpoint = folder / f"{model}.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(
            "Model checkpoint hash mismatch; restore the saved experiment."
        )
    return checkpoint, digest


def character_catalog(folder):
    """Index explicit, case-sensitive single-character labels, never strings."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError("Select a folder containing individual character photos.")
    catalog = {}
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_TYPES:
            continue
        match = re.search(r"(?:^|_)(digit-[0-9]|upper-[A-Z]|lower-[a-z])$", path.stem)
        if match:
            character = match[1][-1]
        elif re.fullmatch(r"[A-Za-z0-9]", path.stem):
            character = path.stem
        else:
            continue
        catalog.setdefault(character, []).append(path)
    return catalog


def generate_text(folder, text, seed=42):
    """Compose handwriting tiles; labels choose sources but never guide prediction."""
    from hnrs_chars.preprocessing import prepare_character

    if not re.fullmatch(r"[A-Za-z0-9]{1,12}", text):
        raise ValueError("Enter 1 to 12 characters: A-Z, a-z or 0-9, without spaces.")
    catalog = character_catalog(folder)
    missing = sorted(set(text) - catalog.keys())
    if missing:
        raise ValueError(
            "No single-character photo for: "
            + ", ".join(missing)
            + ". Case matters. Use Name_01_upper-A.jpg, Name_02_lower-a.jpg or Name_03_digit-0.jpg."
        )
    rng = random.Random(seed)
    sources = []
    canvas = Image.new(
        "RGB", (32 + len(text) * 112 + (len(text) - 1) * 16, 144), "white"
    )
    for i, character in enumerate(text):
        path = rng.choice(catalog[character])
        try:
            array = prepare_character(open_photo(path))
        except (ValueError, OSError) as exc:
            raise ValueError(f"Cannot prepare {path.name}: {exc}") from exc
        tile = Image.fromarray(255 - np.rint(array * 255).astype("uint8")).resize(
            (112, 112), Image.Resampling.NEAREST
        )
        canvas.paste(tile, (16 + i * 128, 16))
        sources.append(
            {
                "character": character,
                "file": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return canvas, {
        "route": "generated",
        "mode": "Alphanumeric",
        "requested_text": text,
        "seed": seed,
        "sources": sources,
        "warnings": [],
        "note": "Composed from labelled single-character photos; source labels are not supplied to recognition. Generated images are not independent handwritten-string evaluation.",
    }


class RecognitionService:
    """GUI worker service; mode-specific checkpoints and no reference-label input."""

    def __init__(self, root):
        self.root = Path(root)
        self.predictors = {}
        # Models trained in the Model lab this session: mode -> name -> (path, sha256).
        self.custom = {mode: {} for mode in DEFAULT_MODELS}

    def register_custom(self, mode, name, checkpoint, digest):
        """Make a Model-lab checkpoint selectable; its hash is rechecked on use."""
        if mode not in DEFAULT_MODELS:
            raise ValueError("Unsupported recognition mode")
        if name in MODEL_NAMES:
            raise ValueError("Custom model names must differ from the shipped models")
        self.custom[mode][name] = (Path(checkpoint), digest)

    def models_for(self, mode):
        return list(MODEL_NAMES) + list(self.custom[mode])

    def resolve(self, mode, model):
        if model in self.custom.get(mode, {}):
            checkpoint, expected = self.custom[mode][model]
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected:
                raise ValueError(
                    "Trained checkpoint changed since training; train it again."
                )
            return checkpoint, expected
        return resolve_checkpoint(self.root, mode, model)

    def recognise(self, image, settings):
        import torch

        torch.set_num_threads(4)
        settings = dict(settings)
        mode = settings.get("mode", "Numbers")
        model = settings["model"]
        checkpoint, digest = self.resolve(mode, model)
        if mode == "Alphanumeric":
            from hnrs_chars.predict import CharacterPredictor
            from hnrs_chars.segmentation import recognise_text

            settings["representation"] = "otsu"
            settings["preparation"] = (
                "Paarth character preparation: local illumination, Otsu, speck removal, ink normalisation"
            )
            factory = CharacterPredictor
        else:
            from hnrs_ml.predict import DigitPredictor

            factory = DigitPredictor
        key = (mode, model, digest)
        if key not in self.predictors:
            self.predictors[key] = factory(checkpoint)
        settings["mode"] = mode
        settings["checkpoint_sha256"] = digest
        settings["checkpoint"] = (
            checkpoint.relative_to(self.root).as_posix()
            if checkpoint.is_relative_to(self.root)
            else str(checkpoint)
        )
        if mode == "Alphanumeric":
            settings["line_case"] = bool(settings.get("line_case", False))
            result = recognise_text(
                image, self.predictors[key], settings["method"], settings["line_case"]
            )
        else:
            from segment_digits import recognise_number

            result = recognise_number(
                image,
                self.predictors[key],
                settings["method"],
                settings["representation"],
            )
        return result, settings


# Handles GUI image input, number generation and evidence export
def open_photo(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def digit_catalog(folder):
    """Index single-digit labels only; never use number photos as digit tiles."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError("Select a folder containing individual digit images.")
    catalog = {str(i): [] for i in range(10)}
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_TYPES:
            continue
        match = re.search(r"(?:^|_label-)([0-9])$", path.stem)
        if match:
            catalog[match[1]].append(path)
    return catalog


def generate_number(folder, text, seed=42):
    """Assemble actual handwriting samples. Requested text is NOT a prediction."""
    if not re.fullmatch(r"[0-9]{1,12}", text):
        raise ValueError("Enter 1 to 12 digits (0-9). Leading zeros are allowed.")
    catalog = digit_catalog(folder)
    missing = sorted(set(text) - {digit for digit, paths in catalog.items() if paths})
    if missing:
        raise ValueError(
            "No single-digit image for: "
            + ", ".join(missing)
            + ". Use filenames such as Name_01_label-7.jpg or 7.png."
        )
    rng = random.Random(seed)
    canvas = Image.new(
        "RGB", (32 + len(text) * 112 + (len(text) - 1) * 16, 144), "white"
    )
    sources = []
    warnings = []
    for i, digit in enumerate(text):
        path = rng.choice(catalog[digit])
        try:
            variants, _, warning = prepare_variants(open_photo(path))
        except (ValueError, OSError) as exc:
            raise ValueError(f"Cannot prepare {path.name}: {exc}") from exc
        tile = Image.fromarray(
            255 - np.rint(variants["otsu"] * 255).astype("uint8")
        ).resize((112, 112), Image.Resampling.NEAREST)
        canvas.paste(tile, (16 + i * 128, 16))
        sources.append(
            {
                "digit": digit,
                "file": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        if warning:
            warnings.append(f"{path.name}: {warning}")
    return canvas, {
        "route": "generated",
        "requested_text": text,
        "seed": seed,
        "sources": sources,
        "warnings": warnings,
        "note": "Composed from labelled handwriting crops; requested text was not supplied to the recogniser.",
    }


def annotated_image(segmented):
    image = segmented.gray.convert("RGB")
    draw = ImageDraw.Draw(image)
    for i, (left, top, right, bottom) in enumerate(segmented.boxes, 1):
        draw.rectangle((left, top, right - 1, bottom - 1), outline="#d32828", width=2)
        draw.text((left, max(0, top - 14)), str(i), fill="#d32828")
    return image


def export_result(parent, image, origin, result, settings):
    """Create a fresh evidence directory; never replace previous experiments."""
    destination = Path(parent) / (
        "gui_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    )
    destination.mkdir(parents=True, exist_ok=False)
    image.save(destination / "input.png")
    seg = result["segmentation"]
    arrays = result["prepared"]
    annotated_image(seg).save(
        destination
        / (
            "detected_characters.png"
            if settings.get("mode") == "Alphanumeric"
            else "detected_digits.png"
        )
    )
    Image.fromarray(seg.mask.astype("uint8") * 255).save(
        destination / "foreground_mask.png"
    )
    np.save(destination / "prepared_crops.npy", arrays)
    for index, (crop, array) in enumerate(zip(seg.crops, arrays)):
        crop.save(destination / f"crop_{index:02d}.png")
        Image.fromarray(np.rint(array * 255).astype("uint8")).save(
            destination / f"prepared_{index:02d}.png"
        )
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": origin,
        "settings": settings,
        "prediction": result["prediction"],
        "boxes": seg.boxes,
        "coordinate_space": "resized grayscale image",
        "resized_size": seg.gray.size,
        "warnings": seg.warnings,
        "input_png_sha256": hashlib.sha256(
            (destination / "input.png").read_bytes()
        ).hexdigest(),
        "component_groups": getattr(seg, "groups", None),
        "score_note": "Per-character confidence is an uncalibrated model score, not a verified probability.",
    }
    (destination / "result.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    return destination
