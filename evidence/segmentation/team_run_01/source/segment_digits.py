"""Manula's segmentation component for one line of separated handwritten digits.

Two methods share grayscale/autocontrast/Otsu and small-component filtering:
components = 8-connected ink regions; projection = occupied column intervals.
No labels, expected digit counts or model predictions influence segmentation.
Uses existing NumPy/Pillow dependencies, not OpenCV.
"""
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageOps
from preprocess_digits import otsu_threshold, prepare_variants

METHODS = ("components", "projection")


@dataclass
class SegmentationResult:
    gray: Image.Image
    mask: np.ndarray
    boxes: list
    crops: list
    warnings: list
    removed_components: int


def occupied_intervals(values):
    """Return half-open intervals of consecutive True values."""
    padded = np.pad(np.asarray(values, dtype=np.int8), (1, 1))
    changes = np.diff(padded)
    return list(zip(np.flatnonzero(changes == 1).tolist(),
                    np.flatnonzero(changes == -1).tolist()))


def connected_regions(mask):
    """8-connected component labels using row runs and union-find.

    Returns (label image, [box/area/id dictionaries]). Diagonal neighbours
    connect. Bounding boxes use Pillow's exclusive right/bottom coordinates.
    """
    if mask.ndim != 2:
        raise ValueError("Expected a two-dimensional foreground mask")
    parents, runs, previous = [], [], []

    def find(label):
        while parents[label] != label:
            parents[label] = parents[parents[label]]
            label = parents[label]
        return label

    for y, row in enumerate(mask):
        current = []
        cursor = 0
        for start, end in occupied_intervals(row):
            label = len(parents)
            parents.append(label)
            # For 8-connectivity, adjacent row runs may meet diagonally.
            while cursor < len(previous) and previous[cursor][1] < start:
                cursor += 1
            k = cursor
            while k < len(previous) and previous[k][0] <= end:
                parents[find(previous[k][2])] = find(label)
                k += 1
            current.append((start, end, label))
            runs.append((y, start, end, label))
        previous = current
    labels = np.zeros(mask.shape, dtype=np.int32)
    groups = {}
    for y, start, end, label in runs:
        root = find(label)
        if root not in groups:
            groups[root] = {"id": len(groups)+1, "box": [start, y, end, y+1], "area": 0}
        item = groups[root]
        box = item["box"]
        box[:] = [min(box[0],start), min(box[1],y), max(box[2],end), max(box[3],y+1)]
        item["area"] += end-start
        labels[y,start:end] = item["id"]
    return labels, list(groups.values())


def segment_image(image, method="projection", max_side=1000,
                  min_area_fraction=0.01, min_height_fraction=0.10):
    """Find ordered crops; boxes refer to returned gray, not original pixels.

    Supports dark ink on pale paper and one horizontal line. Touching digits,
    overlapping columns, severe shadows and multi-line pages are limitations.
    Filtering thresholds are relative to the largest area/tallest component;
    they are fixed across photos, not chosen using expected labels.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    if max_side < 28 or not 0 <= min_area_fraction <= 1 or not 0 <= min_height_fraction <= 1:
        raise ValueError("Invalid segmentation settings")
    gray = ImageOps.exif_transpose(image).convert("L")
    gray.thumbnail((max_side,max_side), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray)
    warnings = []
    if int(pixels.max()) - int(pixels.min()) < 10:
        return SegmentationResult(gray,np.zeros(pixels.shape,dtype=bool),[],[],["Blank or very low contrast image"],0)
    adjusted = np.asarray(ImageOps.autocontrast(gray))
    mask = adjusted <= otsu_threshold(adjusted)
    labels, regions = connected_regions(mask)
    if not regions:
        return SegmentationResult(gray,mask,[],[],["No ink found"],0)
    minimum_area = max(3, max(r['area'] for r in regions)*min_area_fraction)
    minimum_height = max(2, max(r['box'][3]-r['box'][1] for r in regions)*min_height_fraction)
    kept = [r for r in regions if r['area'] >= minimum_area and r['box'][3]-r['box'][1] >= minimum_height]
    clean = np.isin(labels,[r['id'] for r in kept])
    if clean[0].any() or clean[-1].any() or clean[:,0].any() or clean[:,-1].any():
        warnings.append("Ink touches image edge; inspect for clipping or background artifacts")
    if clean.mean() > 0.45:
        warnings.append("Large foreground fraction; shadow or polarity may be wrong")
    if method == "components":
        boxes = [tuple(r['box']) for r in kept]
    else:
        boxes = []
        for left,right in occupied_intervals(clean.any(axis=0)):
            ys = np.flatnonzero(clean[:,left:right].any(axis=1))
            boxes.append((left,int(ys[0]),right,int(ys[-1])+1))
    boxes.sort(key=lambda b:(b[0],b[1]))
    # Retain a little real background where available for Paarth's preparation.
    crops=[]
    for left,top,right,bottom in boxes:
        pad=max(2,round(.03*(bottom-top)))
        box=(max(0,left-pad),max(0,top-pad),min(gray.width,right+pad),min(gray.height,bottom+pad))
        crops.append(gray.crop(box))
    return SegmentationResult(gray,clean,boxes,crops,warnings,len(regions)-len(kept))


def prepare_crops(crops, representation="otsu"):
    """Reuse Paarth's unchanged code, preserving crop order for Haresh."""
    if representation not in ("grayscale","fixed","otsu"):
        raise ValueError("Unknown preprocessing representation")
    prepared = [prepare_variants(crop)[0][representation] for crop in crops]
    return np.stack(prepared) if prepared else np.empty((0,28,28),dtype=np.float32)


def recognise_number(image, predictor, method="projection", representation="otsu"):
    """GUI integration: load predictor once, call this per image. Keeps zeros."""
    segmented = segment_image(image,method=method)
    prepared = prepare_crops(segmented.crops,representation)
    return {"segmentation":segmented,"prepared":prepared,
            "prediction":predictor.predict_number(prepared)}
