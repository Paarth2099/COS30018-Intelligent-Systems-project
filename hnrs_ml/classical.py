"""Classical (non-neural) digit classifiers for comparison with the networks.

Features are raw pixels or a histogram of oriented gradients (HOG). Inputs
follow the shared contract: float32 [N,28,28] or [N,1,28,28] in [0,1], light
ink on black. scikit-learn is a research-only dependency; the GUI does not
need it to recognise photographs.
"""
import numpy as np

FEATURES = ("pixels", "hog")


def pixel_features(images):
    """Flatten each 28x28 crop into a 784-value vector."""
    return np.asarray(images, dtype=np.float32).reshape(len(images), -1)


def hog_features(images, cell=4, bins=9, block=2, chunk=4096):
    """Histogram of oriented gradients after Dalal and Triggs (2005).

    Centred gradients, unsigned orientation (0-180 degrees) with linear
    interpolation between neighbouring bins, 4x4-pixel cells, overlapping
    2x2-cell blocks and L2-Hys block normalisation. A 28x28 crop gives
    6x6 blocks x 4 cells x 9 bins = 1,296 features.
    """
    x = np.asarray(images, dtype=np.float32).reshape(len(images), 28, 28)
    if 28 % cell:
        raise ValueError("Cell size must divide 28")
    cells = 28 // cell
    blocks = cells - block + 1
    output = np.empty((len(x), blocks * blocks * block * block * bins), np.float32)
    # Process in chunks so the per-pixel orientation array stays small.
    for start in range(0, len(x), chunk):
        part = x[start:start + chunk]
        gx = np.zeros_like(part)
        gy = np.zeros_like(part)
        gx[:, :, 1:-1] = part[:, :, 2:] - part[:, :, :-2]
        gy[:, 1:-1, :] = part[:, 2:, :] - part[:, :-2, :]
        magnitude = np.hypot(gx, gy)
        position = (np.rad2deg(np.arctan2(gy, gx)) % 180) / (180 / bins) - 0.5
        lower = np.floor(position)
        upper_weight = position - lower
        lower = lower.astype(np.int64) % bins
        upper = (lower + 1) % bins
        votes = np.zeros(part.shape + (bins,), np.float32)
        # The two neighbouring bins always differ, so the second write never
        # overwrites the first.
        np.put_along_axis(votes, lower[..., None], (magnitude * (1 - upper_weight))[..., None], axis=-1)
        np.put_along_axis(votes, upper[..., None], (magnitude * upper_weight)[..., None], axis=-1)
        histogram = votes.reshape(len(part), cells, cell, cells, cell, bins).sum(axis=(2, 4))
        grouped = np.stack([histogram[:, y:y + block, x0:x0 + block].reshape(len(part), -1)
                            for y in range(blocks) for x0 in range(blocks)], axis=1)
        # L2-Hys: normalise, clip large components at 0.2, then renormalise.
        grouped /= np.sqrt((grouped ** 2).sum(-1, keepdims=True) + 1e-6)
        np.minimum(grouped, 0.2, out=grouped)
        grouped /= np.sqrt((grouped ** 2).sum(-1, keepdims=True) + 1e-6)
        output[start:start + len(part)] = grouped.reshape(len(part), -1)
    return output


def features(images, kind):
    if kind == "pixels":
        return pixel_features(images)
    if kind == "hog":
        return hog_features(images)
    raise ValueError(f"Unknown feature type: {kind}")


# Each family is one classical approach; the listed settings form a small
# validation grid. Names are stable identifiers used in result files.
FAMILIES = {
    "knn_pixels": {"features": "pixels", "grid": [{"n_neighbors": k} for k in (1, 3, 5)],
                   "description": "k-nearest neighbours, raw pixels, distance-weighted votes"},
    "knn_hog": {"features": "hog", "grid": [{"n_neighbors": k} for k in (1, 3, 5)],
                "description": "k-nearest neighbours, HOG features, distance-weighted votes"},
    "linear_svm_hog": {"features": "hog", "grid": [{"C": c} for c in (0.1, 1.0)],
                       "description": "Linear support vector machine (one-vs-rest), HOG features"},
    "rbf_svm_hog": {"features": "hog", "grid": [{"C": c} for c in (1.0, 10.0)],
                    "description": "RBF-kernel support vector machine (one-vs-one), HOG features"},
    "random_forest_pixels": {"features": "pixels", "grid": [{"n_estimators": 200}],
                             "description": "Random forest of 200 trees, raw pixels"},
}


def build_classifier(family, params, seed=42, jobs=-1):
    """Return an unfitted scikit-learn estimator for one family/setting."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC, LinearSVC
    if family.startswith("knn_"):
        return KNeighborsClassifier(weights="distance", algorithm="brute", n_jobs=jobs, **params)
    if family == "linear_svm_hog":
        return LinearSVC(dual="auto", max_iter=5000, random_state=seed, **params)
    if family == "rbf_svm_hog":
        return SVC(kernel="rbf", gamma="scale", cache_size=2000, random_state=seed, **params)
    if family == "random_forest_pixels":
        return RandomForestClassifier(n_jobs=jobs, random_state=seed, **params)
    raise ValueError(f"Unknown classical family: {family}")
