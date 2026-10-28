"""Audit saved string evidence and replay the selected public pipeline."""

import csv
import json
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from research.compare_segmentation import edit_distance
from .data import sha256
from .segmentation import DEFAULT_METHOD, recognise_text
from .string_experiment import load_predictor


def rows(path):
    with path.open(encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main():
    base = Path("evidence/character_segmentation")
    inputs = Path("data/own_strings/development")
    manifests = []
    for number in range(1, 5):
        run = base / f"run_{number:02d}"
        manifest = rows(run / "manifest.csv")
        manifests.append(manifest)
        assert len(manifest) == 40 and sum(len(r["reference"]) for r in manifest) == 160
        assert len({r["file"] for r in manifest}) == 40
        for item in manifest:
            assert sha256(inputs / item["file"]) == item["sha256"]
        summary = json.loads((run / "summary.json").read_text())
        assert sha256(Path(summary["checkpoint"])) == summary["checkpoint_sha256"]
        for name, digest in summary["source_sha256"].items():
            assert sha256(run / "source" / name) == digest, (run, name)
        results = rows(run / "results.csv")
        assert (
            len(results) == 80
            and len({(r["file"], r["method"]) for r in results}) == 80
        )
        for method, metrics in summary["methods"].items():
            subset = [r for r in results if r["method"] == method]
            assert len(subset) == 40
            for r in subset:
                assert not r["error"]
                assert int(r["exact_correct"]) == int(r["reference"] == r["prediction"])
                assert int(r["count_correct"]) == int(
                    len(r["reference"]) == int(r["detected_count"])
                )
                assert int(r["edit_distance"]) == edit_distance(
                    r["reference"], r["prediction"]
                )
            assert metrics["exact_correct"] == sum(
                int(r["exact_correct"]) for r in subset
            )
            assert metrics["count_correct"] == sum(
                int(r["count_correct"]) for r in subset
            )
            assert metrics["edit_distance"] == sum(
                int(r["edit_distance"]) for r in subset
            )
            assert metrics["character_error_rate"] == metrics["edit_distance"] / 160
    assert all(m == manifests[0] for m in manifests)
    final = base / "run_04"
    summary = json.loads((final / "summary.json").read_text())
    for name in (
        "hnrs_chars/segmentation.py",
        "hnrs_chars/preprocessing.py",
        "hnrs_chars/predict.py",
    ):
        assert sha256(Path(name)) == summary["source_sha256"][name]
    decision = json.loads((base / "selection.json").read_text())
    assert decision["selected_method"] == DEFAULT_METHOD
    assert decision["summary_sha256"] == sha256(final / "summary.json")
    assert decision["review_sha256"] == sha256(final / "visual_review.csv")
    review = rows(final / "visual_review.csv")
    assert len(review) == 80 and all(
        r["visual_status"] in ("pass", "fail", "uncertain") for r in review
    )
    for method, counts in decision["visual_review"].items():
        for status, count in counts.items():
            assert count == sum(
                r["method"] == method and r["visual_status"] == status for r in review
            )
    torch.set_num_threads(4)
    predictor, _ = load_predictor(Path("experiments/emnist_byclass/run_01"))
    for item in manifests[-1]:
        with Image.open(inputs / item["file"]) as photo:
            result = recognise_text(photo, predictor)
        saved = final / item["writer"] / Path(item["file"]).stem / DEFAULT_METHOD
        geometry = json.loads((saved / "segmentation.json").read_text())
        assert [list(b) for b in result["segmentation"].boxes] == geometry["boxes"]
        assert result["segmentation"].groups == geometry["component_groups"]
        assert np.array_equal(result["prepared"], np.load(saved / "prepared.npy"))
        expected = json.loads((saved / "prediction.json").read_text())
        assert result["prediction"]["text"] == expected["text"]
    demo = json.loads((base / "demo_verified/demo.json").read_text())
    assert demo["input_sha256"] == sha256(Path(demo["input"]))
    saved = final / "Manula" / "Manula_01_text-Ab12" / DEFAULT_METHOD
    assert (
        demo["prediction"]["text"]
        == json.loads((saved / "prediction.json").read_text())["text"]
    )
    assert np.array_equal(
        np.load(base / "demo_verified/prepared.npy"), np.load(saved / "prepared.npy")
    )
    unchanged = [
        "app.py",
        "gui_core.py",
        "preprocess_digits.py",
        "segment_digits.py",
        "hnrs_ml",
        "experiments",
        "hnrs_chars/preprocessing.py",
        "hnrs_chars/predict.py",
        "hnrs_chars/models.py",
    ]
    assert not subprocess.check_output(
        ["git", "diff", "d6690e5c6672f064d6c9b77b1f6a56e0760a64be", "--", *unchanged]
    )
    test_log = (base / "unit_tests.txt").read_text(encoding="utf-8-sig")
    assert "Ran 47 tests" in test_log and test_log.strip().endswith("OK")
    subprocess.run(["git", "diff", "--check"], check=True)
    report = {
        "status": "PASS",
        "runs_audited": 4,
        "input_photos": 40,
        "method_image_results_checked": 320,
        "selected_pipeline_photos_replayed": 40,
        "prepared_arrays_and_boxes_match": True,
        "predicted_strings_match": True,
        "demo_verified": True,
        "unit_tests_passed": 47,
        "previous_components_unchanged": True,
        "verification_source_sha256": sha256(Path(__file__)),
        "limits": "Evidence consistency and replay checks; visual review remains assistant assessment; no unseen final test",
    }
    (base / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
