"""Line-geometry letter case: decisions, limits and integration."""
import json
from pathlib import Path
import unittest

import numpy as np
from PIL import Image, ImageDraw

from hnrs_chars.case_rule import THRESHOLD, adjust_prediction, apply_line_case, line_reference
from hnrs_chars.segmentation import recognise_text

ROOT = Path(__file__).resolve().parents[1]
TALL = (0, 100, 40, 200)  # Cap line at y=100, baseline at y=200.


def box(x, top, bottom=200):
    return (x, top, x + 40, bottom)


class CaseRuleTests(unittest.TestCase):
    def test_short_letters_become_lowercase_and_tall_become_uppercase(self):
        # R and 7 are references; the short shapes are o, o, m; the tall Z stays a capital.
        boxes = [box(0, 100), box(50, 150), box(100, 152), box(150, 148), box(200, 100), box(250, 102)]
        text, details = apply_line_case("R00M7z", boxes)
        self.assertEqual(text, "Room7Z")
        self.assertEqual([c["final"] for c in details["changes"]], ["o", "o", "m", "Z"])
        self.assertEqual(details["reference_characters"], 2)

    def test_tall_zero_and_capital_o_keep_the_classifier_choice(self):
        boxes = [box(0, 100), box(50, 98), box(100, 104)]
        self.assertEqual(apply_line_case("70O", boxes)[0], "70O")
        # A tall shape read as lowercase o must be a capital.
        self.assertEqual(apply_line_case("7o", boxes[:2])[0], "7O")

    def test_without_reference_characters_text_is_unchanged(self):
        for text, boxes in [("S", [box(0, 150)]), ("sxz", [box(0, 150), box(50, 100), box(100, 150)]),
                            ("00", [box(0, 100), box(50, 150)])]:
            result, details = apply_line_case(text, boxes)
            self.assertEqual(result, text)
            self.assertFalse(details["applied"])

    def test_unambiguous_letters_and_descenders_are_never_changed(self):
        boxes = [box(0, 100), box(50, 150), box(100, 150, 240), box(150, 150, 240)]
        self.assertEqual(apply_line_case("Aepy", boxes)[0], "Aepy")
        self.assertEqual(apply_line_case("APY4", boxes)[0], "APY4")

    def test_reference_uses_median_and_rejects_degenerate_boxes(self):
        line = line_reference([box(0, 90), box(50, 100), box(100, 400)], "AB1")
        self.assertEqual((line["top"], line["bottom"], line["count"]), (100, 200, 3))
        self.assertIsNone(line_reference([(0, 5, 10, 5)], "A"))
        with self.assertRaises(ValueError):
            apply_line_case("AB", [box(0, 100)])

    def test_adjust_prediction_keeps_the_classifier_output(self):
        prediction = {"text": "S0", "characters": [{"character": "S"}, {"character": "0"}]}
        boxes = [box(0, 100), box(50, 150)]
        # S is itself ambiguous, so no reference exists and nothing changes.
        adjust_prediction(prediction, boxes)
        self.assertEqual((prediction["text"], prediction["model_text"]), ("S0", "S0"))
        prediction = {"text": "B0", "characters": [{"character": "B"}, {"character": "0"}]}
        adjust_prediction(prediction, boxes)
        self.assertEqual((prediction["text"], prediction["model_text"]), ("Bo", "B0"))
        self.assertEqual(prediction["characters"][1], {"character": "o", "model_character": "0"})

    def test_recognise_text_applies_rule_only_when_requested(self):
        image = Image.new("L", (360, 220), 255)
        draw = ImageDraw.Draw(image)
        draw.line([(40, 30), (40, 190)], fill=0, width=9)            # Tall stroke: reference.
        draw.ellipse((120, 105, 190, 190), outline=0, width=9)       # Short round shape.

        class Fixed:
            def predict_text(self, crops):
                return {"text": "10", "characters": [{"character": "1", "confidence": 1.0},
                                                     {"character": "0", "confidence": 1.0}]}
        self.assertEqual(recognise_text(image, Fixed())["prediction"]["text"], "10")
        result = recognise_text(image, Fixed(), line_case=True)
        self.assertEqual((result["prediction"]["text"], result["prediction"]["model_text"]), ("1o", "10"))

    def test_shipped_threshold_matches_recorded_study(self):
        path = ROOT / "results/case_rule/summary.json"
        if not path.exists():
            self.skipTest("Run python -m research.study_case_rule to record the study")
        record = json.loads(path.read_text())
        self.assertAlmostEqual(record["fitted_threshold"], THRESHOLD, delta=0.005)
        self.assertEqual(record["training_errors"], 0)
        self.assertTrue(record["adoption_gate"]["passed"])
        self.assertLess(record["uppercase_or_zero_scores"]["maximum"], THRESHOLD)
        self.assertGreater(record["lowercase_scores"]["minimum"], THRESHOLD)


if __name__ == "__main__":
    unittest.main()
