"""Choose letter case from line geometry for one line of separated characters.

Some lowercase letters are scaled copies of their capitals (c/C, o/O, s/S ...).
After segmentation each crop is resized to 28x28, so the classifier cannot see
how tall the letter was. In a line of text the writer's own capitals and
digits supply that missing reference:

* reference characters (digits 1-9, unambiguous capitals and lowercase
  ascenders such as b, d, h) give the cap line and the baseline;
* a case-ambiguous letter that starts well below the cap line and is short
  relative to the reference height is lowercase, otherwise uppercase;
* a predicted 0/O/o that is short becomes lowercase o.

The score combines both cues: score = offset + (1 - height), with offset and
height measured in units of the reference height. A larger score means a
lower, shorter character. THRESHOLD is the fitted 0.316 rounded to two
places; it was fitted on the 40 development string photos only (see
research/study_case_rule.py and docs/reports/case-rule-study.md).

Limits: needs at least one reference character, so single-character photos
are unchanged. Generated GUI images are unsuitable because every tile is
resized to the same height. Descender letters (p, y) are excluded because the
development photos contain none to fit a rule on, and a writer who forms
small capitals will be read as lowercase.
"""
import statistics

CASE_PAIRS = frozenset("cmosuvwxz")
ZERO_LIKE = frozenset("0Oo")
REFERENCE = frozenset("123456789ABDEFGHIKLNRTbdhklt")
THRESHOLD = 0.32


def line_reference(boxes, text):
    """Median cap line, baseline and height from reference characters, or None."""
    references = [box for box, character in zip(boxes, text) if character in REFERENCE]
    if not references:
        return None
    top = statistics.median(box[1] for box in references)
    bottom = statistics.median(box[3] for box in references)
    if bottom <= top:
        return None
    return {"top": top, "bottom": bottom, "height": bottom - top, "count": len(references)}


def lowercase_score(box, reference):
    """Offset below the cap line plus shortfall in height, in reference units."""
    offset = (box[1] - reference["top"]) / reference["height"]
    height = (box[3] - box[1]) / reference["height"]
    return offset + (1 - height)


def is_candidate(character):
    return character.lower() in CASE_PAIRS or character in ZERO_LIKE


def apply_line_case(text, boxes, threshold=THRESHOLD):
    """Return (adjusted text, details). Never uses reference labels.

    `boxes` are (left, top, right, bottom) segmentation boxes in reading order,
    one per predicted character.
    """
    if len(text) != len(boxes):
        raise ValueError("Need exactly one box per predicted character")
    reference = line_reference(boxes, text)
    details = {"applied": reference is not None, "threshold": threshold,
               "reference_characters": reference["count"] if reference else 0, "changes": []}
    if reference is None:
        return text, details
    adjusted = list(text)
    for index, (character, box) in enumerate(zip(text, boxes)):
        if not is_candidate(character):
            continue
        score = lowercase_score(box, reference)
        lowercase = score >= threshold
        if character in ZERO_LIKE:
            # A short round shape is lowercase o; a tall one keeps the model's
            # choice between 0 and O, except that a tall 'o' must be a capital.
            new = "o" if lowercase else ("O" if character == "o" else character)
        else:
            new = character.lower() if lowercase else character.upper()
        if new != character:
            adjusted[index] = new
            details["changes"].append({"index": index, "model": character, "final": new,
                                       "score": round(score, 3)})
    return "".join(adjusted), details


def adjust_prediction(prediction, boxes, threshold=THRESHOLD):
    """Apply the rule to a CharacterPredictor.predict_text result in place."""
    text, details = apply_line_case(prediction["text"], boxes, threshold)
    prediction["model_text"] = prediction["text"]
    prediction["text"] = text
    prediction["case_rule"] = details
    for change in details["changes"]:
        item = prediction["characters"][change["index"]]
        item["model_character"] = item["character"]
        item["character"] = change["final"]
    return prediction
