# Letter case from line context (extension improvement)

**Problem.** In the frozen final evaluation, 7 of the 16 failed strings and 5 of the 11 failed letters differed from the truth only in case (for example `XyZ2` read as `XYZ2`, `vWx9` as `VWX9`). Several letters are drawn the same way in both cases at different sizes: c, m, o, s, u, v, w, x, z. Lowercase o and zero differ in the same way. The pipeline loses that size information on purpose: every crop is scaled to 28x28 before classification, so the classifier cannot tell a small `o` from a large `O` or `0`.

**Idea.** In a line of text, the writer's own unambiguous characters give the missing scale. Digits 1–9, unambiguous capitals and ascender letters such as `b`, `d` and `h` mark the cap line and baseline. A case-ambiguous letter that starts well below the cap line and is short compared with them is lowercase.

## Rule (`hnrs_chars/case_rule.py`)

1. **Reference line.** Take the characters the classifier read as `1-9 A B D E F G H I K L N R T b d h k l t`. Use the median of their box tops as the cap line and the median of their box bottoms as the baseline, which gives a reference height *H*. Medians resist one badly segmented box.
2. **Score** each character read as one of `c m o s u v w x z` (either case) or `0 O o`:
   score = (top − cap line)/H + (1 − box height/H).
   Both terms grow for a character that is lower and shorter.
3. **Decide.** If score ≥ 0.32, the character becomes lowercase (a round shape becomes `o`). Otherwise it becomes uppercase, and a tall round shape keeps the classifier's choice between `0` and `O`.
4. **Abstain.** With no reference character (single-character photos, or strings such as `sxz`), or for any other class, the classifier's output is kept. `p` and `y` are excluded: their lowercase forms descend below the baseline, and the development photos contain no examples to fit them on.

The rule never sees the reference text. It runs after classification and keeps the classifier's own string as `model_text` in results and exports. It is not applied to GUI-generated images, because generation resizes every tile to the same height.

## Fitting the threshold, on development data only

`research/study_case_rule.py` segments and classifies the 40 development string photos with the shipped models. It then scores every true case-ambiguous letter found where segmentation produced the right number of characters (35 letters: 21 lowercase, 14 capitals or zeros).

| Group | n | Score range |
|---|---:|---|
| Capitals and zeros (O, Z, 0) | 14 | −0.45 to 0.27 |
| Lowercase (o, m, z) | 21 | 0.36 to 1.03 |

The two groups do not overlap on development data. The fitted threshold is the midpoint of the gap, 0.316, shipped rounded as **0.32**. The study refuses to run if the shipped value drifts from the fit.

## Leave-one-writer-out evaluation (selected ResNet-18)

Fitting and evaluating on the same 40 photos would be optimistic. Instead, each writer's 10 strings were scored with a threshold fitted only on the other three writers:

| Held-out writer | Threshold fitted without them | Exact before → after | CER before → after | Gained / lost |
|---|---:|---:|---:|---:|
| Haresh | 0.241 | 6 → 7 | 17.5% → 10.0% | 2 / 1 |
| Manula | 0.316 | 2 → 4 | 30.0% → 20.0% | 2 / 0 |
| Paarth | 0.331 | 2 → 5 | 30.0% → 17.5% | 3 / 0 |
| Saniru | 0.316 | 2 → 4 | 30.0% → 17.5% | 2 / 0 |
| **All 40 strings** | | **12 → 20** | **26.9% → 16.3%** | **9 / 1** |

The single loss shows the rule's limit honestly. Without Haresh's data, the fitted threshold falls to 0.24, and Haresh's slightly short second zero in `007Ab` (score 0.27) becomes `o`. The gains are `Room7` and `Hello` for every writer (`R00M7` → `Room7`, `Hell0` → `Hello`) and `ZZ25` → `Zz25`.

**Adoption gate** (the project's existing rule: no fall in exact matches or rise in CER on either development collection, plus a strict improvement): **passed**. Development single characters are unchanged at 52/80, because the rule abstains without a reference line. Strings improve from 12/40 to 21/40 with the all-writer threshold.

## Effect on every model (all-writer threshold 0.32)

| Model | Dev strings exact | Dev strings CER | Final strings exact* | Final strings CER* |
|---|---:|---:|---:|---:|
| mlp | 3 → 3 | 46.9% → 39.4% | 2 → 3 | 42.5% → 38.8% |
| cnn | 8 → 16 | 33.1% → 20.0% | 6 → 7 | 25.0% → 21.3% |
| **resnet18 (default)** | **12 → 21** | **26.9% → 15.6%** | **4 → 6** | **27.5% → 25.0%** |
| mobilenet_v3_small | 4 → 6 | 48.8% → 36.9% | 6 → 9 | 31.3% → 31.3% |

Single-letter photos (80 development, 40 final) are unchanged for every model.

\*The 120-photo final collection was analysed before this rule was designed, and its case errors motivated it. These columns are **post-hoc diagnostics**, not untouched test evidence. For the default ResNet-18 the rule fixes `uVw3`, `vWx9` and `YzA4` and breaks `AbC2` → `Abc2`, because Paarth wrote that capital C small, at x-height. Two further changes (`O0l1`, `StU5`) alter characters in strings that were already wrong. A truly independent estimate needs the planned fresh v2 collection, which should now be evaluated with the rule included.

## Discussion

- The improvement comes from information the classifier cannot see, not from tuning the classifier. That is why the rule helps all four models.
- It turns the 62-class problem into "shape from the classifier, size from the line". This is a simple form of the contextual post-processing used in practical OCR systems, where line metrics (x-height, cap height, baseline) are standard features.
- **Failure modes:** small capitals (the `AbC2` case); lines with no reference character; strongly slanted or curved baselines; segmentation errors that distort a box; and descender letters (`p`, `y`), which remain unhandled.
- **Possible extensions:** fit separate rules for descenders once data exists; use the x-height of unambiguous lowercase letters (a, e, n, r) as a second reference; or combine the score with the classifier's probabilities for both cases instead of overriding only the top-1 class.

## Use and reproduce

In the GUI (Alphanumeric mode, loaded photo), **Letter case from line context** is on by default. The result line reports how many characters changed and the classifier's original reading. Each crop label shows `final←classifier`, for example `o←0`. Untick it to see the raw classifier output.

```powershell
.\.venv\Scripts\python.exe -m research.study_case_rule   # about 2 minutes on CPU
```

Outputs in [`results/case_rule/`](../../results/case_rule/): `summary.json` (protocol, fit, leave-one-writer-out folds, gate), `summary.csv`, `predictions.csv` (every photo and model, before/after) and `development_scores.csv` (the 35 fitted scores).
