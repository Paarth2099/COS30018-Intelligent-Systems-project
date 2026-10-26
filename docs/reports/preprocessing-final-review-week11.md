> Recorded study: results and original source hashes are retained. Detailed outputs are in the [research archive](../guides/research-archive.md); current commands use the `research` package.

# Week 11: Paarth final preprocessing error review

## Purpose and method

Paarth's preprocessing responsibility is to prepare useful 28x28 model inputs while preserving handwriting. This final-stage review checks where that preparation succeeds or visibly damages the image. It is an analysis of the frozen evaluation, not another improvement experiment.

All 120 original photos were compared visually with their saved selected-route crops and prepared arrays across 16 sheets. The three flagged cases were also inspected in enlarged individual panels. All 34 selected-system errors remain in the review alongside 86 correct cases. The review was performed by the assistant; Paarth should inspect the linked examples and understand the findings before presenting them as his own explanation.

No recognition, training, preprocessing rerun, new threshold search or changes to the frozen pipeline were performed. Panels use only saved evidence. The 0.1 intensity threshold below is an inspection statistic, not a new recognition setting.

## Selected preparation and why it was retained

Numbers use the existing digit route: grayscale/autocontrast and Otsu foreground localisation, aspect-preserving resizing into a padded 28x28 canvas, followed by the adopted peak contrast normalisation in `segment_digits.prepare_crops`. Alphanumeric inputs use local illumination correction, Otsu localisation, speck removal and peak-normalised character preparation, preserving dots when present in the crop. All models consume light ink on black.

These choices were fixed from the prior development investigations, not selected from the final photos. See [Week 10 preprocessing experiment](preprocessing-experiment-week10.md) and [final evaluation](final-evaluation-week11.md). Normalising characters independently can remove relative size/context clues that help people distinguish cases; that is a limitation of the representation, not proof of the cause of a particular error.

## Array checks

- 120 selected-route samples yielded 229 saved crops. The intended total is 228 characters; one split adds a crop.
- No empty prepared arrays, non-finite pixels or pixels outside [0,1]. All arrays have 28x28 spatial shape.
- No prepared ink above 0.1 touches the outer canvas edge. This does not rule out clipping or feature loss before padding.
- 119/120 crop counts match reference length. Count agreement alone is not evidence that each crop is complete or correctly grouped.
- The one count mismatch is `Paarth_numbers_03`: reference `462`, four crops, output `6162`.

## Visual findings

| Evidence category | Samples | Interpretation |
|---|---:|---|
| Correct prediction; no obvious preparation damage | 86 | Gross glyph structure is retained at inspection resolution; not a guarantee of perfect preprocessing. |
| Incorrect prediction with visible preparation damage | 2 | Haresh_strings_01 and Haresh_strings_04 contain badly distorted prepared inputs. |
| Incorrect prediction with an upstream split | 1 | Paarth_numbers_03 already contains a split character in the saved crops. |
| Incorrect prediction without obvious preparation damage | 31 | Prepared shapes remain recognisable; classification or handwriting ambiguity is plausible, but root cause is not established. |

These categories are descriptive visual judgments, not independent annotations or a measured preprocessing accuracy score. No causal claim that exactly 2/120 errors were caused solely by preprocessing is made.

### Complete crop but distorted preparation

`Haresh_strings_01`: the first saved crop contains U, but its prepared representation has a broad white mass and loses the U outline. The selected model outputs `PvW1` for `UvW1`.

[U preparation damage (archive: evidence/final_evaluation/preprocessing_review/Haresh_strings_01.png)](../guides/research-archive.md)

`Haresh_strings_04`: the source has a dark illumination gradient. The x input becomes an almost solid white block and Y becomes a distorted triangular mass. The selected model outputs `BDz4` for `xYz4`. This supports a preparation/background-handling limitation. It does not isolate whether illumination correction, threshold selection or their interaction is the sole cause.

[Shadow and background contamination (archive: evidence/final_evaluation/preprocessing_review/Haresh_strings_04.png)](../guides/research-archive.md)

### Split before preparation

`Paarth_numbers_03`: the 4 in 462 is already divided into two crop images before preparation. The prepared arrays preserve those fragments and the final output is 6162. This is a segmentation review case for Manula; it should not be counted as a demonstrated resize or padding defect.

[Four crops for three digits (archive: evidence/final_evaluation/preprocessing_review/Paarth_numbers_03.png)](../guides/research-archive.md)

### Wrong prediction with recognisable input

`Paarth_single_digits_10`: the loop and stem of 9 remain visible, yet CNN predicts 7. `Paarth_single_letters_09`: the dot and curved stem of i survive preparation, yet ResNet18 predicts 6. `Haresh_single_letters_08`: a preserved c is predicted C. These show why a wrong prediction alone does not prove preprocessing failed. The dot-survival example is not proof that every dot in the collection was grouped correctly.

## Every selected-system failure

| Sample | Reference | Output | Review category | Evidence |
|---|---|---|---|---|
| Haresh_single_digits_10 | `9` | `4` | incorrect_with_no_obvious_preparation_damage | [panel (archive: evidence/final_evaluation/preprocessing_review/Haresh_single_digits_10.png)](../guides/research-archive.md) |
| Haresh_single_letters_08 | `c` | `C` | incorrect_with_no_obvious_preparation_damage | [panel (archive: evidence/final_evaluation/preprocessing_review/Haresh_single_letters_08.png)](../guides/research-archive.md) |
| Haresh_strings_01 | `UvW1` | `PvW1` | visible_preparation_damage | [panel (archive: evidence/final_evaluation/preprocessing_review/Haresh_strings_01.png)](../guides/research-archive.md) |
| Haresh_strings_02 | `XyZ2` | `XYZ2` | incorrect_with_no_obvious_preparation_damage | [panel (archive: evidence/final_evaluation/preprocessing_review/Haresh_strings_02.png)](../guides/research-archive.md) |
| Haresh_strings_03 | `uVw3` | `UVw3` | incorrect_with_no_obvious_preparation_damage | [panel (archive: evidence/final_evaluation/preprocessing_review/Haresh_strings_03.png)](../guides/research-archive.md) |
| Haresh_strings_04 | `xYz4` | `BDz4` | visible_preparation_damage | [panel (archive: evidence/final_evaluation/preprocessing_review/Haresh_strings_04.png)](../guides/research-archive.md) |
| Paarth_single_digits_05 | `4` | `7` | incorrect_with_no_obvious_preparation_damage | [panel (archive: evidence/final_evaluation/preprocessing_review/Paarth_single_digits_05.png)](../guides/research-archive.md) |
| Paarth_single_digits_10 | `9` | `7` | incorrect_with_no_obvious_preparation_damage | [panel (archive: evidence/final_evaluation/preprocessing_review/Paarth_single_digits_10.png)](../guides/research-archive.md) |
| Paarth_numbers_02 | `815` | `803` | incorrect_with_no_obvious_preparation_damage | [panel (archive: evidence/final_evaluation/preprocessing_review/Paarth_numbers_02.png)](../guides/research-archive.md) |
| Paarth_numbers_03 | `462` | `6162` | upstream_segmentation_split | [panel (archive: evidence/final_evaluation/preprocessing_review/Paarth_numbers_03.png)](../guides/research-archive.md) |
| Paarth_numbers_04 | `9071` | `7071` | incorrect_with_no_obvious_preparation_damage | [panel (archive: evidence/final_evaluation/preprocessing_review/Paarth_numbers_04.png)](../guides/research-archive.md) |
| Paarth_numbers_05 | `2386` | `2385` | incorrect_with_no_obvious_preparation_damage | [panel (archive: evidence/final_evaluation/preprocessing_review/Paarth_numbers_05.png)](../guides/research-archive.md) |
| Paarth_single_letters_06 | `f` | `F` | incorrect_with_no_obvious_preparation_damage | [panel (archive: evidence/final_evaluation/preprocessing_review/Paarth_single_letters_06.png)](../guides/research-archive.md) |
| Paarth_single_letters_07 | `g` | `9` | incorrect_with_no_obvious_preparation_damage | [panel (archive: evidence/final_evaluation/preprocessing_review/Paarth_single_letters_07.png)](../guides/research-archive.md) |
| Paarth_single_letters_09 | `i` | `6` | incorrect_with_no_obvious_preparation_damage | [panel (archive: evidence/final_evaluation/preprocessing_review/Paarth_single_letters_09.png)](../guides/research-archive.md) |
| Paarth_strings_02 | `dEf7` | `dEF7` | incorrect_with_no_obvious_preparation_damage | [panel (archive: evidence/final_evaluation/preprocessing_review/Paarth_strings_02.png)](../guides/research-archive.md) |
| Paarth_strings_04 | `jKl8` | `jkl8` | incorrect_with_no_obvious_preparation_damage | [panel (archive: evidence/final_evaluation/preprocessing_review/Paarth_strings_04.png)](../guides/research-archive.md) |
| Paarth_strings_05 | `O0l1` | `00L1` | incorrect_with_no_obvious_preparation_damage | [panel (archive: evidence/final_evaluation/preprocessing_review/Paarth_strings_05.png)](../guides/research-archive.md) |
| Manula_single_letters_05 | `O` | `0` | incorrect_with_no_obvious_preparation_damage | [panel (archive: evidence/final_evaluation/preprocessing_review/Manula_single_letters_05.png)](../guides/research-archive.md) |
| Manula_single_letters_07 | `l` | `1` | incorrect_with_no_obvious_preparation_damage | [panel (archive: evidence/final_evaluation/preprocessing_review/Manula_single_letters_07.png)](../guides/research-archive.md) |
| Manula_single_letters_08 | `m` | `M` | incorrect_with_no_obvious_preparation_damage | [panel (archive: evidence/final_evaluation/preprocessing_review/Manula_single_letters_08.png)](../guides/research-archive.md) |
| Manula_single_letters_10 | `o` | `0` | incorrect_with_no_obvious_preparation_damage | [panel (archive: evidence/final_evaluation/preprocessing_review/Manula_single_letters_10.png)](../guides/research-archive.md) |
| Manula_strings_01 | `MnO3` | `Mn53` | incorrect_with_no_obvious_preparation_damage | [panel (archive: evidence/final_evaluation/preprocessing_review/Manula_strings_01.png)](../guides/research-archive.md) |
| Manula_strings_02 | `pQr6` | `PQrb` | incorrect_with_no_obvious_preparation_damage | [panel (archive: evidence/final_evaluation/preprocessing_review/Manula_strings_02.png)](../guides/research-archive.md) |
| Manula_strings_03 | `StU5` | `5tU5` | incorrect_with_no_obvious_preparation_damage | [panel (archive: evidence/final_evaluation/preprocessing_review/Manula_strings_03.png)](../guides/research-archive.md) |
| Manula_strings_04 | `vWx9` | `VWX9` | incorrect_with_no_obvious_preparation_damage | [panel (archive: evidence/final_evaluation/preprocessing_review/Manula_strings_04.png)](../guides/research-archive.md) |
| Manula_strings_05 | `Ji10` | `jil0` | incorrect_with_no_obvious_preparation_damage | [panel (archive: evidence/final_evaluation/preprocessing_review/Manula_strings_05.png)](../guides/research-archive.md) |
| Saniru_single_letters_06 | `p` | `P` | incorrect_with_no_obvious_preparation_damage | [panel (archive: evidence/final_evaluation/preprocessing_review/Saniru_single_letters_06.png)](../guides/research-archive.md) |
| Saniru_single_letters_08 | `r` | `T` | incorrect_with_no_obvious_preparation_damage | [panel (archive: evidence/final_evaluation/preprocessing_review/Saniru_single_letters_08.png)](../guides/research-archive.md) |
| Saniru_single_letters_09 | `s` | `S` | incorrect_with_no_obvious_preparation_damage | [panel (archive: evidence/final_evaluation/preprocessing_review/Saniru_single_letters_09.png)](../guides/research-archive.md) |
| Saniru_strings_01 | `YzA4` | `YZA4` | incorrect_with_no_obvious_preparation_damage | [panel (archive: evidence/final_evaluation/preprocessing_review/Saniru_strings_01.png)](../guides/research-archive.md) |
| Saniru_strings_02 | `bCd8` | `b6d8` | incorrect_with_no_obvious_preparation_damage | [panel (archive: evidence/final_evaluation/preprocessing_review/Saniru_strings_02.png)](../guides/research-archive.md) |
| Saniru_strings_03 | `EfG2` | `EF62` | incorrect_with_no_obvious_preparation_damage | [panel (archive: evidence/final_evaluation/preprocessing_review/Saniru_strings_03.png)](../guides/research-archive.md) |
| Saniru_strings_04 | `hIj6` | `hIJ6` | incorrect_with_no_obvious_preparation_damage | [panel (archive: evidence/final_evaluation/preprocessing_review/Saniru_strings_04.png)](../guides/research-archive.md) |

Of the 34 failed samples, 12 differ only in letter case. This is an output pattern, not a causal classification.

## Conclusions and future work

The preparation generally produces nonempty, padded inputs with recognisable strokes, but valid array shape and range do not guarantee useful handwriting content. Two shadow/background-related examples have visibly damaged prepared inputs even though all numerical checks pass. Conversely, many recognition errors retain recognisable input and should not be attributed to preparation without further evidence.

Retain the frozen final results. Future work could investigate illumination handling on new development examples, assess preservation of fine strokes and relative character size, and evaluate any repair on a new untouched collection. No such repair or improvement is claimed here. The final data are now used for error analysis and must not be repeatedly tuned against and then reported as a fresh test.

## Evidence and validation

- [Array audit and input/prepared hashes (archive: evidence/final_evaluation/preprocessing_review/array_audit.json)](../guides/research-archive.md)
- [All 120 visual-review records (archive: evidence/final_evaluation/preprocessing_review/visual_review.json)](../guides/research-archive.md)
- [Verification (archive: evidence/final_evaluation/preprocessing_review/verification.json)](../guides/research-archive.md)
- [Panel/audit generator](../../research/review_final_preprocessing.py)

The original frozen-run metrics remain in [Haresh's report](final-evaluation-week11.md). Its independent verifier was rerun after this review to confirm unchanged input/model/runtime-source hashes and unchanged score arithmetic. Review metadata and panels are new evidence, not alterations to run_01.

## Explanation for the tutor

“For my Week 11 preprocessing review, I examined the saved originals, crops and 28x28 inputs from the 120-photo final evaluation. I distinguished numerical validity from whether the handwriting remained useful. Two string photos had visible preparation distortion, while one number was already split at segmentation. Many other errors retained recognisable shapes, so I did not claim preprocessing caused every wrong answer. I documented the limitations without tuning on the final test set.”

Use that explanation after personally reviewing the linked examples; it describes the completed analysis and does not invent hours or independent manual work.
