> Recorded study: results and original source hashes are retained. Detailed outputs are in the [research archive](../guides/research-archive.md); current commands use the `research` package.

# Week 11: Manula final segmentation review

## Responsibility and method

Manula's segmentation component separates a number or string into ordered character images before Paarth's preparation and Haresh's classifier. It uses rule-based connected components/grouping and projection; it is not a separately trained neural network. The four neural architectures classify the resulting crops.

This review compares the saved selected method with projection for all 20 numbers and 20 mixed strings: 40 unique photos and 80 method-specific outputs. It examines annotated boxes and actual crop content, including whole glyphs, detached dots, ordering, splits, merges and clipping. All eight paired contact sheets were inspected, followed by enlarged views of all five flagged outputs. Reference labels are used for review only; they were not passed to the original segmentation or prediction functions.

The assistant performed this review and produced the evidence. Manula should inspect the examples before presenting the explanation. No recognition, segmentation rerun, training, tuning or change to the frozen default was performed.

## Results

| Collection | Method | Visually acceptable structural separation | Complete recognition | Character error rate |
|---|---|---:|---:|---:|
| numbers | components | 19/20 | 16/20 | 8.82% |
| numbers | projection | 19/20 | 16/20 | 8.82% |
| strings | grouped_components | 20/20 | 4/20 | 27.50% |
| strings | projection | 17/20 | 4/20 | 31.25% |

Structural separation means the crop set visually contains one complete glyph per expected position, with no observed extra/missing crop, wrong grouping or order swap. It does not require a correct classifier output or a perfectly clean background. These are descriptive assistant visual judgments, not independent human annotations, a pixel-level IoU score or a generalisation guarantee. Here the structural failures coincide with count mismatches, but every other case was also inspected rather than treating matching counts as proof.

Selected methods have one observed structural issue among 40 inputs; projection has four. Both have the same complete-recognition counts on each collection. For numbers projection fixes one image and breaks another. For strings projection has higher CER (31.25% versus 27.50%) despite the same 4/20 exact-match result.

## Five observed failures

| Input | Method | Finding | Saved output |
|---|---|---|---|
| `xYz4` (Haresh_strings_04) | projection | merge: Y and z occupy one crop in the shadowed image. Grouped components separates the four glyphs, although contaminated backgrounds remain a preparation concern. Dark background remains in one or more crops; structural separation is not the same as clean foreground extraction. See Paarth preparation review. | `Ba4` |
| `462` (Paarth_numbers_03) | components | split: The 4 is divided into an L-like left fragment and a vertical stroke; four crops represent three digits. Projection retains the whole 4 and recognises 462. | `6162` |
| `O0l1` (Paarth_strings_05) | projection | merge: The adjacent O and 0 occupy one crop. Grouped components isolates four characters; case/shape recognition remains a separate limitation. | `QL1` |
| `620` (Manula_numbers_03) | projection | merge: The 2 and 0 share one crop. Only two crops represent 620; components separates all three. | `67` |
| `vWx9` (Manula_strings_04) | projection | extra_fragment: An additional small crop occurs between W and x. Its origin as a detached stroke versus background artifact cannot be established confidently from this saved review alone. Grouped components returns four glyph crops. | `VW0X9` |

[Paarth_numbers_03 components (archive: evidence/final_evaluation/segmentation_review/Paarth_numbers_03_components.png)](../guides/research-archive.md)

[Manula_numbers_03 projection (archive: evidence/final_evaluation/segmentation_review/Manula_numbers_03_projection.png)](../guides/research-archive.md)

[Haresh_strings_04 projection (archive: evidence/final_evaluation/segmentation_review/Haresh_strings_04_projection.png)](../guides/research-archive.md)

[Paarth_strings_05 projection (archive: evidence/final_evaluation/segmentation_review/Paarth_strings_05_projection.png)](../guides/research-archive.md)

[Manula_strings_04 projection (archive: evidence/final_evaluation/segmentation_review/Manula_strings_04_projection.png)](../guides/research-archive.md)

## Dot grouping, order and clipping

- Three string photos visibly contain a detached dot: Haresh's `iJ09`, Paarth's `jKl8`, and Manula's `Ji10`. Both methods retain each visible dot with its intended stem/body: six reviewed method outputs. This small set does not prove robustness to arbitrary dot displacement.
- Saniru's confirmed `hIj6` is written with an undotted j-like/J-like shape. The source does not contain a separate dot, so its absence must not be attributed to segmentation.
- All 80 box lists have nondecreasing left edges; visual inspection found no character-order swaps. Split/merged crops can still invalidate the sequence despite sorted boxes.
- No obvious loss of an entire glyph stroke through boundary clipping was found at panel resolution. This is not a pixel-perfect guarantee. The synthetic margins and surrounding paper can remain visible; padding does not recover ink absent from the source.
- Haresh's shadowed strings keep dark background in their crops. The selected method can separate characters structurally while the later prepared input remains damaged. These cases are linked to [Paarth's preprocessing review](preprocessing-final-review-week11.md) rather than declared a classifier-only failure.

## Interpretation and limitations

The `462` example shows the weakness of relying on connected regions when one handwritten digit has disconnected strokes. Projection can combine horizontally overlapping fragments into a single digit. Conversely, the `620` and `O0l1` examples show that projection can merge separate characters when their horizontal occupancy has no clean gap. In the shadowed `xYz4` case, background can also affect foreground occupancy; its precise contribution was not isolated by an additional experiment.

The extra small crop in `vWx9` is retained as an extra-fragment finding; whether it is a detached stroke or an image artifact remains uncertain. Do not call it a proven dot-grouping failure.

Correct segmentation is necessary but insufficient for correct recognition. For example, complete case-ambiguous glyph crops can still be classified with the wrong case. All final failures remain included. No model or segmentation method was switched after viewing final scores.

Only four known writers and 40 multi-character photos were reviewed. There are no new unseen-writer samples, human inter-rater measurements or annotated pixel masks. Proposed future work is to test fragment grouping, horizontal overlap and illumination robustness on a separate development collection, then evaluate a changed system on new untouched samples. No improvement from such work is claimed here.

## Per-image comparison

| Sample | Selected structural review | Selected output | Projection structural review | Projection output | Side-by-side evidence |
|---|---|---|---|---|---|
| Haresh_numbers_01 | no_visible_structural_error | `007` | no_visible_structural_error | `007` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Haresh_numbers.jpg)](../guides/research-archive.md) |
| Haresh_numbers_02 | no_visible_structural_error | `240` | no_visible_structural_error | `240` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Haresh_numbers.jpg)](../guides/research-archive.md) |
| Haresh_numbers_03 | no_visible_structural_error | `586` | no_visible_structural_error | `586` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Haresh_numbers.jpg)](../guides/research-archive.md) |
| Haresh_numbers_04 | no_visible_structural_error | `1029` | no_visible_structural_error | `1029` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Haresh_numbers.jpg)](../guides/research-archive.md) |
| Haresh_numbers_05 | no_visible_structural_error | `7314` | no_visible_structural_error | `7314` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Haresh_numbers.jpg)](../guides/research-archive.md) |
| Haresh_strings_01 | no_visible_structural_error | `PvW1` | no_visible_structural_error | `PvW1` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Haresh_strings.jpg)](../guides/research-archive.md) |
| Haresh_strings_02 | no_visible_structural_error | `XYZ2` | no_visible_structural_error | `XYZ2` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Haresh_strings.jpg)](../guides/research-archive.md) |
| Haresh_strings_03 | no_visible_structural_error | `UVw3` | no_visible_structural_error | `UVw3` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Haresh_strings.jpg)](../guides/research-archive.md) |
| Haresh_strings_04 | no_visible_structural_error | `BDz4` | merge | `Ba4` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Haresh_strings.jpg)](../guides/research-archive.md) |
| Haresh_strings_05 | no_visible_structural_error | `iJ09` | no_visible_structural_error | `iJ09` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Haresh_strings.jpg)](../guides/research-archive.md) |
| Paarth_numbers_01 | no_visible_structural_error | `003` | no_visible_structural_error | `003` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Paarth_numbers.jpg)](../guides/research-archive.md) |
| Paarth_numbers_02 | no_visible_structural_error | `803` | no_visible_structural_error | `803` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Paarth_numbers.jpg)](../guides/research-archive.md) |
| Paarth_numbers_03 | split | `6162` | no_visible_structural_error | `462` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Paarth_numbers.jpg)](../guides/research-archive.md) |
| Paarth_numbers_04 | no_visible_structural_error | `7071` | no_visible_structural_error | `7071` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Paarth_numbers.jpg)](../guides/research-archive.md) |
| Paarth_numbers_05 | no_visible_structural_error | `2385` | no_visible_structural_error | `2385` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Paarth_numbers.jpg)](../guides/research-archive.md) |
| Paarth_strings_01 | no_visible_structural_error | `AbC2` | no_visible_structural_error | `AbC2` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Paarth_strings.jpg)](../guides/research-archive.md) |
| Paarth_strings_02 | no_visible_structural_error | `dEF7` | no_visible_structural_error | `dEF7` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Paarth_strings.jpg)](../guides/research-archive.md) |
| Paarth_strings_03 | no_visible_structural_error | `GhI4` | no_visible_structural_error | `GhI4` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Paarth_strings.jpg)](../guides/research-archive.md) |
| Paarth_strings_04 | no_visible_structural_error | `jkl8` | no_visible_structural_error | `jkl8` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Paarth_strings.jpg)](../guides/research-archive.md) |
| Paarth_strings_05 | no_visible_structural_error | `00L1` | merge | `QL1` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Paarth_strings.jpg)](../guides/research-archive.md) |
| Manula_numbers_01 | no_visible_structural_error | `009` | no_visible_structural_error | `009` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Manula_numbers.jpg)](../guides/research-archive.md) |
| Manula_numbers_02 | no_visible_structural_error | `374` | no_visible_structural_error | `374` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Manula_numbers.jpg)](../guides/research-archive.md) |
| Manula_numbers_03 | no_visible_structural_error | `620` | merge | `67` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Manula_numbers.jpg)](../guides/research-archive.md) |
| Manula_numbers_04 | no_visible_structural_error | `5812` | no_visible_structural_error | `5812` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Manula_numbers.jpg)](../guides/research-archive.md) |
| Manula_numbers_05 | no_visible_structural_error | `9460` | no_visible_structural_error | `9460` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Manula_numbers.jpg)](../guides/research-archive.md) |
| Manula_strings_01 | no_visible_structural_error | `Mn53` | no_visible_structural_error | `Mn53` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Manula_strings.jpg)](../guides/research-archive.md) |
| Manula_strings_02 | no_visible_structural_error | `PQrb` | no_visible_structural_error | `PQrb` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Manula_strings.jpg)](../guides/research-archive.md) |
| Manula_strings_03 | no_visible_structural_error | `5tU5` | no_visible_structural_error | `5tU5` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Manula_strings.jpg)](../guides/research-archive.md) |
| Manula_strings_04 | no_visible_structural_error | `VWX9` | extra_fragment | `VW0X9` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Manula_strings.jpg)](../guides/research-archive.md) |
| Manula_strings_05 | no_visible_structural_error | `jil0` | no_visible_structural_error | `jil0` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Manula_strings.jpg)](../guides/research-archive.md) |
| Saniru_numbers_01 | no_visible_structural_error | `005` | no_visible_structural_error | `005` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Saniru_numbers.jpg)](../guides/research-archive.md) |
| Saniru_numbers_02 | no_visible_structural_error | `169` | no_visible_structural_error | `169` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Saniru_numbers.jpg)](../guides/research-archive.md) |
| Saniru_numbers_03 | no_visible_structural_error | `832` | no_visible_structural_error | `832` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Saniru_numbers.jpg)](../guides/research-archive.md) |
| Saniru_numbers_04 | no_visible_structural_error | `4067` | no_visible_structural_error | `4067` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Saniru_numbers.jpg)](../guides/research-archive.md) |
| Saniru_numbers_05 | no_visible_structural_error | `7251` | no_visible_structural_error | `7251` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Saniru_numbers.jpg)](../guides/research-archive.md) |
| Saniru_strings_01 | no_visible_structural_error | `YZA4` | no_visible_structural_error | `YZA4` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Saniru_strings.jpg)](../guides/research-archive.md) |
| Saniru_strings_02 | no_visible_structural_error | `b6d8` | no_visible_structural_error | `b6d8` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Saniru_strings.jpg)](../guides/research-archive.md) |
| Saniru_strings_03 | no_visible_structural_error | `EF62` | no_visible_structural_error | `EF62` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Saniru_strings.jpg)](../guides/research-archive.md) |
| Saniru_strings_04 | no_visible_structural_error | `hIJ6` | no_visible_structural_error | `hIJ6` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Saniru_strings.jpg)](../guides/research-archive.md) |
| Saniru_strings_05 | no_visible_structural_error | `B8S5` | no_visible_structural_error | `B8S5` | [sheet (archive: evidence/final_evaluation/segmentation_review/sheet_Saniru_strings.jpg)](../guides/research-archive.md) |

## Evidence and verification

- [Saved-artifact audit (archive: evidence/final_evaluation/segmentation_review/audit.json)](../guides/research-archive.md): source crop/preparation/annotation hashes, boxes, count and order checks.
- [All 80 visual reviews (archive: evidence/final_evaluation/segmentation_review/visual_review.json)](../guides/research-archive.md): observations, dot checks and descriptive categories.
- [Verification record (archive: evidence/final_evaluation/segmentation_review/verification.json)](../guides/research-archive.md).
- [Panel generator](../../research/review_final_segmentation.py): saved evidence only, no recognition or segmentation calls.
- [Final evaluation report](final-evaluation-week11.md): authoritative four-model and complete-system metrics.

All saved artifacts used by this review were hash-checked again. The independent final-results verifier was rerun to confirm unchanged input/model/runtime-source hashes and score arithmetic. Final-test results and runtime files remain unchanged.

## Explanation for the tutor

“My part separates characters and keeps them in reading order. I reviewed both segmentation methods on the 40 final numbers and strings, checking actual crop contents and dots rather than only crop counts. The selected methods had one visible structural failure; projection had four. Projection fixed the disconnected 4 in 462 but merged the 2 and 0 in 620, so number recognition stayed at 16 out of 20. Grouped components had lower string character error rate, but many recognitions were still wrong even with complete crops. I documented these limitations without tuning on the final test.”

Use this explanation after personally inspecting the evidence; it does not establish unrecorded hours or an independent human review.
