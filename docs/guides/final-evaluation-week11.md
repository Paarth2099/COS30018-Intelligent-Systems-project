# Week 11 final evaluation protocol

Original status: prepared before fresh-photo inference. Counts are the team's chosen plan, not a PDF requirement.

Update, 26 October 2026: the user confirmed all assigned labels, close camera framing and that none of the 120 new photos had been tested. The user reports Saniru's second-computer check completed and explicitly authorised evaluation before its evidence is uploaded. Run 01 locks the collection and exact code/model hashes before inference; acceptance evidence remains pending and is not independently verified. The recorded version is the base Git commit plus archived source hashes, including the uncommitted evaluator. Separate teammate label inspection is not claimed: Paarth's collective attestation and the assistant's visual review are recorded. See `evidence/final_evaluation/run_01/` for run state; only completed verification establishes completion. The original gate sequence below is retained to distinguish the plan from this authorised evidence-upload deferral.

## Purpose and scope

Evaluate the selected number and approved alphanumeric systems on newly written photos, separate from all development images. Alphanumeric scope is single-line, separately written A–Z, a–z and 0–9, without cursive, spaces, punctuation or multiple lines. The four group members are known development writers: fresh samples measure new inputs from those writers, not unseen-writer generalisation.

## Collection fixed before predictions

Each of Haresh, Paarth, Manula and Saniru supplies 10 single digits, 5 numbers, 10 single letters and 5 mixed strings: 30 each, 120 total. Across the group this gives 40 digits, 20 numbers, 40 letters and 20 strings. Single-digit photos also cover the digit classes of the 62-class recogniser in a separate auxiliary analysis; do not pool them into the primary letter score or count them as extra independent photos.

Follow [exact assignments](final-photo-assignments.md). All 52 letter cases occur across letters and strings. Four number photos use lighter visible strokes; four strings use close but non-touching spacing. The remaining samples are ordinary handwriting, including assigned leading zeros and ambiguous shapes. Do not deliberately distort characters. This is a small designed collection, not a random population sample.

Paarth keeps originals under `data/final_evaluation/raw/`. The [manifest (archive: evidence/final_evaluation/protocol/collection_manifest.json)](research-archive.md) supplies IDs and intended labels. Record actual filename, verified label, writer, date, hash and second-person label check before inference. Preserve labels as text, including leading zeros and case. If the writing differs from the assigned text, resolve the label with the writer before prediction and document it.

Check readability, valid image format, complete character visibility and duplicates against this collection and existing development photos. Use byte hashes plus visual checks for re-encoded duplicates. Retain rejected originals and record objective reasons (corrupt image, duplicate, wrong task or cropped-off writing). Request replacements before locking. Never exclude a valid sample because recognition fails. Do not manually crop or enhance originals. Keep the raw collection in OneDrive too: the repository's existing `data/` ignore rule means Git does not back it up.

## Configuration and comparison design

The [candidate snapshot (archive: evidence/final_evaluation/protocol/candidate_configuration.json)](research-archive.md) records the current code commit, source and eight checkpoint hashes. It is not yet the final freeze.

- Primary Numbers: CNN, `components`, `otsu`, current digit contrast normalisation.
- Primary Alphanumeric: ResNet18, `grouped_components`, `otsu`, existing character preparation and adopted segmentation repairs.
- Model comparison: all four architectures on the same relevant photos and identical prepared crops. Evaluate digits/numbers with MNIST heads, letters/strings with EMNIST heads. Report the auxiliary EMNIST digit result separately.
- Segmentation comparison: on the 20 numbers and 20 strings, compare `projection` against the selected method with the primary model held fixed. Never supply true labels or expected character counts to segmentation or recognition.
- No new training, preprocessing tuning, model promotion or threshold selection from these final results. Report a worse primary result honestly. If a real defect requires a fix, retain the original run, identify the new version, and label reuse of this collection as post-test development; a new untouched set is needed for a new independent final claim.

## Gate before scoring

1. Saniru performs and records the updated GUI check on another computer with a clean environment, native file/export dialogs, both input routes, both modes, four models, leading zeros, errors, scrolling and responsiveness. Record machine, dependency versions, actual outcomes and screenshots; do not mark unperformed checks passed. Use development photos for acceptance, not the final collection.
2. Resolve blocking defects, rerun relevant checks, then record the actual final Git commit, source/checkpoint hashes and settings in a separate frozen configuration. Preserve this candidate snapshot.
3. Paarth and a second reviewer verify all 120 manifest entries without seeing predictions, lock labels/files with hashes and record exclusions/replacements and lock time.
4. Haresh records evaluator code/version and environment. Verify it reproduces known development outputs, then run the predeclared comparisons on the locked collection. Do not start scoring until the gates are recorded complete.

## Metrics and evidence

Keep the four primary collections separate. Report correct/total and percentage for single inputs and exact, case-sensitive whole-number/string matches. Preserve leading zeros. Report micro character error rate as total Levenshtein edit distance divided by total reference characters for multi-character collections. CER can exceed 100 percent. Include per-writer counts and descriptive results; this small, known-writer collection cannot support broad generalisation claims.

Retain every valid input in the denominator. Empty output counts as wrong, with full reference-length deletions. Log crashes/errors separately and count them as failed recognition; use empty output for CER rather than dropping the row. For isolated inputs, count multiple/no crops as failures and show those categories alongside class confusion, rather than silently removing them. Report model comparison changes on paired sample IDs.

Manula reviews ordered annotated crops for all 40 multi-character originals under both segmentation methods. Record complete characters, dot assignment, reading order, split and merge errors and clipping, with supporting images. Matching the expected crop count alone is not segmentation success. Distinguish segmentation failures, preparation failures and classification errors; allow multiple causes or uncertain attribution.

For timings use one recorded machine/device and four CPU threads for comparability with current GUI inference. Exclude initial checkpoint loading, warm each configuration on development inputs, then time five repeats per final input using a monotonic timer. Separate end-to-end processing (preparation, segmentation, inference) from model-only timing; exclude disk/image loading, GUI interaction and export. Rotate model order across inputs. Save raw timings and report per-photo medians and collection median/p95; repetitions are not extra independent images.

Save input hashes, settings, per-image predictions, metrics, prepared crops, annotations, error records and timing/environment details in a new run directory. Preserve baseline development results separately. Exact evaluator implementation and validation remain Haresh's next task; this protocol does not claim it has run.

## Responsibility and immediate deliverable

| Person | Next work | Evidence |
|---|---|---|
| Haresh | Review protocol; prepare and validate final evaluator; run only after gates | Frozen configuration, evaluator, prediction tables, metrics, model discussion |
| Paarth | Collect and verify originals/labels; check duplicates; lock collection | Manifest, hashes, review/exclusion record; later preprocessing error analysis |
| Manula | Prepare crop-review form; later review both segmentation methods | Annotated crops, per-image split/merge/dot/order review and discussion |
| Saniru | Run updated second-computer manual acceptance before freeze | Machine/versions, native-dialog and workflow checklist, screenshots, exports |
| Everyone | Supply assigned fresh writing; review findings and assemble report | Report sections, actual contribution/worklogs and video outline |

The immediate next human action is to collect the assigned photos and complete the second-computer GUI check. No final images have been evaluated by this setup step.
