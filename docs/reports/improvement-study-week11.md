> Recorded study: results and original source hashes are retained. Detailed outputs are in the [research archive](../guides/research-archive.md); current commands use the `research` package.

# Non-GUI improvement study after the Week 11 evaluation

Date: 26 October 2026.

**Update after the PC run:** All eight full GPU runs are now verified. Neither augmentation arm passed the pre-recorded adoption rule; production is retained. See [GPU results and decisions](gpu-augmentation-results-week11.md). Remaining-training statements below describe the earlier preparation stage; the fresh-photo collection is now on hold. Scope: preprocessing, segmentation, model investigation, extension and evaluation methodology. GUI/acquisition acceptance remains Saniru's separate work. No new runtime default or model has been installed.

## What the evidence supports

Two bounded development experiments are complete. Neither demonstrated a useful replacement for the selected preparation or segmentation. Rejected changes are retained as investigation evidence. Training infrastructure and an RTX transfer package are complete, but full augmented model training and evaluation require the PC run. Fresh v2 photography and its final evaluation remain pending. This does not constitute completion of every improvement, and no mark is guaranteed.

## Paarth: preprocessing

Hypothesis A: a 15% pale border may stabilise local illumination estimation for tightly framed character crops. Compare with the same baseline crops, models and 160 development inputs. Character and string predictions did not improve, so reject the change under the recorded strict-gain rule.

Hypothesis B, explicitly a subsequent development study: reusing segmentation foreground may avoid damage from a second crop threshold. This candidate changes character preparation only, retains ordering, preserves group boxes, and falls back to production preparation for overlapping group boxes. It reduced individual-character accuracy and slightly increased string CER, so reject it too.

| Collection | Current route | Pale-margin candidate | Foreground-mask candidate |
|---|---:|---:|---:|
| Single digits | 16/20 | unchanged digit route | unchanged digit route |
| Numbers | 19/20 | unchanged digit route | unchanged digit route |
| Individual characters | 52/80; CER 37.5% | 52/80; CER 37.5% | 50/80; CER 40.0% |
| Strings | 12/40; CER 26.875% | 12/40; CER 26.875% | 12/40; CER 27.5% |

The individual-character baseline here includes production segmentation. Earlier 54/80 ResNet results prepared the entire single-character photo directly; they are a different route, not evidence of a model regression. Current routing produces the expected one crop on 77/80 character photos. All 40 string crop counts match, which does not alone prove correct boundaries.

Visual spot checks of changed 0, 1 and Z examples show altered ink representation despite similar gross shapes. Two previously correct character predictions become wrong (0 to i and Z to 2). Other changed predictions and before/after panels are preserved; the visual review is a spot check, not a claim that a human annotated every boundary.

Evidence: `evidence/improvements/development_run01/` and `development_run02/`: protocols saved before each inference run, source snapshots, checkpoint hashes, 320 predictions per run, prepared arrays, previews and summaries. `mask_changed_predictions.png` shows nine prediction changes. No original photo was edited.

## Manula: segmentation

Implemented a candidate that merges nearby complementary wide/narrow digit fragments under fixed geometric conditions. It does not consult labels, expected crop count, or model predictions. Two narrow neighbouring digits and distant components are deliberately excluded. It retains the existing grouped-component character segmentation.

The candidate changed no boxes on the 40 digit/number development photos; results remain 16/20 single digits and 19/20 numbers. Therefore there is no real-photo evidence of a gain and the production segmenter remains selected. A synthetic broken-4 fixture improves from three components to two characters while a following digit stays separate. This demonstrates one targeted capability, not real-world accuracy or a reason to promote the rule. Dotted-character, narrow-pair and distant-component checks also pass.

The existing final segmentation review remains the structural evidence for v1. A second human review of boundary completeness/dot assignment is still useful; do not claim it has happened. New v2 images should be reviewed for complete character crops and reading order, separately from model classification. Cases where crop count matches but a character is merged/clipped still count as segmentation errors.

Evidence: `improvement_candidates.py`, `tests/test_improvement_study.py`, both development studies, and `docs/reports/segmentation-final-review-week11.md`.

## Haresh: paired training and model comparison

The new experiment trains the selected digit CNN and character ResNet-18 from scratch. Each has two arms (control/no augmentation and modest affine/intensity augmentation), two seeds (42, 43), five epochs, Adam at 0.001, batch 256, original train/validation splits, and unweighted cross-entropy. Eight full runs in total. Pair members start from identical weight hashes and receive identical minibatch orders. Augmentation has a separate random generator and never touches validation or test images. Each image receives independent small rotation, translation, scaling and gain; no flips or case relabelling are used.

Best checkpoint per run is selected by validation accuracy, then validation loss, then earlier epoch. The evaluator checks all eight selections before loading official benchmark test data. It produces accuracy, macro-F1, per-class metrics, confusion matrices and paired predictions on the same saved development arrays. Fresh photos are not loaded by either script. Two seeds give a limited repeatability check, not a statistically powered significance claim.

Decision rule recorded before the full GPU experiment: inspect both seed-paired results; require no decrease in mean validation accuracy, no decrease in mean development exact matches or increase in mean CER in either relevant collection, and at least one strict development improvement over paired controls. Choose a representative seed within an arm using validation only. Before promotion, that checkpoint must also avoid exact-match/CER regression against the existing production checkpoint on BOTH relevant development collections. Official benchmark test scores are reported, not used to pick a seed. If these conditions fail, retain production and report the negative experiment. Any changed rule requires a dated amendment and disclosure before a new test.

Do not conclude augmentation helps until the full GPU outputs exist. CPU smoke runs use 128 training / 64 validation examples for one epoch and are only code-path checks. Their accuracy has no performance interpretation.

Evidence: `train_improvements.py`, `evaluate_training_improvements.py`, `preflight_improvement_training.py`, `evidence/improvements/training_smoke/`, `evaluation_smoke/`, and the transfer guide.

## Extension and integration/evaluation

The approved scope remains 62 classes and single-line, separately written alphanumeric input. The baseline is still case sensitive: 29/40 final letters and 4/20 final strings. Of the 11 failed letters, five are case-only differences. Of the 16 failed strings, seven are case-only differences. Ignoring case gives 34/40 letters and 11/20 strings, but these are DIAGNOSTICS, never replacement success scores.

Added Wilson 95% image-level intervals and an exploratory writer-cluster bootstrap, with explicit limits. For example, 4/20 strings has an image-IID interval of approximately 8.1% to 41.6%. Repeated samples from only four known writers violate independent-photo assumptions; neither interval establishes unseen-writer population accuracy. Report per-writer results and sample counts. Preserve case and leading zeros.

The fresh-v2 plan has 180 images across six writers: four known and two new writers, 30 each. All 52 letter cases are covered, alongside digits and mixed strings. This count is our proposed study design, not a requirement quoted from the brief. Anonymous outside-writer IDs and planned labels are templates only. Dates, confirmations and hashes remain blank until actually collected. Keep this collection untested until the model/configuration is frozen. Compare v1 and any accepted v2 system on the same new collection, report known/new cohorts separately, preserve every valid failure, and separate preparation, segmentation and classification errors. Do not reuse the old 120 as an untouched test for changes informed by its failures.

Evidence: `report_improvement_uncertainty.py`, `frozen_v1_uncertainty.json`, `docs/guides/fresh-evaluation-v2-photos.md`, and `fresh_collection_plan/assignment_manifest.json`.

## Verification and boundaries

- Two studies independently checked: 160 input hashes, 640 prediction/array records, arithmetic, source snapshots and unchanged baseline arrays between runs.
- Five new targeted tests pass. Full suite: 77 tests, 76 passed, one CUDA-only test skipped on this CPU laptop. An initial suite run failed on sandbox temporary-directory permissions; rerunning with a writable workspace temporary directory passed. No application repair was needed for that environment issue.
- Tiny CPU train/evaluation completed for both tasks and both arms; pairing checks passed. CUDA execution remains to be verified on the RTX PC.
- Original final-run verifier passes: 120 photo hashes, eight checkpoints and 680 predictions checked. Existing production preprocessing, segmentation, GUI and checkpoint files were not modified.
- Training smoke checkpoints are excluded from Git to avoid treating demonstration weights as deliverable models. Existing real checkpoints are preserved.
- No commit or push was performed. Saniru's GUI/acquisition evidence remains outside this work.

## Research rationale and references

EMNIST ByClass retains 62 unbalanced classes, so class-level analysis and case-sensitive evaluation matter. The NIST page and original paper explain its construction and splits: [NIST EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset), [Cohen et al., 2017](https://arxiv.org/abs/1702.05373). Small geometric transforms are supported image-augmentation operations: [Torchvision 0.21 RandomAffine](https://docs.pytorch.org/vision/0.21/generated/torchvision.transforms.RandomAffine.html). Our batched implementation uses [PyTorch grid_sample](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html). These sources motivate the experiment; they do not prove that this chosen augmentation improves our photographed handwriting.

## What remains before deciding v2

1. Run the RTX package and return both full output folders; verify and interpret the actual paired results.
2. Promote only a supported checkpoint/configuration, with explicit source and checkpoint hashes; then run integration tests for the changed model. No GUI redesign is requested here.
3. Collect, confirm and lock fresh v2 photos, including actual outside-writer coverage. Run the agreed frozen evaluation once.
4. Incorporate real results into the team report and presentation; each member reviews and understands their own evidence. Saniru supplies GUI/acquisition acceptance separately.
