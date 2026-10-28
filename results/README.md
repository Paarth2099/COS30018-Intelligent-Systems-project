# Selected results

These files are compact copies of recorded outputs, not new experiment runs.

| Folder | Evidence |
|---|---|
| `final_evaluation/` | Frozen configuration, locked labels, 120-photo metrics, writer breakdowns and verification |
| `four_models/` | Benchmark and development-photo comparisons of four architectures; `benchmark_confusion.json` holds their official-test confusion matrices for the GUI |
| `classical_baselines/` | kNN, SVM and random-forest validation grid, MNIST test benchmark and identical-crop photo comparison |
| `case_rule/` | Line-context letter-case rule: development fit, leave-one-writer-out folds, before/after predictions for all models |
| `augmentation/` | Control versus augmented training, benchmarks, photo metrics and import verification |
| `preprocessing_candidates/`, `mask_candidate/` | Additional preprocessing and mask investigations |
| `gui/` | Recorded local desktop verification |
| `datasets/` | Exact EMNIST split used by the experiments |

[The archive guide](../docs/guides/research-archive.md) describes the 26 October snapshot and its scope; the later classical/case-rule results are published here separately. Paths inside copied JSON records refer to their original experiment layout; they are provenance, not links into this compact repository.

`verification-checks.txt` and `system-check-2026-10-26/` are historical checks before the Model lab and case-rule additions. The latest cleanup verification is in `system-check-2026-10-28/`. A skipped CUDA check is not a GPU verification. Local GUI verification does not establish another computer's acceptance result.
