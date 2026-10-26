# Selected results

These files are compact copies of recorded outputs, not new experiment runs.

| Folder | Evidence |
|---|---|
| `final_evaluation/` | Frozen configuration, locked labels, 120-photo metrics, writer breakdowns and verification |
| `four_models/` | Benchmark and development-photo comparisons of four architectures |
| `augmentation/` | Control versus augmented training, benchmarks, photo metrics and import verification |
| `preprocessing_candidates/`, `mask_candidate/` | Additional preprocessing and mask investigations |
| `gui/` | Recorded local desktop verification |
| `datasets/` | Exact EMNIST split used by the experiments |

[The archive guide](../docs/guides/research-archive.md) explains how to obtain every prediction, crop, source snapshot and checkpoint behind these summaries. Paths inside copied JSON records refer to their original experiment layout; they are provenance, not links into this compact repository.

`verification-checks.txt` records the current project test run. A skipped CUDA check is not a GPU verification. Local GUI verification does not establish another computer's acceptance result.
