> Recorded study: results and original source hashes are retained. Detailed outputs are in the [research archive](../guides/research-archive.md); current commands use the `research` package.

# Paired GPU augmentation results and selection decision

Reviewed 26 October 2026. All eight full runs returned from the RTX 3070. **Retain the existing production digit CNN and alphanumeric ResNet-18.** No new checkpoint or preparation/segmentation candidate was promoted. GUI/acquisition acceptance is Saniru's separate work.

## Verification

Imported `improvements_gpu_run01` into `experiments/` and `gpu_evaluation_run01` into `evidence/improvements/`, without replacing existing evidence. Archive CRC passed. All eight checkpoint hashes and source snapshots match their recorded hashes. Training source syntax matches the supplied transfer package. Five-epoch histories, best-validation checkpoint choices, original train/validation index hashes, paired initialisation, and configuration consistency were checked.

The independent verifier recalculated eight official benchmark confusion matrices, accuracy, macro-F1 and per-class metrics from saved predictions. It re-ran all **640 development-photo predictions on CPU** using the returned checkpoints and the identical prepared 160-photo manifest. All predictions matched the GPU results. These are full runs (`smoke=false`), not the earlier tiny CPU tests. Training/evaluation used no fresh follow-up photos and did not rerun the old final 120 on these candidates.

Runtime recorded: RTX 3070, CUDA 12.4, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, FP32. Each task used two seeds (42,43), two arms, five epochs and the fixed original splits. This is a limited two-seed investigation; mean results are not independent repeated photo collections.

## Results across both seeds

Each cell is the mean of two model runs on the same collection. Fractional correct counts are arithmetic averages, not partial recognition or additional images.

| Task/measure | Control | Augmented | Interpretation |
|---|---:|---:|---|
| Digit validation accuracy | 98.267% | 97.933% | Down 0.333 percentage points |
| Digit benchmark test accuracy | 98.535% | 98.255% | Down 0.280 percentage points; descriptive only |
| Single-digit development photos | 15.0/20 | 19.0/20 | Better in both seeds |
| Number development photos | 18.5/20 | 19.5/20 | Better mean exact recognition |
| Number development CER | 3.448% | 0.862% | Fewer character errors |
| Character validation accuracy | 87.159% | 86.071% | Down 1.088 percentage points |
| Character benchmark test accuracy | 87.159% | 86.182% | Down 0.976 percentage points; descriptive only |
| Individual-character development photos | 51.0/80 | 56.5/80 | Better individual characters |
| String development photos | 11.5/40 | 10.0/40 | Worse complete strings |
| String development CER | 28.438% | 31.250% | More character errors |

Per-seed exact matches:

| Task and seed | Arm | Individual inputs correct | Complete numbers/strings correct |
|---|---|---:|---:|
| Digits 42 | Control | 16/20 | 19/20 |
| Digits 42 | Augmented | 19/20 | 19/20 |
| Digits 43 | Control | 14/20 | 18/20 |
| Digits 43 | Augmented | 19/20 | 20/20 |
| Characters 42 | Control | 52/80 | 12/40 |
| Characters 42 | Augmented | 56/80 | 11/40 |
| Characters 43 | Control | 50/80 | 11/40 |
| Characters 43 | Augmented | 57/80 | 9/40 |

## Apply the rule recorded before training

The rule in `improvement-study-week11.md` requires no reduction in mean validation accuracy, no reduction in mean development exact matches or increase in CER in either relevant collection, and a strict development gain. Representative seeds must be selected by validation, not by whichever photo score is highest.

- **Digit augmentation:** passes the development requirements, but fails the no-validation-regression requirement. Both seeds improve single-digit photos. This is a promising domain-transfer tradeoff, not proof that augmentation is generally ineffective. Nevertheless, changing the selection rule after seeing these results would need explicit disclosure and a new evaluation plan. Retain the existing production CNN under the current rule.
- **Character augmentation:** fails both validation and string-development requirements. Improved isolated letters do not compensate for poorer complete-string recognition in the approved extension. Retain the existing production ResNet-18.
- Among augmented runs, seed 42 has the highest validation accuracy for both tasks. Do not choose digit seed 43 simply because it reached 20/20 development numbers. The seed-42 candidate would score 19/20 digits and 19/20 numbers against production 16/20 and 19/20; the character candidate scores 56/80 and 11/40 against production 52/80 and 12/40. These are development results, not the 120-photo final evaluation.

No changes were made to the GUI model resolver, production checkpoint paths, preprocessing, segmentation or the original final evaluation. The original frozen evaluation therefore remains the final-photo evidence for the retained system. It does not evaluate these rejected candidates.

## Do we need the extra photos?

**No extra collection is needed solely to evaluate a replacement that we are not adopting.** The original 120-photo evaluation remains usable for the original frozen system, with its known-writer and framing limitations. A new outside-writer sample could strengthen generalisation evidence, but is optional additional research, not something already completed or a guaranteed marking requirement.

The earlier 180-photo draft has been superseded. The latest discussed alternative was 15 new photos from Paarth plus 15 from one outside writer, but the user agreed to wait for GPU results. That collection remains ON HOLD, uncollected. Do not collect 180 images from the old guide. If a later change is adopted, agree and lock a new evaluation plan before predictions.

## Evidence and next work

- `experiments/improvements_gpu_run01/`: eight checkpoints, configurations, histories, original splits and source snapshots.
- `evidence/improvements/gpu_evaluation_run01/`: eight benchmark prediction/confusion archives, per-class metrics, 640 photo predictions, summaries and independent verification.
- `verify_gpu_improvements.py`: repeatable integrity, arithmetic, pairing and CPU reproduction checks.
- `import_provenance.json`: original archive digest and transfer verification.

The non-GUI improvement investigation is complete for this round. Integrate the accepted/rejected experiments and limitations into the report and presentation. Preserve the original final results, include Saniru's GUI/acquisition evidence when supplied, and finish submission checks. No commit or push was performed in this review.
