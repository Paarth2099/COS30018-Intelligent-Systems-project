# Verification and brief review â€” 28 October 2026

## Scope

Reviewed all four pages of the supplied Option B brief, including the marking table and submission requirements. This review maps available implementation/evidence to the requirements; it is not a grade guarantee. Historical reports retain the results and limitations of their original studies.

| Requirement | Implementation and evidence | Remaining boundary |
|---|---|---|
| Task 1: preprocessing (5) | Digit and character preparation; grayscale/fixed/Otsu comparisons; subsequent controlled candidate studies | Explain selection and failures in final report |
| Task 2: segmentation (5) | Components/projection and character grouping; development comparisons and crop reviews | Explain split/merge failures and review limits |
| Task 3: ML comparison (14) | Four neural architectures; five classical baselines; fixed validation selection and benchmark/photo comparisons | Present accuracy, speed, size and reasons for retaining defaults |
| Task 4: integration/evaluation (6) | Integrated photo-to-text pipelines, unit/GUI tests, frozen 120-photo evaluation | Keep final, development and post-hoc evidence separate |
| GUI/acquisition (6) | Photo loading and sample-based generation; model/settings choices; Model lab, live curves, confusion matrices and exports | Native file pickers and another computer require human acceptance evidence |
| Approved extension (up to 12) | Alphanumeric recognition, case-sensitive evaluation and line-context case rule | Scope remains separate single-line characters; final case-rule scores are post-hoc |
| Programming practice (up to -10 penalty) | Consistent Black formatting, helpful existing comments, separated GUI/training/model modules, research-only week modules moved into research | Formatting alone does not establish full marks |
| Team report (9) | Source guides and experiment reports available | Assemble/review required 20â€“30-page report with all listed sections |
| Video (3) | Runnable demonstration system | Produce and verify 10â€“12-minute presentation |
| Individual work | Contributor records and historical progress material | Members must confirm actual contributions and complete weekly/sprint records |
| Submission/access | Git repository maintained; shared OneDrive archive | Confirm teaching staff access; submit code + report + video ZIP by 11:59 pm 3 November 2026 unless formally changed |

## Changes in this cleanup

- Formatted application and model modules using Black 25.1.0; compared their parsed AST before and after, with no logic differences.
- Moved `hnrs_chars/week10_weighted.py`, `week10_evaluate.py` and `week10_report.py` into `research/`; updated imports, tests and command references.
- Updated architecture, installation, results index, archive scope and research command mapping.
- Preserved original experiment snapshots/checksums and the provenance identifying an automated image review. No human ground-truth review is implied.
- Historical final-result verification now reports current source drift separately from frozen snapshot integrity, because optional case correction and Model lab intentionally changed current source.

## Test coverage and limits

Detailed logs in this folder record the final test runs. The automated suite has 98 tests; on this CPU environment the CUDA-only test is skipped. Real-window checks exercise all eight shipped mode/model combinations, acquisition/generation, exports, validation errors, responsiveness, scrolling and the Model lab. The lab check trains a real small MNIST model, checks its validation results and selects it for recognition. A small demonstration model can produce an incorrect photo prediction without a software failure.

Native file-dialog responses are supplied programmatically in GUI checks. This does not replace manual acceptance on another computer. The clean release check reuses this machine's Python environment and uses no ignored photo/evidence folders. Training datasets are intentionally not shipped in GitHub.

Frozen final recognition remains 37/40 digits, 16/20 numbers, 29/40 individual characters and 4/20 exact strings. The later case rule achieves 20/40 writer-held-out development strings and 6/20 post-hoc final strings. These are different evaluation conditions. No new independent final collection is claimed.

## Final recorded outcomes

- Clean package: 98 tests run, 97 passed and one CUDA-only skip; all eight shipped checkpoint hashes verified.
- Clean-package recognition GUI: PASS, including all modes/models, generated images, exports and error recovery.
- Model lab: PASS, all tabs rendered, invalid settings rejected, a real 3,000-image MNIST MLP trained for two epochs (85.9% validation), registered and used for prediction.
- Dependencies: no broken installed requirements. All model/research modules imported; relative Markdown links checked with none broken.
- Frozen evidence: 680 prediction records, 120 photo hashes and eight checkpoint hashes verified. Current source differs intentionally in the GUI and optional case-rule integration; the frozen snapshots remain intact.
- Cosmetic formatting preserved the application AST. A reporting metadata string now says automated visual inspection rather than naming the tool, retaining its non-human review limitation.

No mark guarantee is implied. The original 26 October OneDrive archive remains unchanged; its guide now explicitly excludes the later additions. Current additions and summaries are in GitHub.
