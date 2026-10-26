> Recorded study: results and original source hashes are retained. Detailed outputs are in the [research archive](../guides/research-archive.md); current commands use the `research` package.

# Completion and marking-readiness audit

Audit date: 26 October 2026. Scope excludes authoring the final team report and presentation video. This is a local-artifact audit, not a prediction or guarantee of marks, a new GUI acceptance session, or a verification of Canvas/OneDrive permissions.

## Source requirements

All four pages of `../../assignment options/option B/COS30018 - Project Assignment - Option B.pdf` were extracted and the page 4 marking table visually checked. The complete `../../assignment options/COS30018 - Assignment Deliverable Guide.doc` was read through a hidden read-only Word instance. No separate HD-band rubric was found among the supplied assignment materials. The brief gives category marks and broad requirements, with no numerical accuracy threshold guaranteeing HD/full marks.

The requested exclusion removes report (9) and video (3) from the team discussion, leaving technical/research categories worth 48 possible team marks. These are available marks, not marks awarded. Four individual sprint reports contribute up to 40 separately. Good programming, weekly progress and individual contribution can affect marks through the penalties listed in the brief.

## Requirement-to-evidence review

| Category | Marks available | Verified state | Remaining limitation |
|---|---:|---|---|
| Task 1: preprocessing investigation, comparison, selection and implementation | 5 | Grayscale/fixed/Otsu experiments; adopted peak contrast normalisation; controlled rejected illumination/component/margin/mask candidates; retained arrays, hashes and before/after examples | Shadow/texture and thin-stroke errors remain; final research explanation and citations must justify choices |
| Task 2: segmentation investigation, comparison, selection and implementation | 5 | Connected components and projection; character grouping/dot handling; independent and combined repair experiments; selected method integrated; structural review of 40 multi-character final photos with 80 method outputs | Disconnected strokes/overlap remain limitations; final visual review is assistant assessment, not independently annotated human ground truth |
| Task 3: different ML techniques, implementation, evaluation and selected integration | 14 | MLP, CNN, ResNet-18 and MobileNetV3-Small, digit and 62-class checkpoints; official MNIST/EMNIST benchmarks; validation selection; paired photo comparisons; class-weighted and two-seed augmentation studies; eight production checkpoints resolve and hash-check | Limited search budget/repeats; no evidence that all architectures are globally optimised; four-model tutor request is met |
| Task 4: integration and comprehensive evaluation | 6 | Integrated pipeline; 120 locked fresh known-writer photos; four collections, case-sensitive accuracy/CER, per-writer results, timing, failure analysis; 680 prediction rows independently verified; new GPU study reproduced all 640 development predictions | Known writers only, small collections, tightly framed inputs; clean final distribution and final acceptance evidence not yet verified |
| GUI/acquisition | 6 | Both photo loading and construction from digit-image folders; four-model choice, preparation/segmentation choices, previews, export, errors, threaded work; saved desktop smoke evidence | Saniru final native-dialog/second-computer evidence reported done by user but upload still pending; do not claim that audit independently witnessed it |
| Approved alphanumeric extension | up to 12 | 62-class training, grouping, recognition, acquisition and GUI integration; actual benchmark and real-photo evaluation; controlled investigations | Main performance risk: only 4/20 exact fresh strings. Functional completion does not establish the brief's 'complete it very well' standard or full marks |

Current final selected-system scores: 37/40 single digits, 16/20 numbers, 29/40 letters, 4/20 strings. Those are separate collections, not one combined accuracy. Existing data contain failures; no rejected input or wrong prediction has been hidden by this audit.

Current verification records: 77 unit tests run, 76 passed, one CUDA-only test skipped on this CPU host. This is the saved result from this session; the suite was not needlessly rerun for the audit because production source did not change. All eight GUI checkpoint routes/hash checks were freshly exercised. Current frozen-evaluation runtime source hashes match their recorded values. `app.py` and `gui_core.py` also match the last saved Week 11 GUI smoke source. Latest GPU results have independent checkpoint/source/selection/metric checks and exact CPU reproduction of all 640 development predictions. Automated or scripted GUI checks do not prove fresh-install/native-dialog acceptance.

## Concrete remaining work besides final report/video

1. **Receive and verify Saniru's final GUI/acquisition acceptance evidence.** Record tested version, machine/environment, actual native file selection/cancellation, loading, image generation from supplied samples, recognition, error handling, export and screen usability. User-reported completion is recorded, not treated as missing implementation.
2. **Refresh the current project entry points.** `README.md` still lists only MLP/CNN, earlier development results, 52 tests and says fresh evaluation is outstanding. `docs/guides/installation.md` lists only old saved-model folders, omitting the new default ResNet/MobileNet folder. The detailed GUI guide does list the new folders. Make the quick-start consistent with the current system; retain older experiment reports as dated history, but resolve stale pending-task wording with clear updates.
3. **Complete the required contribution/portfolio paperwork.** The deliverable guide explicitly requires the Who Did What template signed by all members. Record real coding, research, testing and presentation contributions with evidence; do not equate commit authorship with personal mastery or fabricate personal testing. Cover sheet, deliverables checklist and relevant appendices must also be included. Local files include Paarth worklogs through Week 10 and a Sprint 3 report; completeness/submission of all four members' four sprint reports and weekly progress cannot be verified from this checkout.
4. **Confirm teaching-staff access and preservation of approval.** Git read-only access and OneDrive access are mandatory. Tutor extension approval was supplied in chat; retain it in the shared project records. Actual staff permissions, tutor progress records, sprint submissions and OneDrive contents are not verified by local files.
5. **Commit and publish the final approved evidence/code.** Local Git status shows uncommitted/untracked final evaluation, reviews and improvement/GPU evidence. They are not part of the current local commit. No network fetch or remote-permission check was performed in this audit, and no commit/push was authorised here.
6. **Build and test the final runnable distribution.** Source, launchers, requirements, saved checkpoints AND selection files, representative input photos, and labelled generator samples need to be accessible in the submitted working version. `data/` is Git-ignored; a repository-only zip will not automatically contain demonstration/generation examples. Exclude .venv, temporary/corrupt test fixtures and smoke-training weights from the runtime distribution. Test the extracted package on a clean location/computer. The deliverable guide lists source code and executable; the supplied brief requires a working version, not explicitly a standalone Windows .exe. Ensure the Python runnable package is acceptable or clarify if the tutor expects a compiled executable.
7. **Final submission check.** At least one member submits the single complete ZIP to Canvas; oversized video may be downloadable on OneDrive. The supplied due date is 11:59 pm 3 November 2026 unless the unit has formally changed it. Verify access/download and keep submission confirmation.

## HD/max-mark judgement

There is substantial evidence for the core technical requirements. The rejected experiments add useful research evidence, but do not automatically earn marks. The strongest unresolved technical concern is generalisation and exact alphanumeric-string accuracy. High isolated benchmark scores are not interchangeable with reliable whole-string recognition. Explain accepted/rejected choices and tradeoffs, support algorithm descriptions with appropriate sources, and have each member understand and personally check their section.

No requirement in the supplied brief mandates another architecture, a segmentation neural network, a specific number of fresh photos, a particular accuracy, or the earlier proposed 180-image collection. The old 120-image evaluation remains valid for the unchanged frozen system. The optional 30-image follow-up is on hold because no new candidate was adopted. Additional unseen-writer evidence could strengthen the case, but it is not needed merely to claim another implementation task complete. If the tutor expects stronger extension performance for full research marks, that remains a real risk; report polish cannot guarantee those marks.

Conclusion: core implementation and this round of research/evaluation are complete within the approved scope; whole-project closeout is not complete, and HD/full marks cannot be certified. Prioritise the seven concrete closeout items rather than claim the project has nothing left beyond report/video.

## Primary project evidence

- `docs/reports/preprocessing-experiment-week10.md`
- `docs/reports/segmentation-experiment-week10.md`
- `docs/reports/model-comparison-week11.md`
- `docs/reports/final-evaluation-week11.md`
- `docs/reports/preprocessing-final-review-week11.md`
- `docs/reports/segmentation-final-review-week11.md`
- `docs/reports/gpu-augmentation-results-week11.md`
- `docs/reports/gui-integration-week11.md`
- `evidence/final_evaluation/status.json`
- `evidence/improvements/regression_tests.txt`
- `evidence/improvements/gpu_evaluation_run01/independent_verification.json`

This audit adds a review document only. It does not change the application, trained models, prior scores, staff permissions or submitted work.
