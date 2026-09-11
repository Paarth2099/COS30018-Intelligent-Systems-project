# Haresh: evidence record and worklog guidance

The source code, training runs, evaluation and these notes were produced with
Codex assistance at Haresh's request on 11 September 2026. The recorded results
come from executed code, not invented performance figures. This record is not
a claim that Haresh independently wrote every file or has already learned it all.

## Evidence available

| Work | Evidence |
|---|---|
| Shared setup and data inspection | requirements.txt; inspect_mnist.py; evidence/dataset_report.json |
| Two model implementations | hnrs_ml/models.py |
| Fixed train/validation separation | experiments/baseline/split_indices.npz; config.json |
| Training and checkpoint selection | history.csv; selection.json; mlp.pt; cnn.pt |
| Final digit evaluation | evaluation/results.json; per-class and per-image CSV files |
| Ordered number reconstruction | predict.py; generated_strings.csv; example_prediction.json |
| Error analysis | cnn_errors.png; cnn_confusion.png; docs/ML_REPORT.md |
| Verification | tests/test_contract.py; evidence/verification.txt |

## Fill this using your actual experience

- Date/week: [your actual work date and teaching week]
- Task assigned: machine-learning setup, baseline comparison and prediction handoff.
- What I personally reviewed/ran/changed: [complete after doing it].
- Actual time spent: [record your time; do not use model training time as work hours].
- What I learned: [explain a concept in your own words].
- Problems and how I addressed them: [actual experience].
- Evidence: [link to the branch/commit and relevant result files].
- AI assistance: [describe generation/debugging/documentation assistance according
  to the unit's rules].
- Next work: confirm a teammate's setup; review prediction integration; collect
  real-photo evaluation results; discuss approved extension scope.

Use your official OneDrive worklog and sprint templates for submission. Their
original DOCX files are not available in this workspace, and sprint due dates
still need to follow tutor/Canvas confirmation.

## Questions to be ready to answer at the tutor demonstration

1. Why are training, validation and test data different sets?
2. How does the MLP process an image differently from the CNN?
3. Why was the CNN selected, and which data determined that choice?
4. What does a 28x28 image represent? What is the channel dimension?
5. Why might a high digit accuracy still give an incorrect multi-digit number?
6. What exactly does the generated-string test leave untested?
7. Which code did you review or change, and what did you learn from it?

Answers to the technical questions are in START_HERE.md and ML_REPORT.md. The last
question needs your own account of your contribution.
