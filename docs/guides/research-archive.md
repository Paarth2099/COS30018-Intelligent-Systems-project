# Research archive

The full pre-organisation project evidence is preserved in [the shared OneDrive Evidence or experiments folder](https://liveswinburneeduau-my.sharepoint.com/my?id=%2Fpersonal%2F105344622%5Fstudent%5Fswin%5Fedu%5Fau%2FDocuments%2FCOS30018%20Intelligent%20System%2FEvidence%20or%20experiments). Access uses the team's university permissions.

- File: `COS30018_Research_Archive_2026-10-26.zip`
- Size: 899,392,526 bytes
- SHA-256: `3d2715a6d1c577c0d9eba4110f4d30faecc800b83a725064814b9314517f4476`
- Files: 17,768; every archived entry was reopened and checked against its recorded SHA-256.
- Cloud upload: confirmed by Paarth on 26 October 2026. The local checksum verification is independent of that upload confirmation.

## Contents

The archive includes original source files, documents, experiment checkpoints, training histories, prediction rows, crops, manifests, source snapshots, verification records and collected photos. Downloadable MNIST/EMNIST datasets, virtual environments and Git internals are excluded.

GitHub keeps the application, eight deployable checkpoints, research scripts, reports and selected results. The larger `evidence/`, supplementary experiment outputs and downloadable MNIST/EMNIST datasets remain locally available but are excluded from future commits. Collected handwriting photos are now also included under `data/` in GitHub; their archived originals remain preserved.

## Restore and reproduce

1. Download the ZIP and verify its SHA-256 with `Get-FileHash -Algorithm SHA256`.
2. Extract into a separate folder. Inspect `ARCHIVE_MANIFEST.json` for exact paths and hashes.
3. For historical reproduction, use the archived source tree with its matching frozen evidence. Do not overwrite the current organised repository with old source files.
4. To use current research commands, restore needed `data/`, `evidence/` and supplementary `experiments/` paths into this checkout, preserving their relative locations. Create an environment using the current requirements.
5. Current research commands run as `python -m research.<name>`. Older reports show historical invocation names and source hashes. Exact-byte historical verification must use the corresponding archived sources.
6. Use a new output directory for new runs. Keep development and final-test results separate.

Do not retune a model on the final evaluation images and then claim those images provide an independent test. Preserve valid failures and the case-sensitive labels.
