# System verification — 26 October 2026

## Outcome

Core application workflows passed on this Windows computer, including a clean package containing only published files plus the verification-fixture fix. Production recognition code and trained models were not changed. This is functional/regression verification, not a guarantee of recognition accuracy or assessment marks.

## Checks and evidence

| Check | Outcome | Evidence |
|---|---|---|
| Unit suite and eight checkpoint hashes | 76 passed; one CUDA-only test skipped | `automated-tests.txt` |
| Installed dependency compatibility | No broken requirements | `dependencies.txt` |
| Real Tk GUI, original working tree | Passed | `gui-smoke.txt` |
| Clean-package GUI after fixture fix | Passed on rerun; earlier window-boundary failure retained | `clean-package-gui-recheck.txt`, `clean-package-gui.txt` |
| Three additional launches | All windows inside physical work area | `window-launch-checks.json` |
| Full final-photo pipeline | 120 photos; 200 preparations/crop boundaries matched; all 680 predictions reproduced using eight models | `full-pipeline-reproduction.json` |
| Final evidence integrity | 120 photo hashes, eight checkpoints and 680 prediction records checked | `final-evidence.txt` |
| Additional preprocessing study | 640 predictions/array contracts verified across two runs | `improvement-evidence.txt` |
| Eight GPU-trained augmentation checkpoints | Hashes/selection checked; 640 development predictions reproduced on CPU | `gpu-results-verification.txt` |

## GUI coverage

Both recognition modes and all four architectures; loading photographs; generation with leading zeros and case-sensitive characters; both segmentation methods; crop previews; result exports and checkpoint hashes; mode/model changes invalidating stale results; long-number crop scrolling; small-window scrolling and keyboard focus; worker responsiveness and disabled busy controls; blank images; invalid files/text; missing character samples; cancelled file dialogs; missing/corrupt checkpoints and recovery.

The test invokes actual Tk controls and real models but supplies native file-dialog responses programmatically. Native picker interaction was not manually tested. The clean package uses the existing verified Python environment; this is not a new operating system installation or another computer's acceptance test. CUDA execution was unavailable here.

## Fix made

The GUI verification script read two historical reference files from the excluded `evidence/` folder. It now reads a compact `tests/fixtures/gui_regression.json` containing the original recorded predictions and source hashes. A fresh GitHub package no longer needs the research archive to run that GUI regression test. No predictions were changed to match newly produced output.

One clean-package run failed the physical window-boundary assertion. The complete rerun and three additional launches passed. Diagnostic messages now record the actual window/work-area bounds if it recurs. The original failed log remains available; no layout-code fix is claimed.

## Recognition limitations

The frozen selected system remains 37/40 correct for individual digits, 16/20 exact numbers, 29/40 individual letters/digits and 4/20 exact mixed strings. Functional success means the application works and reproduces its recorded outputs, not that all handwriting is recognised correctly. Already analysed evaluation images are not a fresh independent test for subsequent tuning.

## Re-run

From the repository root, run the following with Python dependencies installed:

```powershell
.\.venv\Scripts\python.exe scripts/check_project.py
.\.venv\Scripts\python.exe -m pip check
.\run_gui.ps1 -Verify
```

The deeper research-verification commands require the local archived evidence. Detailed GUI exports remain in the local timestamped evidence folders. These verification changes and results accompany the GUI regression fixture fix in version control.
