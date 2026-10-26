> Recorded study: results and original source hashes are retained. Detailed outputs are in the [research archive](../guides/research-archive.md); current commands use the `research` package.

# Week 11: integration of four recognition architectures

Saniru's integration connects Haresh's evaluated ResNet-18 and MobileNetV3-Small
checkpoints to the existing GUI. Numbers defaults to the baseline CNN;
Alphanumeric defaults to ResNet-18. All four architectures are selectable in
both modes. This completes the local implementation and automated desktop checks,
not the fresh final evaluation or updated second-computer acceptance.

## Why these defaults

The [model comparison](model-comparison-week11.md) found that ResNet-18 improved
single-digit development photos (19/20 versus 16/20), but reduced complete-number
accuracy (17/20 versus 19/20). The number system therefore retains CNN. ResNet-18
improved isolated alphanumerics from 53/80 to 54/80 and complete strings from 8/40
to 12/40, supporting its selection for Alphanumeric. These are development results;
the integration test does not constitute another accuracy study.

## Implementation

- `app.py`: four architecture identifiers in the selector; mode changes select
  the evaluated default and clear the old result/export state.
- `gui_core.py`: `resolve_checkpoint()` selects the correct digit or character
  run, checks new-run task/model metadata and verifies every checkpoint's SHA-256.
  The service maintains separate caches for all eight mode/model combinations.
- Existing prediction classes load the saved checkpoints on CPU. No GPU or CUDA
  installation is required for GUI use.
- Exports retain actual predictions, active architecture, task-specific checkpoint
  path/hash, prepared crops and source provenance. Requested generated text remains
  separate from the model output.
- Missing/corrupt models produce visible errors and leave export disabled. There
  is no silent fallback to another architecture.

Preprocessing, segmentation, trained weights and Haresh's saved comparison evidence
were not altered by this integration.

## Verification

The unit suite ran 72 tests: **71 passed and one CUDA training test was skipped**
because this computer uses CPU PyTorch. The skipped test is not a failed GUI check.
New tests cover mode-specific checkpoint routing, corruption, mismatched task
metadata and missing-model behaviour.

The real Tk smoke run passed at
`evidence/gui_week11/smoke_20261024_153126_595602/`. It exercised real saved model
inference, button callbacks and exports, with native file/error dialogs mocked.

Checks included:

- All four models in both modes matched the corresponding Week 11 saved predictions
  for the selected number and string photos; eight caches stayed separate.
- ResNet-18 recognised the generated `00Ai` example as `00Ai`; display and export
  retained the actual output and requested input separately.
- Mode/model switches cleared stale output and reset defaults correctly.
- Missing and corrupt new-model fixtures produced errors; restoring the real
  service allowed recognition to recover without touching real checkpoints.
- Existing number photos `420` and `240`, generated `007`, character segmentation
  alternatives, blank input, invalid files/text, missing source characters and
  cancelled selection retained their checked behaviour.
- Worker responsiveness, disabled controls, small-window scrolling, keyboard focus,
  twelve-digit crop access and physical screen fitting passed.

The `Room7` photo still produced `R00M7` under ResNet-18. This is a recognition
limitation, not a GUI test failure: the GUI correctly reproduced and exported the
model's actual prediction. Most development strings remain imperfect.

## Evidence

- [Unit test output (archive: evidence/gui_week11/unit_tests.txt)](../guides/research-archive.md)
- [Desktop verification and source hashes](../../results/gui/verification.json)
- [Alphanumeric default](../assets/week11_alphanumeric_default.png)
- [Numbers default](../assets/week11_numbers_default.png)
- [MobileNet selection and cleared result (archive: evidence/gui_week11/smoke_20261024_153126_595602/week11_alphanumeric_model_selector.png)](../guides/research-archive.md)
- [ResNet generated-text result](../assets/week11_resnet_generated.png)

The run folder also retains individual exports and the tested GUI source snapshots.
The corrupt fixture is intentionally invalid test data, not a deployable model.

## Environment and remaining acceptance

Windows 11 build 26200, Python 3.12.10, Torch 2.6.0+cpu, Tcl 8.6.15;
reported screen/work area 1920x1080. Normal and smaller window sizes were tested
on this environment. Native dialog interaction was mocked; clean installation
and another computer were not exercised by this run.

Before final submission, run the updated GUI on the second computer, manually
load a photo and export through native dialogs, then freeze the selected system
for fresh final-photo evaluation. Report and video preparation remain separate.

```powershell
.\run_gui.ps1
.\run_gui.ps1 -Verify
```
