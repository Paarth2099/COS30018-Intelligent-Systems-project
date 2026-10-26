> Recorded study: results and original source hashes are retained. Detailed outputs are in the [research archive](../guides/research-archive.md); current commands use the `research` package.

# Week 9 local GUI verification — 19 October 2026

**Outcome: functional checks passed; display-fit issue remains open.** This is automated local verification on one Windows computer, not cross-computer or manual acceptance testing.

## Environment and execution

- Windows 11 build 26200, Python 3.12.10, Tcl 8.6.15.
- PyTorch 2.6.0, torchvision 0.21.0, NumPy 2.2.3, Pillow 11.1.0; installed versions match requirements.txt.
- Launched through `run_gui.ps1 -Verify`, using the existing sibling baseline virtual environment because this checkout has no `.venv`. This confirms the local fallback, not a fresh installation.
- Real Tk widgets, callbacks, worker threads and saved models were exercised. File-dialog return values and error-dialog handling were supplied by the test; the native file picker was not manually exercised.
- All 52 regression tests passed in 2.112 seconds. Existing application and recognition code were unchanged.

## Workflow checks

| Check | Result | Evidence/limit |
|---|---|---|
| Start GUI and load saved models | PASS | Existing configured environment; four mode/model cache entries |
| Load number photo | PASS | CNN reads the 420 photo as 420 |
| Generate number from samples | PASS | Requested 007; CNN predicts 007, MLP predicts 002; actual prediction is retained |
| Load alphanumeric photo | PASS | Room7 yields R00M7 with CNN/grouped components, matching the saved pipeline |
| Generate alphanumeric image | PASS | 00Ai source composition preserves case and leading zeros; CNN predicts 00Ai |
| Switch character model | PASS | Room7 with MLP yields e00n7; selected model is recorded correctly |
| Switch character segmentation | PASS | Projection produces four crops and R00A; grouped components produces five crops |
| Invalidate stale results | PASS | Mode/model/method changes clear existing results and disable export |
| Invalid image/text and missing sample | PASS | Invalid image, a space-containing string and unavailable character C produce errors |
| Cancel file selection | PASS | Existing image retained when the supplied dialog response is empty |
| Blank input | PASS | No characters detected; empty result export handled |
| Export | PASS | Number, character and blank results retain actual prediction/settings; generation request remains separate |
| Responsiveness | PASS within test scope | Tk heartbeat advances while worker recognises; controls disabled during processing |
| Controls inside logical window | PASS | Existing widget-bound assertions pass |
| Whole window fits physical screen | **FAIL** | Client bottom is y=1359; available screen ends at y=1200 |
| Clean installation / other computer | NOT TESTED | Requires a separate environment and machine |

These eight recognition examples are workflow checks, not an accuracy benchmark. The known recognition mistakes were preserved. No accuracy improvement is claimed.

## Open issue: display scaling and window height

The minimum window is 1020x860 Tk units. At this machine's effective 1.5 physical-pixel scale, its client area measures 1530x1290 physical pixels. At the tested position, it spans y=69 to y=1359 on a 1920x1200 display. The bottom is off-screen, including some result/status content. A control being inside the application window does not establish that it is inside the visible screen.

The first run used Tk coordinates directly with ImageGrab, producing incorrectly cropped screenshots. The verification script now queries physical Windows client coordinates for screenshots using a temporary thread DPI context. This changes test capture only, not the app's layout or process DPI behaviour. The repeat run confirms functional passes and exposes the actual screen-fit issue. Black pixels below the visible screen in the capture are not an application panel.

Week 10 should investigate screen-aware window sizing and a scrollable or adaptive layout so recognition, export, breakdown and status remain accessible at common scaling settings. Test the physical work-area bounds as well as widget bounds. Avoid treating a lower Windows scaling setting as proof that the application supports the current setting.

## Evidence

Authoritative repeat run:

- [Functional verification and source hashes (archive: evidence/gui_extension/smoke_20261019_163319_961004/verification.json)](../guides/research-archive.md)
- [Environment and physical screen measurements (archive: evidence/gui_extension/smoke_20261019_163319_961004/local_environment.json)](../guides/research-archive.md)
- [Minimum-window capture (archive: evidence/gui_extension/smoke_20261019_163319_961004/minimum_window.png)](../guides/research-archive.md)
- [Alphanumeric photo capture (archive: evidence/gui_extension/smoke_20261019_163319_961004/alphanumeric_photo.png)](../guides/research-archive.md)
- [Generated characters (archive: evidence/gui_extension/smoke_20261019_163319_961004/generated_characters.png)](../guides/research-archive.md)
- [Number photo capture (archive: evidence/gui_extension/smoke_20261019_163319_961004/numbers_regression.png)](../guides/research-archive.md)

The earlier `smoke_20261019_163202_921842` run is retained with passing functional results, but its screenshots are cropped by the old DPI-coordinate calculation. Do not use those captures to claim full layout visibility. The functional PASS value in verification.json covers its assertions; this report adds the physical-display failure found during visual inspection.

## Remaining acceptance checks

On another computer, record tester, date, commit, OS, Python/package versions, screen resolution/scaling, steps, expected versus actual behaviour and screenshot/export evidence. Mark a case passed only after running it.

1. Create a fresh virtual environment from the installation guide and start the app without the sibling-environment fallback.
2. Copy the shared sample photos; verify both modes, both acquisition routes, model changes and character separation changes.
3. Use the native file picker, cancel a selection, try a bad file and a missing sample, and inspect the displayed error messages.
4. Export a number and a string result; check that the saved prediction and settings match the display.
5. Check controls, complete results and status at the machine's normal scaling and supported window sizes; record the current screen-fit issue if reproduced.
6. Observe responsiveness during recognition and close/reopen normally.

This local run does not cover every malformed input, checkpoint-removal scenario, long-running interaction or display configuration. Cross-computer testing and clean-install verification remain pending.

## Reproduction

From the repository root:

```powershell
.\run_gui.ps1 -Verify
# With a project virtual environment installed:
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The GUI verifier creates a new evidence folder. Review the screenshots and physical bounds as well as the assertion result before declaring usability passed.
