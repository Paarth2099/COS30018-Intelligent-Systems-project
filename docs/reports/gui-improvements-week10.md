# Week 10 GUI improvements and local verification

**Outcome: the Week 9 off-screen-window defect is resolved on the tested computer. Updated recognition results display and export correctly.** Verified on 20 October 2026, Australia/Sydney.

## Problem and implementation

The previous minimum window was 1020 × 860 Tk units. At the tested effective 150% scaling, the client extended below the 1920 × 1200 physical screen. Logical widget-bound checks passed even though some content was physically inaccessible.

The application now chooses an initial size from the Windows work area in the same virtualised coordinate system as Tk, allowing space for the window frame. It keeps the existing readable layout inside a two-axis scrolling viewport. The status line is outside that viewport and stays visible. Mouse wheel scrolls vertically, Shift + wheel horizontally, and keyboard focus brings active controls into view. The crop breakdown retains its own scrollbar for long results. The normal minimum is 640 × 400 Tk units, reduced further only if the initial available area is smaller.

This is a scrollable layout, not a claim that all content is simultaneously visible in a small window. Resizing does not change inputs, models, recognition settings or results. No model, preprocessing or segmentation code was changed in this GUI work.

## Physical display checks

The verifier uses physical Windows window/client rectangles and compares the entire outer frame with the physical work area. Screenshots were also inspected visually.

| Tested size in Tk units | Physical outer window bounds (left, top, right, bottom) | Result |
|---|---|---|
| Startup: 1240 × 720 | 24, 24, 1908, 1163 | Fits the 0, 0, 1920, 1200 work area |
| Smaller: 800 × 500 | 24, 24, 1248, 833 | Fits; controls and breakdown reached by scrolling |
| Minimum: 640 × 400 | 24, 24, 1008, 683 | Fits; recognition/export reachable and status visible |

The effective scale is 1.5 physical pixels per Tk coordinate. These are three window sizes on one display configuration, not three independently tested resolutions or operating-system scaling settings. Primary-display startup is covered; multi-monitor layouts and other platforms remain unverified.

## Workflow and integration checks

Real Tk widgets, callbacks, worker threads and saved models were used. File-picker responses and error-dialog handling were supplied by the test, so this does not constitute manual native-dialog acceptance.

| Check | Observed result |
|---|---|
| Number and alphanumeric photo loading | PASS |
| Generate numbers and mixed characters | PASS; request text remains separate from actual prediction |
| CNN/MLP and character separation switching | PASS; stale results cleared and export disabled until rerun |
| Generated `007` | CNN `007`, MLP `002`; the mistake is preserved |
| Generated `00Ai` | CNN `00Ai`; case and leading zeros retained |
| Long generated `007007007007` | Twelve crops; actual CNN output `007007007002` displayed/exported; final crop reachable |
| Paarth's improved `240` example | CNN `240` displayed and exported |
| Manula's repaired `Hello` example | Five crops; actual CNN output `Hell0` displayed and exported |
| Manula's repaired `Bill` example | Four crops; CNN `Bill` displayed and exported |
| Blank input | Empty recognition result and valid export |
| Invalid image, invalid text, missing character sample | Errors handled; no fabricated result |
| Missing model files | Simulated with a separate nonexistent model root; error handled, export disabled, normal service restored |
| Cancel selection | Existing image retained |
| Export | Saved prediction/settings match the actual result; previous exports retained |
| Responsiveness | Tk heartbeat continues while worker runs; settings disabled during processing |
| Mouse/keyboard navigation | Vertical wheel, Shift-wheel and focus visibility assertions pass |
| Number/character model caching | Four independent mode/model entries retained |

These examples verify GUI integration, not an accuracy benchmark. In particular, successful display of `Hell0` does not mean that `Hello` was recognised correctly.

## Environment and regression evidence

- Windows 11 build 26200; Python 3.12.10; Tcl 8.6.15.
- PyTorch 2.6.0+cpu, NumPy 2.2.3 and Pillow 11.1.0.
- Launched using `run_gui.ps1 -Verify` and the existing sibling virtual-environment fallback. This is not a fresh installation.
- **63 regression tests passed in 2.242 seconds.** `pip check` reported no broken requirements.
- The final GUI run passed all asserted workflow and physical-display checks. The app closed after each run and subsequent runs launched successfully.

The first Week 10 run passed the initial checks. An intermediate strengthened test failed because it attempted to focus the Choose Image button while its Open Photo tab was inactive. The test was corrected to focus the visible folder entry on the active Generate tab. The final run then passed both focus checks without a manual visibility helper. Earlier runs are retained as an audit trail; use the final run below for acceptance evidence.

## Evidence

Authoritative final run: `evidence/gui_week10/smoke_20261020_075611_735600`.

- [Final verification, environment, predictions and physical bounds](../../evidence/gui_week10/smoke_20261020_075611_735600/verification.json)
- [Normal window](../../evidence/gui_week10/smoke_20261020_075611_735600/alphanumeric_photo.png)
- [Small-window controls](../../evidence/gui_week10/smoke_20261020_075611_735600/small_window_controls.png)
- [Small-window crop breakdown](../../evidence/gui_week10/smoke_20261020_075611_735600/small_window_crops.png)
- [Minimum window](../../evidence/gui_week10/smoke_20261020_075611_735600/minimum_window.png)
- [Long number and final crops](../../evidence/gui_week10/smoke_20261020_075611_735600/long_number_breakdown.png)
- [Repaired Hello through the GUI](../../evidence/gui_week10/smoke_20261020_075611_735600/Haresh_03_text-Hello.png)
- [Repaired Bill through the GUI](../../evidence/gui_week10/smoke_20261020_075611_735600/Haresh_06_text-Bill.png)
- [Full regression log](../../evidence/gui_week10/regression_tests_week10.txt)
- [Source snapshots and audit manifest](../../evidence/gui_week10/smoke_20261020_075611_735600/audit_manifest.json)

The final run folder also contains exported inputs, detected regions, prepared arrays, predictions and settings in its `gui_*` subfolders.

## Remaining acceptance boundaries

A clean installation, manual native file-picker/error-dialog interaction, another computer and other operating-system scaling settings remain pending. Earlier cross-computer operation of the original app does not establish acceptance of these new changes. Use the [Week 9 acceptance checklist](gui-review-week09.md#remaining-acceptance-checks) with the new commit, recording tester, date, environment, expected/actual result and evidence. Fresh final recognition evaluation remains separate from GUI verification.

## Reproduction

```powershell
.\run_gui.ps1 -Verify
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The second command assumes a project-local virtual environment has been installed. The launcher supports the existing sibling fallback on this computer. Each GUI verification creates a new timestamped evidence folder; inspect screenshots as well as the PASS result.
