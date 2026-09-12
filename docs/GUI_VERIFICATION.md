# GUI verification record

Owner: Saniru. This records assisted implementation and execution, not a claim
about independently completed work or hours spent.

## Delivered behaviour

- Load an image file and preview it without requiring a labelled filename.
- Compose a number from labelled single-digit images in a selected folder.
- Preserve requested leading zeros and compute a separate model prediction.
- Select CNN/MLP, components/projection, and Otsu/fixed/grayscale.
- Display detected regions, complete prediction, prepared crops and model scores.
- Cache saved models and perform processing outside the Tk event loop.
- Invalidate old results when settings or the input change.
- Export the actual input, masks, crops, settings, prediction and provenance.
- Report blank input, invalid number requests and missing digit samples.

## Verified checks

All 18 tests in `python -m unittest discover -s tests -v` passed. The saved log is
`evidence/gui/unit_tests.txt`. This includes the 13 prior model/segmentation tests
and five new generation/catalog/export tests. Source photos remain unchanged;
multi-digit source files are excluded from the generator's digit catalog.

The final desktop smoke test is recorded in
`evidence/gui/smoke_20260911_231001_309363/verification.json`.
It used real Tk widgets, their callbacks and the actual saved models; file-dialog
choices were substituted by the test. Manual interaction with native file
pickers still needs the user's check. Screenshots were visually inspected at
1180x860 and the minimum supported 1020x860 window size.

| Check | Recorded outcome |
|---|---|
| Open Manula 420, CNN + components + Otsu | 420 |
| Switch to projection on the same image | 47, the retained segmentation limitation |
| Generate 007 from the team digit folder, CNN | 007 |
| Switch to MLP on the generated image | 002, an actual model error |
| Change model or segmentation | Old result cleared; recognition reruns |
| Export loaded/generated route | Separate folders with input/provenance/prediction |
| Cancel file selection | Existing input retained |
| Invalid generation request abc | Error message, no crash |
| Uniform white input | No digits detected |
| Minimum window size | Controls and status visible |

The generation instruction is never supplied as the recogniser's answer. For
example, the MLP result 002 is shown even though 007 was requested. Model scores
are labelled as uncalibrated, not guaranteed correctness. These few examples are
functional checks and cannot be reported as general recognition accuracy.

## Issue corrected during development

The visual refresh introduces a persistent input/settings sidebar, white preview
cards, a blue result panel and individual digit tiles. Typography, spacing and
button states are consistent throughout. The default window is 1240x860 and the
minimum remains 1020x860. Both input routes and existing processing callbacks
passed the desktop smoke test after the redesign; screenshots in the folder
above show the refreshed appearance. The recognition algorithms are unchanged.

Pillow's optional ImageTk extension could not load its DLL from the long Windows
project path. The app now uses Tk's native PNG decoder with in-memory encoded
images. No dependency version change or move of the checkout was required.
An initial layout clipped the lower controls; the final layout was adjusted and
checked at its supported minimum size. Earlier smoke folders are intermediate
attempts; use the final verified folder above for the current UI evidence.

## Remaining boundaries

The application supports local digits and inherits the current segmentation and
model limitations. It does not recognise text, joined digits reliably, paragraphs
or arbitrary backgrounds. The window needs a display that can accommodate its
1020x860 client area and title bar. Cross-platform and clean-machine verification
are not established by this Windows test.

Next, Saniru should manually use both input routes and export a result, review
the code and record actual learning in the worklog. The team should share the
local changes through its Git workflow and test another member's installation.
