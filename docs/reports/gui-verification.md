> Recorded study: results and original source hashes are retained. Detailed outputs are in the [research archive](../guides/research-archive.md); current commands use the `research` package.

# GUI extension verification — 13 October 2026

## Scope and result

Implemented Numbers/Alphanumeric mode switching, case-sensitive generation from character photos, integration with both saved character models and both string segmenters, ordered crop/score display, and mode-aware evidence export. The underlying character preparation, segmentation, models and checkpoints were unchanged.

**52 software tests passed** (`python -m unittest discover -s tests -v`, 2.598 seconds reported by unittest). This includes the prior 47 tests and five new acquisition/export/checkpoint tests in `tests/test_gui_characters.py`.

**Desktop integration checks passed** with real Tk widgets, real worker threads and saved CNN/MLP checkpoints. Dialog selections were supplied by the test to make local inputs repeatable; this is automated desktop testing, not a teammate's manual acceptance test.

Passing evidence: [verification JSON (archive: evidence/gui_extension/smoke_20261013_144252_167841/verification.json)](../guides/research-archive.md).

| Input / configuration | Actual displayed result | Interpretation |
|---|---|---|
| Manula `Room7`, character CNN/grouped components | `R00M7`, 5 crops | Matches the previously saved pipeline; classification errors remain |
| Same photo, character CNN/projection | `R00A`, 4 crops | Projection merges neighbouring characters |
| Same photo, character MLP/grouped components | `e00n7`, 5 crops | Model switching works; recognition is imperfect |
| Generated `00Ai`, character CNN | `00Ai`, 4 crops | Case/leading-zero composition and recognition demonstrated |
| Blank image, character CNN | No characters detected | Empty output and warning handled |
| Original Manula `420` photo, digit CNN | `420`, 3 crops | Existing photo route regression passed |
| Generated `007`, digit CNN | `007`, 3 crops | Existing generator preserves zeros |
| Generated `007`, digit MLP | `002`, 3 crops | Digit model switching works; model error retained |

The eight displayed examples are workflow checks, not an accuracy benchmark. The existing character development benchmark remains 7/40 exact strings; adding the GUI did not improve it.

## Other desktop checks

- Character and number exports retain actual predictions, mode and source provenance separately from generation requests.
- The character-photo route matches the frozen Manula pipeline output.
- The four mode/model combinations use separate predictor-cache entries.
- Mode/model/separation changes clear stale results and prevent stale exports.
- Invalid image file, invalid string, missing source character, cancelled selection and blank image are handled.
- Tk heartbeat callbacks continue while recognition runs; controls are disabled during processing.
- Export, recognition controls and status fit inside the 1020×860 minimum window at this machine's display settings.

Inspected screenshots:

- [Alphanumeric photo (archive: evidence/gui_extension/smoke_20261013_144252_167841/alphanumeric_photo.png)](../guides/research-archive.md)
- [Generated characters (archive: evidence/gui_extension/smoke_20261013_144252_167841/generated_characters.png)](../guides/research-archive.md)
- [Minimum window (archive: evidence/gui_extension/smoke_20261013_144252_167841/minimum_window.png)](../guides/research-archive.md)
- [Number regression (archive: evidence/gui_extension/smoke_20261013_144252_167841/numbers_regression.png)](../guides/research-archive.md)

The earlier `smoke_20261013_144030_221516` directory is explicitly incomplete: the test encountered a missing original digit dataset, and its screenshot exposed sidebar clipping. The 40 original digit photos were copied from the existing baseline into this checkout's ignored `data/segmentation_photos` directory, the sidebar spacing was corrected, and the desktop suite was repeated successfully. No training or development photos were relabelled during GUI work.

## Reproduction and limits

Run `run_gui.ps1 -Verify` with the models and local development photo folders present. The test makes a new evidence directory instead of overwriting earlier results. The verification JSON includes hashes of the tested GUI source and smoke script. The existing digit and character pipelines remain separate; generated-text labels are used to choose source tiles, not fed to recognition.

Verification covers one local Windows environment. Cross-machine acceptance testing remains outstanding for loading, generation, recognition, mode switching and export. Display scaling, missing local datasets and environment installation can differ. See [the user guide](../guides/gui.md) for setup.

Recognition error analysis and independent final evaluation remain outstanding. Do not treat a successful generated example or automated GUI test as proof of high accuracy on fresh handwriting.
