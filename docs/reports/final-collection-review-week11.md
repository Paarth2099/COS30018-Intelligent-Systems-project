> Recorded study: results and original source hashes are retained. Detailed outputs are in the [research archive](../guides/research-archive.md); current commands use the `research` package.

# Week 11: Paarth final-photo collection review

Review date: 26 October 2026 (Australia/Sydney).

## Responsibility

Paarth organises and verifies the fresh-photo evaluation collection, preserves labels and originals, records provenance and hashes, and later analyses preprocessing failures after evaluation. This record covers collection verification only. No recognition, segmentation or preprocessing experiment was run on these final photos.

## Completed checks

- All 120 assigned files are present, with exactly one supported image per sample ID and no unexpected files in the raw folders.
- Each writer has 30 photos: 10 digits, 5 numbers, 10 letters and 5 strings.
- All 120 images decode successfully. Their dimensions, byte hashes and decoded-pixel hashes were recorded.
- No byte-identical or decoded-pixel-identical duplicates were found within the collection or against 160 images in `data/own_characters`, `data/own_strings` and `data/segmentation_photos`.
- A weak dHash screen (distance at most 4) returned no near-duplicate candidates. This cannot prove that handwriting was newly written, or exclude every crop/rephotograph of an old sample. Dataset benchmark files and photos elsewhere were not searched.
- All 16 contact sheets (120 originals) were visually reviewed. All glyphs appear present; no image was rejected for a difficult handwriting style. Saniru's single-digit 5 has softer focus but remains readable and is retained.
- The existing intended labels preserve leading zeros and letter case. Four string labels require writer confirmation; no labels were silently changed.
- Original files were not cropped, enhanced, renamed or replaced by this review. Final readback confirmed every original still matches its recorded SHA-256.

## Initial label questions (resolved by writer confirmation)

| File stem | Planned label | Confirmation needed |
|---|---|---|
| Paarth_strings_01 | AbC2 | Third character is small: uppercase C or lowercase c? |
| Paarth_strings_04 | jKl8 | Second character is lowercase-sized: uppercase K or lowercase k? |
| Manula_strings_03 | StU5 | Third character is lowercase-sized: uppercase U or lowercase u? |
| Saniru_strings_04 | hIj6 | Third character appears undotted: lowercase j or uppercase J? |

These are ambiguous appearance flags, not automatic relabel decisions. Preserve the intended label if the writer confirms it. Do not replace legitimate difficult handwriting simply to help a model.

## Provenance and collection limitations

The supplied images are tightly framed around the writing. The collection instructions requested preserved originals without manual cropping. Ask whether this was camera framing or post-capture cropping. If cropped, obtain the original captures and document their relationship before lock, or explicitly amend the evaluation scope before predictions; do not describe manually cropped inputs as an untouched camera-photo evaluation.

Writer confirmation that samples are newly written and untested is pending, as are writing dates and a named human second check of labels. A filesystem timestamp is not proof of when text was written. The assistant's visual review is recorded as such, not attributed to a teammate.

Planned difficult conditions were not verified as quantitatively controlled: for example, not every fifth number is visibly lighter and string spacing varies. Retain the actual samples and record these as planned conditions rather than claiming a controlled faint-stroke/spacing experiment.

The four writers also supplied development data. Results will describe new samples from known writers, not independent unseen-writer performance. There are no recognition results yet, and no basis for a preprocessing accuracy claim at this stage.

## Evidence

- [Integrity audit (archive: evidence/final_evaluation/collection_review/integrity_audit.json)](../guides/research-archive.md): image checks, dimensions, hashes and duplicate scan scope.
- [Visual review (archive: evidence/final_evaluation/collection_review/visual_review.json)](../guides/research-archive.md): per-image observations and pending questions.
- [Input hash record (archive: evidence/final_evaluation/collection_review/input_hashes.json)](../guides/research-archive.md): exact 120 supplied file versions.
- [Updated manifest (archive: evidence/final_evaluation/protocol/collection_manifest.json)](../guides/research-archive.md): files and intended labels, with final verification fields still pending.
- [Collection verification script](../../research/verify_final_collection.py): integrity checks and review-sheet generation without inference.

### Review sheets

- Haresh: [single_digits (archive: evidence/final_evaluation/collection_review/Haresh_single_digits.jpg)](../guides/research-archive.md), [single_letters (archive: evidence/final_evaluation/collection_review/Haresh_single_letters.jpg)](../guides/research-archive.md), [numbers (archive: evidence/final_evaluation/collection_review/Haresh_numbers.jpg)](../guides/research-archive.md), [strings (archive: evidence/final_evaluation/collection_review/Haresh_strings.jpg)](../guides/research-archive.md)
- Paarth: [single_digits (archive: evidence/final_evaluation/collection_review/Paarth_single_digits.jpg)](../guides/research-archive.md), [single_letters (archive: evidence/final_evaluation/collection_review/Paarth_single_letters.jpg)](../guides/research-archive.md), [numbers (archive: evidence/final_evaluation/collection_review/Paarth_numbers.jpg)](../guides/research-archive.md), [strings (archive: evidence/final_evaluation/collection_review/Paarth_strings.jpg)](../guides/research-archive.md)
- Manula: [single_digits (archive: evidence/final_evaluation/collection_review/Manula_single_digits.jpg)](../guides/research-archive.md), [single_letters (archive: evidence/final_evaluation/collection_review/Manula_single_letters.jpg)](../guides/research-archive.md), [numbers (archive: evidence/final_evaluation/collection_review/Manula_numbers.jpg)](../guides/research-archive.md), [strings (archive: evidence/final_evaluation/collection_review/Manula_strings.jpg)](../guides/research-archive.md)
- Saniru: [single_digits (archive: evidence/final_evaluation/collection_review/Saniru_single_digits.jpg)](../guides/research-archive.md), [single_letters (archive: evidence/final_evaluation/collection_review/Saniru_single_letters.jpg)](../guides/research-archive.md), [numbers (archive: evidence/final_evaluation/collection_review/Saniru_numbers.jpg)](../guides/research-archive.md), [strings (archive: evidence/final_evaluation/collection_review/Saniru_strings.jpg)](../guides/research-archive.md)

## Remaining gate

Do not mark the dataset locked yet. Resolve the four label questions and provenance, obtain the human second review, then save a separate locked manifest with its hash. Preserve this review evidence. Saniru's second-computer acceptance and the final code/configuration freeze remain separate prerequisites before Haresh scores the collection.

Paarth's later preprocessing error analysis depends on those final results; it is not claimed complete here.

## Confirmation update — 26 October 2026

Paarth confirmed: “we wrote it exactly how it was asked” and “all new photos photographed this way”. All assigned labels are therefore retained, including `AbC2`, `jKl8`, `StU5` and `hIj6`. The visual ambiguity notes above are preserved as review history, not unresolved label changes. The reported capture method is tight camera framing, not post-capture cropping; replacements are not needed on that basis.

All 120 original hashes were checked again and remain unchanged. The manifest now records confirmed labels and the source of confirmation. This is a collective team attestation, not a claim that a named second teammate independently inspected every image. Writing dates were not supplied and remain blank.

Evidence added: [team confirmation (archive: evidence/final_evaluation/collection_review/user_confirmation.json)](../guides/research-archive.md), [verified collection snapshot (archive: evidence/final_evaluation/protocol/verified_collection_snapshot.json)](../guides/research-archive.md) and its [SHA-256 (archive: evidence/final_evaluation/protocol/verified_collection_snapshot.sha256)](../guides/research-archive.md). The pre-confirmation manifest is preserved in the review folder.

The reply did not explicitly answer whether any images had already been tested in the GUI. Untouched-test status remains unconfirmed, so the final evaluation gate and dataset lock remain pending. No recognition was run in this review. Second-computer GUI acceptance and the code/configuration freeze remain separate tasks.

## Evaluation completion update

Paarth subsequently confirmed, on behalf of the team, that none of the photos had been tested. Run 01 locked and evaluated all 120 images after Paarth authorised proceeding with acceptance evidence upload pending. The old unresolved-status statements above are historical review stages. See [final evaluation](final-evaluation-week11.md) for verified results. The frozen manifest retains the exact pre-inference state; current collection status is recorded separately.
