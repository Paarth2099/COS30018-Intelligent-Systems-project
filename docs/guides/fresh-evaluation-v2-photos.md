# Fresh evaluation v2: photo instructions

> **SUPERSEDED DRAFT — DO NOT COLLECT 180 PHOTOS.** The latest alternative discussed was 15 photos from Paarth and 15 from one outside writer. Collection was deferred until GPU results. Both augmentation arms have now failed the adoption rule, so production is unchanged and this follow-up collection is ON HOLD. Retain these earlier assignments only as planning history. See [GPU review](../reports/gpu-augmentation-results-week11.md).

This is a proposed team collection, not a prescribed PDF sample count. Collect **180 new images: 30 each from six writers**. Four writers are group members; invite two adults whose writing has not been used by this project. Use anonymous W05/W06 IDs, obtain permission for coursework use, and do not include personal information in photos. If outside writers are unavailable, record the actual coverage and revise the protocol BEFORE scoring; do not invent writer identities.

1. Write each assigned item freshly on plain pale paper with a dark pen. Do not reuse earlier handwriting or photograph it again. Keep letters separate, on one line, in natural handwriting. Do not use cursive, punctuation or multiple lines.
2. Take one photograph per item, straight-on, with the full item and a little paper margin visible. Keep it readable; no desk edges or fingers covering writing. Do not edit, crop, enhance, or threshold the photos. Keep originals. Avoid systematic large shadows; natural ordinary variation is welcome. This assesses framed handwriting, not full-page document detection.
3. Use the exact filename pattern below. A real PNG/JPEG extension is fine; do not rename a JPEG into a PNG. Keep every valid photo, including hard handwriting. Replace only unreadable/corrupt/wrong-item captures BEFORE predictions and log why. Do not retake an image because recognition failed.
4. Each writer checks every reference, including case and leading zeros, before submitting. Record actual date, conditions and any ambiguity. A second teammate checks the manifest and image correspondence; if a glyph is ambiguous, keep its intended label with an ambiguity note.
5. **Do not try these photos in the GUI.** Store them in `data/final_evaluation_v2/raw/`. We will validate and hash-lock the completed manifest after GPU/development decisions and before any prediction. Leave placeholder confirmation/date/hash fields unset until actually checked.
6. After freezing a chosen version, compare frozen v1 and v2 on these SAME fresh images once. Report known and new writers separately, all four collections separately, exact case-sensitive accuracy and CER, and structural segmentation review. The old 120 photos remain the v1 evaluation; they cannot be recast as fresh v2 evidence.

Example: `data/final_evaluation_v2/raw/W01/strings/W01_strings_01.jpg` contains exactly `Ab7q`.

Collection planning must not be influenced by model predictions. Do not give the writer model outputs or ask them to make handwriting resemble MNIST. The mixed strings deliberately cover dots, case and similar-looking characters. This is still a small convenience sample; two new writers do not establish population-wide accuracy.

## Paarth (W01; known writer)

30 newly written photos: one item per photo. Write the exact case shown.

| Folder | Values in file-number order |
|---|---|
| `single_digits` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9` |
| `numbers` | `007`, `28`, `463`, `1905`, `82614` |
| `single_letters` | `A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `J` |
| `strings` | `Ab7q`, `iJ24`, `O0l1`, `Uv8x`, `Room6` |

## Manula (W02; known writer)

30 newly written photos: one item per photo. Write the exact case shown.

| Folder | Values in file-number order |
|---|---|
| `single_digits` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9` |
| `numbers` | `013`, `57`, `604`, `2381`, `97026` |
| `single_letters` | `K`, `L`, `M`, `N`, `O`, `P`, `Q`, `R`, `S`, `T` |
| `strings` | `Cd3r`, `jI65`, `S5s2`, `Wx9v`, `Hello4` |

## Haresh (W03; known writer)

30 newly written photos: one item per photo. Write the exact case shown.

| Folder | Values in file-number order |
|---|---|
| `single_digits` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9` |
| `numbers` | `009`, `36`, `725`, `8402`, `16359` |
| `single_letters` | `U`, `V`, `W`, `X`, `Y`, `Z`, `a`, `b`, `c`, `d` |
| `strings` | `Ef8s`, `Ji36`, `B8b0`, `Yz4u`, `Gate2` |

## Saniru (W04; known writer)

30 newly written photos: one item per photo. Write the exact case shown.

| Folder | Values in file-number order |
|---|---|
| `single_digits` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9` |
| `numbers` | `041`, `82`, `396`, `5074`, `28163` |
| `single_letters` | `e`, `f`, `g`, `h`, `i`, `j`, `k`, `l`, `m`, `n` |
| `strings` | `Gh2t`, `Ij79`, `Z2z5`, `Qr6n`, `Code3` |

## Outside writer A (W05; new writer)

30 newly written photos: one item per photo. Write the exact case shown.

| Folder | Values in file-number order |
|---|---|
| `single_digits` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9` |
| `numbers` | `005`, `64`, `817`, `3690`, `45278` |
| `single_letters` | `o`, `p`, `q`, `r`, `s`, `t`, `u`, `v`, `w`, `x` |
| `strings` | `Mn5p`, `ij48`, `O0o1`, `Tu7w`, `Desk9` |

## Outside writer B (W06; new writer)

30 newly written photos: one item per photo. Write the exact case shown.

| Folder | Values in file-number order |
|---|---|
| `single_digits` | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9` |
| `numbers` | `026`, `93`, `548`, `1706`, `83524` |
| `single_letters` | `y`, `z`, `A`, `B`, `C`, `D`, `E`, `F`, `G`, `H` |
| `strings` | `Rs6k`, `jI82`, `S5s0`, `Vx3y`, `Lab4` |
