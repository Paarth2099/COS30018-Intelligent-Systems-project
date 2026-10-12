"""Generate the development comparison report from saved experiment results."""
import csv
import json
from pathlib import Path


def main():
    base=Path('evidence/character_preprocessing')
    summaries=[json.loads((base/f'run_{i:02d}/summary.json').read_text()) for i in range(1,5)]
    final=summaries[-1]
    with (base/'run_04/breakdown.csv').open() as stream:breakdown=list(csv.DictReader(stream))
    with (base/'run_04/results.csv').open() as stream:rows=list(csv.DictReader(stream))
    lines=['# Paarth: character-photo preprocessing experiment','',
        'Component owner: Paarth. Code, experiment execution and report preparation used Codex assistance. Record actual review, testing and learning in individual worklogs.','',
        '## Outcome','',
        '**Selected development configuration:** local lighting correction, conservative speck removal, Otsu representation, aspect-preserving resize/padding and peak ink normalisation. The original digit pipeline and Haresh checkpoints were not changed.','',
        'The final fixed-threshold and Otsu variants each recognised **53/80 characters (66.25%)**, compared with **42/80 (52.5%)** for the original Otsu baseline: 11 more correct, or 13.75 percentage points. The declared tie preference retains Otsu; this is not evidence that Otsu is superior to fixed thresholding.','',
        'These photos were inspected and reused for development decisions. This is **development accuracy**, not untouched final-test accuracy or complete-string performance.','',
        '## Inputs and controls','',
        '- 80 single-character photographs: 20 each from Paarth, Haresh, Manula and Saniru.',
        '- 20 distinct labels, four examples each: `0 1 2 5 8 A B I O Z a b g i j l o q t z`. This does not cover all 62 model classes.',
        '- All images were used in every run; no hard example was dropped, replaced or assigned a label based on its prediction.',
        '- Paarth confirmed that the file originally named upper-J was intended as lowercase j. Its name was corrected before the first experiment. Two underscore separators were standardised; image bytes were not changed.',
        '- Labels come from explicit digit/upper/lower filenames; they are used only for scoring. The preparation and prediction functions never receive the reference label.',
        '- Same frozen validation-selected CNN checkpoint and same input SHA-256 hashes across all four runs. No model retraining or EMNIST test-set tuning.',
        f"- Duplicate byte-identical photo groups: {len(final['duplicate_hash_groups'])}. Different encodings of the same photo would not be detected by a byte hash.",'',
        '## Controlled development sequence','',
        '| Run | Change from preceding run | Grayscale | Fixed | Otsu | Photos with warnings |',
        '|---|---|---:|---:|---:|---:|']
    changes=['Original common-crop baseline','Add local lighting correction','Add conservative speck filtering','Add post-resize ink peak normalisation']
    for i,(summary,change) in enumerate(zip(summaries,changes),1):
        cells=[f"{summary['methods'][m]['correct']}/80 ({summary['methods'][m]['accuracy']:.2%})" for m in ('grayscale','fixed','otsu')]
        lines.append(f"| {i} | {change} | "+' | '.join(cells)+f" | {summary['methods']['otsu']['warning_images']} |")
    lines+=['','Lighting correction improved grayscale results and reduced warning flags; it did not increase the Otsu score by itself. Speck filtering improved crop localisation, and normalisation countered faint downsampled strokes. The before/after numbers are observed changes on these same development photos, not statistical evidence of generalisation.','',
        'All three representations share the localisation mask and geometry. Fixed uses intensity <=128; Otsu uses an image-derived cutoff; grayscale keeps intensities. Resampling leaves intermediate pixel values even for thresholded representations.','',
        '## Final results by writer','',
        '| Writer | Grayscale | Fixed | Otsu |','|---|---:|---:|---:|']
    for writer in ('Haresh','Manula','Paarth','Saniru'):
        group=[r for r in breakdown if r['group_type']=='writer' and r['group']==writer]
        lines.append('| '+writer+' | '+' | '.join(f"{r['correct']}/{r['total']} ({float(r['accuracy']):.1%})" for r in group)+' |')
    lines+=['','These small samples combine writer style, pen and photography differences. They are not rankings of team members or an isolated test of handwriting quality.','',
        '## Remaining errors','',
        'No photo raised a preprocessing exception in the four runs. That only means an array was produced; it does not mean every crop was correct. The selected pipeline misclassified 27/80 photos.','',
        '| Reference | Prediction | Count |','|---|---|---:|']
    for item in final['selected_confusions']:
        lines.append(f"| `{item['true']}` | `{item['predicted']}` | {item['count']} |")
    lines+=['','Uppercase O and lowercase o were often read as zero; g and q were confused with digits, and lowercase z with uppercase Z. Isolated character shape can be ambiguous, but these counts alone do not establish the cause of each error.','',
        'Three photos retain background/clipping warning flags:','']
    for row in rows:
        if row['method']=='otsu' and row['warnings']:
            lines.append(f"- `{row['file']}`: true `{row['label']}`, predicted `{row['prediction']}`. {row['warnings']}.")
    lines+=['','Saniru\'s digit 1 has an oversized/offset crop despite speck removal; the warning and preview retain this failure. Warning flags are advisory, not a complete detector of poor preparation. Some intended lowercase forms resemble uppercase handwriting; supplied labels were retained.','',
        '## Verification and evidence','',
        '- 33 software tests passed, including nine new preprocessing tests: dots and small detached parts, lighting gradient, thin strokes, EXIF rotation, blank/low-contrast input, clipping warnings, aspect ratio, non-mutation and label validation.',
        '- Each run includes source snapshots, checkpoint hash, input manifest, 240 method/sample results, 80 per-photo comparisons, per-writer overviews, diagnostic JSON and prepared arrays.',
        '- Inference uses an uncalibrated softmax score; it is not a probability that the prediction is correct.',
        '- Timing in results.csv is local CPU timing. Shared preparation computes all three variants once per photo; inference is one prepared image. File loading and evidence export are excluded.','',
        '## Limits and next stage','',
        'The code supports one dark character on pale paper. Whole-string splitting, spaces, joined handwriting and multiline pages are not implemented here. Tiny genuine components below max(3 pixels, 0.5% of the largest component area) can still be removed; inspect i/j previews when integrating new photo conditions. Strong shadows and texture remain limitations.','',
        'Manula should supply complete character crops with a small pale-paper margin, preserving each dot with its stem, and order crops left to right. Saniru then integrates the character result schema into the GUI. Reserve fresh photos/strings for later final system evaluation; do not present these 80 development examples as an independent test.','',
        'See [PAARTH_EXTENSION.md](PAARTH_EXTENSION.md) for code locations, commands and the handover contract.','']
    Path('PAARTH_PREPROCESSING_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('PAARTH_PREPROCESSING_RESULTS.md')


if __name__=='__main__':main()
