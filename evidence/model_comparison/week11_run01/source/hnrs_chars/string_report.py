"""Build a measured segmentation report after final visual review."""
import csv
import json
from pathlib import Path
from .data import sha256
from .segmentation import DEFAULT_METHOD


def read_csv(path):
    with path.open(encoding='utf-8-sig') as stream:return list(csv.DictReader(stream))


def main():
    base=Path('evidence/character_segmentation');final=base/'run_04'
    runs=[json.loads((base/f'run_{i:02d}/summary.json').read_text()) for i in range(1,5)]
    review=read_csv(final/'visual_review.csv');rows=read_csv(final/'results.csv')
    if len(review)!=80 or any(r['visual_status'] not in ('pass','fail','uncertain') for r in review):
        raise ValueError('Finish visual review before generating a final selection/report')
    methods=runs[-1]['methods']
    visual={m:{s:sum(r['method']==m and r['visual_status']==s for r in review) for s in ('pass','fail','uncertain')} for m in methods}
    selected=max(methods,key=lambda m:(visual[m]['pass'],-methods[m]['character_error_rate'],m=='projection'))
    if selected!=DEFAULT_METHOD:raise ValueError('Public default does not match reviewed selection')
    decision={'selected_method':selected,'run':'run_04','visual_review':visual,
              'rule':'More visually acceptable full-image separations; then lower CER; then projection on a full tie',
              'review_scope':'Codex inspection of annotated originals and ordered crops; not human ground-truth boxes or independent evaluation',
              'summary_sha256':sha256(final/'summary.json'),'review_sha256':sha256(final/'visual_review.csv'),
              'segmentation_source_sha256':sha256(Path('hnrs_chars/segmentation.py'))}
    (base/'selection.json').write_text(json.dumps(decision,indent=2)+'\n')
    lines=['# Alphanumeric string segmentation experiment','',
        'Comparison of projection and grouped connected-component segmentation for single-line alphanumeric images.','',
        '## Outcome','',f'**Selected method: `{selected}`.** It is connected-component analysis with geometric grouping, not a trained neural segmentation model. It feeds complete character crops to Paarth\'s unchanged preparation and Haresh\'s frozen CNN.','',
        '| Final method | Correct crop count | Visually acceptable separation | Exact strings | Character error rate |',
        '|---|---:|---:|---:|---:|']
    for name,r in methods.items():
        lines.append(f"| {name} | {r['count_correct']}/40 | {visual[name]['pass']}/40 | {r['exact_correct']}/40 ({r['exact_accuracy']:.1%}) | {r['edit_distance']}/160 ({r['character_error_rate']:.2%}) |")
    lines+=['','All scores describe the same inspected **development set**, not a fresh final test. Exact-string accuracy is low even where segmentation is acceptable: a single wrong character makes the whole string incorrect. Do not describe crop-count agreement as recognition accuracy.','',
        '## Dataset and label controls','',
        '- 40 photographs, 10 from each writer; 160 reference characters in total.',
        '- Strings: `Ab12`, `Room7`, `Hello`, `ij12`, `jig`, `Bill`, `007Ab`, `O0o`, `Il1`, `Zz25`.',
        '- Paarth\'s seventh filename said `00Ab`, but the photographed text visibly contains `007Ab`. The filename was corrected before scoring; original image bytes were unchanged. The initial review sheet preserves the old filename.',
        '- Intended case labels were otherwise retained, including ambiguous uppercase-looking lowercase handwriting. No label was altered to match a prediction.',
        '- All photos were retained in every run. Reference text is used only after segmentation and prediction for evaluation.',
        f"- Byte-identical duplicate groups: {len(runs[-1]['duplicate_hash_groups'])}. This does not detect re-encoded duplicates.",
        '- The same CNN checkpoint and Paarth preprocessing source were held fixed across all runs. No retraining, spelling correction, expected-length hint or dictionary lookup was used.','',
        '## Method','',
        'Both methods use EXIF-corrected grayscale, bounded working resolution, the lighting correction used by Paarth, Otsu foreground separation and conservative speck removal. Small morphological closing repairs tiny gaps in the connectivity mask; it does not paint artificial strokes into the grayscale crops.','',
        '**Projection:** locate consecutive occupied image-column intervals. Vertically overlapping column ranges can merge neighbouring characters; horizontally offset dots can create extra intervals.','',
        '**Grouped components:** identify substantial character bodies, cluster nearby small dot fragments and attach plausible upper parts to nearby stems. Prefer horizontal alignment and a body below a dot over a tall neighbouring bounding box. Retain unassigned small parts with warnings rather than silently dropping them. Sort final groups left to right.','',
        'Each crop retains a pale margin and suppresses labelled ink from other groups. The pipeline calls `prepare_character()` on each crop and passes the stacked [N,28,28] float array to `CharacterPredictor.predict_text()`. Blank images produce an empty batch and string.','',
        '## Development history','',
        '| Run | Change | Projection count / exact / CER | Grouped count / exact / CER |',
        '|---|---|---|---|']
    descriptions=['Initial shared mask and dot grouping','Small-gap closing and revised short-body/fragment handling',
                  'Fragmented-dot clustering and compact-dot/body distinction','Prefer aligned stems over neighbouring boxes for dot assignment']
    for i,(run,description) in enumerate(zip(runs,descriptions),1):
        cells=[]
        for method in ('projection','grouped_components'):
            r=run['methods'][method];cells.append(f"{r['count_correct']}/40; {r['exact_correct']}/40; {r['character_error_rate']:.2%}")
        lines.append(f'| {i} | {description} | '+ ' | '.join(cells)+' |')
    lines+=['','These are sequential development changes, sometimes involving multiple geometric rules. They are not independent experiments that isolate each rule\'s causal effect. Full outputs and source snapshots for every run are retained.','',
        '## Visual review and remaining failures','',
        'Final visual review inspected all 80 method/image combinations using annotated originals and ordered crop strips. A pass requires one crop per visible character in reading order with appropriate dot grouping, not a correct model prediction. This qualitative preview assessment is not independently annotated ground truth or a bounding-box IoU measurement.','',
        'Run 3 illustrates why counts alone are insufficient: grouped components matched 38/40 counts, but two of those attached an i dot to a neighbour. Its visual pass count was 36/40; that finding motivated run 4.','',
        '| Image | Method | Visual issue |','|---|---|---|']
    for r in review:
        if r['visual_status']!='pass':lines.append(f"| {r['file']} | {r['method']} | {r['notes'].split(';')[0]} |")
    lines+=['','Model errors remain after acceptable segmentation, including `O/o/0`, `I/l/1`, and case changes. Photo ink style, size normalisation and EMNIST-to-photo differences may contribute; this comparison does not isolate those causes. Failures and warnings remain visible in the evidence.','',
        '## Per-writer results for selected method','',
        '| Writer | Count matches | Exact strings | Character errors / reference characters |',
        '|---|---:|---:|---:|']
    for writer in ('Haresh','Manula','Paarth','Saniru'):
        subset=[r for r in rows if r['writer']==writer and r['method']==selected]
        lines.append(f"| {writer} | {sum(int(r['count_correct']) for r in subset)}/10 | {sum(int(r['exact_correct']) for r in subset)}/10 | {sum(int(r['edit_distance']) for r in subset)}/{sum(len(r['reference']) for r in subset)} |")
    lines+=['','These small samples mix handwriting, pen and photography differences; they do not rank team members.','',
        '## Metrics and timing','',
        '- Count agreement: number of predicted crops equals reference string length; no label is supplied to the segmenter.',
        '- Exact-string accuracy: case-sensitive full equality, preserving leading zeros.',
        '- Character error rate (CER): total Levenshtein insertions, deletions and substitutions divided by 160 reference characters. It is not clipped and can exceed 100% when there are many insertions.',
        '- Visual separation: full-image pass/fail review of crops, independent of whether the classifier guessed correctly.','',
        '| Method | Mean segmentation | Mean preparation | Mean inference |','|---|---:|---:|---:|']
    for name,r in methods.items():
        lines.append(f"| {name} | {r['mean_segmentation_ms']:.2f} ms | {r['mean_preparation_ms']:.2f} ms | {r['mean_inference_ms']:.2f} ms |")
    lines+=['','Single local CPU run with a warmed predictor; excludes loading and evidence export. Timings are not cross-machine guarantees.','',
        '## Verification and evidence','',
        '- At the segmentation experiment checkpoint, 47 software tests passed: the previous 33 plus 14 new segmentation/interface tests.',
        '- Tests cover dots, fragmented dots, short stems, misleading neighbour boxes, small pen gaps, overlapping crop boxes, ordering, blank images, EXIF rotation and preservation of case/zeros.',
        '- Existing digit GUI, digit algorithms, Paarth preparation and trained models remain unchanged.',
        '- Every run has source/checkpoint/input hashes, per-image CSVs, masks, boxes, crop images, prepared arrays and predictions.',
        '- Final default and review provenance are in `evidence/character_segmentation/selection.json`.','',
        'The component supports a single line of separated characters. Touching/cursive writing, multiline pages, word-space inference, severe shadows and very small detached parts remain limitations. The selected pipeline is integrated into the GUI. Fresh end-to-end testing and further error analysis remain necessary.','',
        'See [the segmentation guide](../guides/string-segmentation.md) for commands and the recognition interface.','']
    Path('docs/reports').mkdir(parents=True,exist_ok=True)
    Path('docs/reports/string-segmentation-results.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(decision,indent=2))


if __name__=='__main__':main()
