"""Validate and report the Week 9 paired development experiment."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.run/'summary.json').read_text())
    rows = read(args.run/'results.csv'); pairs = read(args.run/'paired_results.csv')
    assert len(rows)==320 and len(pairs)==160
    assert len({r['file'] for r in pairs})==160
    for key, record in summary['checkpoints'].items():
        path = Path(record['file'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
        selected = json.loads((path.parent/'selection.json').read_text())
        assert next(r['checkpoint_sha256'] for r in selected['models'] if r['model']==path.stem)==record['sha256']
    for i, pair in enumerate(pairs):
        assert hashlib.sha256(Path(pair['file']).read_bytes()).hexdigest()==pair['sha256']
        arrays=np.load(args.run/'prepared'/f'{i:03d}.npy')
        assert arrays.shape==(int(pair['crop_count']),28,28)
        assert arrays.dtype==np.float32 and np.isfinite(arrays).all()
        assert not arrays.size or (arrays.min()>=0 and arrays.max()<=1)
        assert (args.run/pair['preview']).exists()
        matching=[r for r in rows if r['file']==pair['file']]
        assert {r['model'] for r in matching}=={'mlp','cnn'} and len(matching)==2
        for r in matching:
            assert r['prediction']==pair[r['model']]
            assert int(r['correct'])==int(r['prediction']==r['reference'])
    # Compare every CNN prediction with the saved pre-existing experiments.
    prior={}
    for path, method in [('evidence/segmentation/team_run_02/results.csv','components'),
                         ('evidence/character_preprocessing/run_04/results.csv','otsu'),
                         ('evidence/character_segmentation/run_04/results.csv','grouped_components')]:
        for r in read(path):
            if r['method']==method:
                prior[Path(r['file'].replace('\\','/')).name]=r['prediction']
    assert all(prior[Path(p['file']).name]==p['cnn'] for p in pairs)
    text=['# Week 9: real-photo MLP/CNN comparison', '',
          'Paired evaluation of the existing saved classifiers on all 160 development photographs. No models, preprocessing or segmentation algorithms were changed.', '',
          '## Method', '',
          'The digit models use connected components and Otsu preparation. Individual characters use the existing local lighting correction, speck removal, Otsu and ink normalisation. Strings use grouped components and the same character preparation. Each photo is prepared once and exactly the same array is passed to both models. Reference labels are used only for scoring; case and leading zeros are significant.', '',
          'All four checkpoints match the hashes in their original selection records. All 160 CNN predictions reproduce the previous experiment predictions. All photos remain in the comparison. A failed run stops rather than silently dropping a photo.', '',
          '## Recognition results', '',
          '| Collection | Model | Correct / photos | Accuracy | CER | Mean inference ms | Mean pipeline ms |',
          '|---|---|---:|---:|---:|---:|---:|']
    for r in summary['results']:
        subset=[x for x in rows if x['collection']==r['collection'] and x['model']==r['model']]
        assert sum(int(x['correct']) for x in subset)==r['correct']
        assert abs(sum(int(x['edit_distance']) for x in subset)/sum(len(x['reference']) for x in subset)-r['character_error_rate'])<1e-12
        text.append(f"| {r['collection']} | {r['model']} | {r['correct']}/{r['images']} | {r['accuracy']:.2%} | {r['character_error_rate']:.2%} | {r['mean_inference_ms']:.3f} | {r['mean_pipeline_ms']:.2f} |")
    text += ['', 'Single-image accuracy means exact equality: for numbers/strings every character must match. CER is total Levenshtein edits divided by reference character count, not one minus exact-string accuracy.', '',
             '## Paired outcomes', '', '| Collection | Both correct | Only MLP correct | Only CNN correct | Both wrong |', '|---|---:|---:|---:|---:|']
    for collection in ('single_digits','numbers','characters','strings'):
        group=[p for p in pairs if p['collection']==collection]
        counts=Counter((int(p['mlp_correct']),int(p['cnn_correct'])) for p in group)
        text.append(f"| {collection} | {counts[1,1]} | {counts[1,0]} | {counts[0,1]} | {counts[0,0]} |")
    text += ['', '## Representative errors and disagreements', '',
             'Examples below are selected deterministically by file order, not as an accuracy sample. Every input has a preview in the evidence directory.', '',
             '| Collection | Reference | MLP | CNN | Review priority | Preview |', '|---|---|---|---|---|---|']
    chosen=[]
    for collection in ('single_digits','numbers','characters','strings'):
        group=[p for p in pairs if p['collection']==collection]
        for condition in (lambda p:p['mlp_correct']=='1' and p['cnn_correct']=='0',
                          lambda p:p['mlp_correct']=='0' and p['cnn_correct']=='1',
                          lambda p:p['mlp_correct']=='0' and p['cnn_correct']=='0'):
            chosen += [p for p in group if condition(p)][:2]
    chosen += [p for p in pairs if p['review_category']=='count_mismatch' and p not in chosen]
    for p in chosen:
        link='../../'+args.run.as_posix()+'/'+p['preview']
        text.append(f"| {p['collection']} | `{p['reference']}` | `{p['mlp']}` | `{p['cnn']}` | {p['review_category']} | [Image]({link}) |")
    text += ['', '## Frequent isolated-character confusions', '', '| Model | Reference | Prediction | Count |', '|---|---|---|---:|']
    for model in ('mlp','cnn'):
        counts=Counter((r['reference'],r['prediction']) for r in rows if r['collection']=='characters' and r['model']==model and r['correct']=='0')
        text += [f'| {model} | `{a}` | `{b}` | {n} |' for (a,b),n in counts.most_common(8)]
    text += ['', '## Interpretation and next investigation', '',
             'The CNN has higher exact accuracy in all four collections: 75% versus 55% on single digits, 90% versus 45% on numbers, 66.25% versus 53.75% on isolated characters, and 17.5% versus 7.5% on strings. String CER is also lower for CNN (35.625% versus 49.375%). This supports retaining the CNN baseline, while the low string accuracy remains a major limitation. MLP inference is faster in this local run, but shared preparation dominates total processing time.', '',
             'Visual inspection of selected original/prepared previews identified these concrete investigation cases:', '',
             '- Preview 113, Saniru lower-i: the prepared image is dominated by paper texture rather than a clear dot and stem. Both models fail. This is direct evidence of a preparation problem worth investigating.',
             '- Preview 104, Saniru digit-1: the digit occupies a small, offset area in the prepared tile, with a stray mark near the top. Both models fail. Investigate localisation/normalisation; the preview does not prove that preparation alone caused the errors.',
             '- Preview 122, Haresh Hello: H becomes two crops, producing six crops for five characters. Both models receive the same incorrect separation.',
             '- Preview 125, Haresh Bill: the i dot and stem remain separate, producing five crops for four characters. This requires segmentation investigation.', '',
             'Use the per-collection results rather than combining unrelated digit and character datasets into a single accuracy. A benchmark advantage does not guarantee a better prediction for every photograph. Retain the existing validation-selected CNN as the baseline; this experiment does not retrain or automatically change GUI defaults.', '',
             'A crop-count mismatch establishes that the number of supplied regions differs from the reference length. Matching counts alone do not establish correct separation. Both classifiers failing on identical crops does not establish whether preparation or the model is responsible. Review originals and prepared crops before attributing causes.', '',
             'Preprocessing investigation should inspect faint/thin strokes, oversized background crops and clipping, retaining before/after examples. Segmentation investigation should review count mismatches and misplaced dots. Model investigation should examine case and shape confusions after crop quality is assessed. Do not relabel ambiguous examples to match predictions.', '',
             '## Timing and limitations', '', summary['timing'], '',
             'Inference is warmed, CPU-only with four Torch threads; model execution order alternates and the median of five runs is used for each image/model by default. Shared preparation is timed once, so total pipeline figures are estimates from shared preparation plus inference, not GUI latency. No statistical speed claim is made.', '',
             'These are reused development photos from four writers, not an independent final test. The 80 character photos cover only 20 of 62 labels. String examples repeat the same ten texts across writers. No causal or generalisation claim follows from this small collection. Frozen checkpoints and all original inputs are unchanged. Reserve fresh handwriting for the final evaluation.', '',
             '## Reproduction and evidence', '', 'From the repository root with dependencies installed:', '', '```powershell',
             'python compare_photo_models.py --output evidence/photo_model_comparison/week09_run02',
             'python report_photo_models.py --run evidence/photo_model_comparison/week09_run02 --output docs/reports/photo-model-comparison-week09.md', '```', '',
             f'- [Summary JSON](../../{args.run.as_posix()}/summary.json)',
             f'- [All 320 model predictions](../../{args.run.as_posix()}/results.csv)',
             f'- [160 paired photo results](../../{args.run.as_posix()}/paired_results.csv)',
             f'- [Input manifest](../../{args.run.as_posix()}/manifest.csv)', '',
             'The run also retains all prepared arrays, original/crop previews and source snapshots. The report generator validates checkpoint/input hashes, array contracts, pair coverage, summary counts and exact agreement with historical CNN predictions before writing this report.']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('Verified 160 inputs, 320 paired predictions, 4 checkpoint hashes, array contracts and all historical CNN predictions.')
    print(args.output)


if __name__=='__main__':
    main()
