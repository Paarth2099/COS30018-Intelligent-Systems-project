"""Build the final evaluation report from saved outputs, without inference."""
from pathlib import Path
import json
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/final_evaluation/run_01'
def load(name):return json.loads((OUT/name).read_text())
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(v) for v in row)+' |' for row in rows])+'\n\n'
def main():
    rows=load('predictions.json');summary=load('summary.json');config=load('frozen_configuration.json')
    assert load('verification.json')['status']=='PASS' and load('independent_verification.json')['status']=='PASS'
    primary=[s for s in summary if s['collection']!='auxiliary_emnist_digits' and s['method']!='projection' and s['model']==('cnn' if s['collection'] in ['single_digits','numbers'] else 'resnet18')]
    text='''# Week 11: final fresh-photo evaluation

Haresh's final model evaluation uses 120 new photographs from the four team writers. Paarth confirmed, on behalf of the four writers, the assigned labels, tight camera framing (not post-capture cropping), and that none had previously been tested. No training, tuning or default-model changes were performed on these final results.

The selected defaults were fixed before inference: CNN for numbers and ResNet18 for alphanumeric inputs. Saniru's second-computer acceptance was reported complete, and Paarth authorised proceeding while screenshots/environment records await upload. This report does not claim independent verification of that acceptance.

## Selected-system results

'''
    text+=table(['Collection','Selected model','Correct / total','Exact accuracy','Character error rate'],[[s['collection'],s['model'],f"{s['correct']}/{s['count']}",f"{s['accuracy']:.2%}",f"{s['cer']:.2%}"] for s in primary])
    text+='''Each collection has its own denominator; no combined overall accuracy is presented. Letter case and leading zeros are scored exactly. Single-input accuracy requires one correct character, so extra/missing outputs fail. CER is total edit distance divided by total reference characters; lower is better and it can exceed 100%. Errors remain in the denominator with empty-output scoring.

## Paired four-model comparison

Models receive exactly the same saved ordered crops within each collection. All single-letter photos use the GUI's grouped-component route, so segmentation can fail even for a single character. This differs from the older development isolated-character comparison's direct preparation route and prevents treating those percentages as directly controlled before/after measurements.

'''
    text+=table(['Collection','Model','Correct / total','Accuracy','CER'],[[s['collection'],s['model'],f"{s['correct']}/{s['count']}",f"{s['accuracy']:.2%}",f"{s['cer']:.2%}"] for s in summary if s['method']!='projection' and s['collection']!='auxiliary_emnist_digits'])
    text+='''The comparison describes performance under the predeclared experiment. A model that outperforms the selected default here is not automatically promoted using this test set.

## Segmentation alternative

Projection is compared with components (numbers) or grouped components (strings), holding the selected model fixed. These are whole-pipeline recognition scores, not manual segmentation accuracy. Manula must still review complete character crops, dot grouping, reading order and split/merge failures. Crop counts alone cannot establish segmentation success.

'''
    text+=table(['Collection','Method','Model','Correct / total','CER'],[[s['collection'],s['method'],s['model'],f"{s['correct']}/{s['count']}",f"{s['cer']:.2%}"] for s in summary if s['collection'] in ['numbers','strings'] and (s['method']=='projection' or s['model']==('cnn' if s['collection']=='numbers' else 'resnet18'))])
    text+='## Auxiliary: digits with the 62-class alphanumeric models\n\nThese reuse the 40 single-digit photographs; they are not 40 additional independent samples or part of the primary 40-letter score.\n\n'
    text+=table(['Model','Correct / 40','Accuracy','CER'],[[s['model'],s['correct'],f"{s['accuracy']:.2%}",f"{s['cer']:.2%}"] for s in summary if s['collection']=='auxiliary_emnist_digits'])
    text+='## Per-writer selected-system results\n\n'
    writer_rows=[]
    for writer,groups in load('by_writer.json').items():
        for s in groups:
            if s['collection']!='auxiliary_emnist_digits' and s['method']!='projection' and s['model']==('cnn' if s['collection'] in ['single_digits','numbers'] else 'resnet18'):
                writer_rows.append([writer,s['collection'],f"{s['correct']}/{s['count']}",f"{s['cer']:.2%}"])
    text+=table(['Writer','Collection','Correct / total','CER'],writer_rows)
    text+='## Timing\n\nCPU inference with four threads; models and preparation routes warmed on development photos. Each configuration has five repeated model-only and end-to-end calls per photo, with model order rotated. End-to-end means preprocessing, segmentation and inference; image loading, model loading, exports and GUI interaction are excluded. Repeats are timing observations, not new samples. Collection median and p95 below are calculated from each photo\'s median duration.\n\n'
    text+=table(['Collection','Model','Median end-to-end ms','p95 end-to-end ms'],[[s['collection'],s['model'],f"{s['median_end_to_end_ms']:.2f}" if s['median_end_to_end_ms'] is not None else 'unavailable',f"{s['p95_end_to_end_ms']:.2f}" if s['p95_end_to_end_ms'] is not None else 'unavailable'] for s in primary])
    text+=f"Environment: {config['platform']}; Python {config['python']}; PyTorch {config['torch']}. Full raw timing repeats and model-only medians are retained in predictions.json.\n\n"
    selected=[r for r in rows if r['collection']!='auxiliary_emnist_digits' and r['method']!='projection' and r['model']==('cnn' if r['task']=='mnist' else 'resnet18')]
    errors=[r for r in selected if not r['correct']]
    text+='## All selected-system errors\n\nThese are all failed selected-system samples, not a favourable subset. Crop counts and predictions are observations, not a diagnosis of the underlying cause. Paarth and Manula should use the saved crops/annotations for their respective reviews.\n\n'
    text+=table(['Sample','Reference','Prediction','Crops','Evidence'],[[r['sample_id'],f"`{r['reference']}`",f"`{r['prediction']}`" if r['prediction'] else '(empty)',r['crop_count'],f"[annotation](../../evidence/final_evaluation/run_01/samples/{r['sample_id']}/{r['task']}_{r['method']}/segmentation.png)"] for r in errors])
    text+=f"Runtime errors across all comparison rows: {sum(bool(r['error']) for r in rows)}.\n\n"
    text+='''## Limitations and evidence boundaries

- Four known team writers, only 120 photographs and deliberately selected content. This is new-input evaluation from known writers, not unseen-writer generalisation or a representative population study. Class coverage is sparse; per-class confusion counts are descriptive.
- Tight framing is team-reported camera framing. Newness/untested status and exact intended case are team attestations, not independently observed capture events. Individual writing dates remain unspecified.
- Human-assigned case can be visually ambiguous. Preserve the confirmed ground truth and disclose this limitation rather than relabel from model output.
- The planned light-stroke and close-spacing conditions were not verified as controlled interventions; no causal robustness claim follows from their labels.
- No hyperparameter search, repeated-seed retraining or new official benchmark evaluation was performed here. The existing Week 11 report retains the benchmark/training evidence.
- A separate teammate label inspection was not claimed; the recorded evidence is assistant visual review plus Paarth's collective confirmation.
- Manual segmentation and preprocessing error attribution, acceptance evidence upload, full team report and video remain outstanding. This is Haresh's component report, not the complete submission.

## Reproducibility and validation

Sixteen saved development predictions were reproduced before first final-photo inference. The locked manifest and frozen configuration were written before scoring. The code state is the recorded base Git commit plus exact archived runtime/evaluator source hashes; evaluation files were uncommitted and no new commit is invented. All 120 input hashes, eight checkpoint hashes and source hashes remained unchanged after the run.

There are 680 scored rows: 480 primary four-model comparisons, 40 projection comparisons and 160 auxiliary alphanumeric-digit comparisons. Independent verification recalculated every edit distance and exact match, collection metrics and timing medians, checked array shape/range, and verified archived sources and input/checkpoint hashes.

- [Frozen configuration](../../evidence/final_evaluation/run_01/frozen_configuration.json)
- [Locked manifest](../../evidence/final_evaluation/run_01/locked_manifest.json)
- [Development reproduction](../../evidence/final_evaluation/run_01/development_reproduction.json)
- [All predictions and timing repeats](../../evidence/final_evaluation/run_01/predictions.json)
- [Summary](../../evidence/final_evaluation/run_01/summary.json)
- [Per-writer metrics](../../evidence/final_evaluation/run_01/by_writer.json)
- [Confusion counts](../../evidence/final_evaluation/run_01/confusion_counts.json)
- [Paired model changes](../../evidence/final_evaluation/run_01/paired_changes.json)
- [Run integrity](../../evidence/final_evaluation/run_01/verification.json)
- [Independent verification](../../evidence/final_evaluation/run_01/independent_verification.json)

`evaluate_final_photos.py` refuses to overwrite run_01. Do not rerun final inference for tuning. To check saved evidence without recognition, run `python verify_final_results.py`. To regenerate this report from saved results, run `python report_final_photos.py`.
'''
    (ROOT/'docs/reports/final-evaluation-week11.md').write_text(text,encoding='utf-8')
    print('Report generated; selected results:')
    print(json.dumps(primary,indent=2))
if __name__=='__main__':main()
