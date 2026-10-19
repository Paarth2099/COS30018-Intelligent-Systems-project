"""Generate the segmentation investigation report and reviewed evidence index."""
import csv
import json
from pathlib import Path
import numpy as np
from hnrs_chars.photo_experiment import write_csv


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def main():
    independent=Path('evidence/segmentation_experiment/week10_run01')
    combined=Path('evidence/segmentation_experiment/week10_combined01')
    summaries=read(independent/'summary.csv')+[r for r in read(combined/'summary.csv') if r['variant']=='combined']
    rows=read(combined/'predictions.csv');visual=[];paired=[]
    geometry=json.loads((combined/'geometry.json').read_text())
    assert [r['index'] for r in geometry if r['variant']=='combined' and r['changed']]==[122,125]
    for r in [r for r in rows if r['variant']=='combined' and r['model']=='cnn']:
        index=int(r['index']);folder=combined/f'{index:03d}'
        if index not in (122,125):
            np.testing.assert_array_equal(np.load(folder/'baseline/prepared.npy'),np.load(folder/'combined/prepared.npy'))
        note=('Broken H fragments now share one complete crop; both l characters remain separate.' if index==122 else
              'Short i stem and its dot now share one complete crop; B and both l characters remain separate.' if index==125 else
              'Complete ordered crops with no obvious neighbouring-character contamination; dot assignments retained where present. Geometry and arrays equal baseline.')
        visual.append(dict(index=index,file=r['file'],reference=r['reference'],visual_separation='acceptable',observation=note,sheet=f'combined_sheet_{(index-120)//5+1:02d}.png',basis='Qualitative inspection of annotated source, raw crops and prepared crops on eight contact sheets; not independently annotated or blinded'))
    for r in [r for r in rows if r['variant']=='combined']:
        b=next(b for b in rows if b['index']==r['index'] and b['model']==r['model'] and b['variant']=='baseline')
        paired.append(dict(index=r['index'],model=r['model'],reference=r['reference'],baseline=b['prediction'],combined=r['prediction'],gain=int(r['correct'])>int(b['correct']),regression=int(r['correct'])<int(b['correct']),edit_improvement=int(b['edits'])-int(r['edits'])))
    assert not any(r['regression'] for r in paired)
    write_csv(combined/'visual_review.csv',visual);write_csv(combined/'paired.csv',paired)
    decision=dict(adopted='combined A_body + B_stem',visual_acceptable=40,visual_review_total=40,
                  changed_geometry_indices=[122,125],other_38_arrays_identical=True,no_exact_regressions_both_models=True,
                  boundary='Development evidence only; no claim of perfect recognition or final generalisation')
    (combined/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    text=['# Week 10 segmentation investigation','',
      '**Decision: adopt bounded broken-body grouping and wider short-stem eligibility in the grouped-components method.** Both previously diagnosed failures are repaired on this development collection.','',
      '## Responsibility and question','',
      'Segmentation separates a single handwritten line into complete characters in reading order. It is geometric component analysis, not a trained segmentation model. This investigation asks whether the Week 9 broken H and detached i failures can be repaired without merging neighbouring letters or damaging existing dot groups.','',
      '## Controlled experiments','',
      '- **A: broken-body grouping.** Merge adjacent groups only when both are tall (at least 60% of the observed 80th-percentile height), horizontal box gap is at most 4% of that height, vertical overlap is at least 65% of the shorter group, one width is at most 30% and the other at least 45% of typical height, and combined width is at most typical height. A single pass merges non-overlapping pairs; narrow/narrow neighbours are excluded. Raw strokes are not drawn or filled.','- **B: short-stem eligibility.** Increase the width/height limit for a short body from 0.45 to 0.55. Retain the minimum-height checks and existing dot-assignment rule.','- **Combined:** test A+B separately after both independent comparisons meet the adoption criteria.','- **Projection comparator:** rerun the existing vertical projection method alongside the independent candidates. It does not use the grouped-component changes.','',
      'The ratios are development hypotheses informed by Week 9 diagnosis, not universal thresholds or a parameter sweep. No filename, reference label, expected count or model prediction enters the segmentation algorithm. Reference strings are used only for evaluation.','',
      'All variants use the same 40 development strings, fixed original CNN/MLP checkpoints, unchanged character preprocessing, foreground mask and crop-preparation settings. Baseline boxes, arrays and both-model predictions were reproduced for all 40 strings. Component IDs are retained exactly once and outputs remain left to right. Number regression checks use Paarth\'s already adopted Week 10 digit normalisation, rather than the older Week 9 recognition baseline.','',
      '## Results','', '| Variant | Model | Matching count | Exact strings | CER | Images with warnings | Segmentation ms |','|---|---|---:|---:|---:|---:|---:|']
    for r in summaries:text.append(f"| {r['variant']} | {r['model']} | {r['count_correct']}/40 | {r['correct']}/40 | {float(r['cer']):.3%} | {r['warning_images']} | {float(r['mean_segmentation_ms']):.2f} |")
    text+=['','CER is total edit distance divided by the 160 reference characters. Lower is better. Timing is single-pass, fixed-order and descriptive; small differences do not establish a speed improvement. All evaluated inputs completed without processing errors.','',
      '## What changed and why it matters','',
      '| Example | Before | After | Interpretation |','|---|---|---|---|',
      '| Haresh Hello (122) | Six crops; CNN `rlell0` | Five complete crops; CNN `Hell0` | Broken H is repaired, but final o is still recognised as zero. Segmentation success does not imply exact recognition. |',
      '| Haresh Bill (125) | Five crops; CNN `BeIll` | Four complete crops; CNN `Bill` | Short stem now qualifies as a body and receives its detached dot. |','',
      'A changes only 122; B changes only 125. The combined variant changes only these two examples. The other 38 grouped outputs have identical geometry, component groups and prepared arrays to baseline. Both models have no newly incorrect full strings. CNN gains one exact string; MLP exact recognition remains 3/40, although its total character errors decrease.','',
      'Visual inspection of all 40 combined outputs across eight sheets found one complete crop per visible character, appropriate detached-dot grouping and no obvious neighbour contamination. Repeated l characters, Room7 neighbours, leading zeros and i/j controls remain separated. This gives **40/40 visually acceptable development separations**, compared with the earlier 38/40; it is a qualitative assessment, not an independently annotated IoU score. Count agreement alone was not used as proof. Projection count agreement is 32/40; no new full projection visual score is claimed.','',
      'The repaired H now emits an inspection warning; the old implementation failed silently on that input. The resolved i no longer produces unassigned-fragment warnings. Warnings are diagnostic prompts, not pass/fail evidence.','',
      '## Integration and verification','',
      'The production `hnrs_chars/segmentation.py` grouped-components defaults now enable both changes. The existing GUI character-recognition route calls this pipeline, so no GUI feature or model switch is required. Projection, digit segmentation, character preprocessing and checkpoint files remain unchanged.','',
      'The production verification regenerates all 40 string outputs, checking exact boxes, component groups, prepared arrays and all 80 CNN/MLP predictions against the combined experiment. The 20 multi-digit regression images preserve all boxes and arrays from Paarth\'s Week 10 result; CNN remains 19/20 and MLP 9/20. The full regression test log is linked below. No new manual GUI acceptance session is claimed.','',
      '## Limits and next evaluation','',
      'The changes address observed development failures and have been tested on only four writers. Tight legitimate neighbouring characters may satisfy the merge geometry on unseen inputs; unusual dots or stems may still be grouped incorrectly. These checks do not establish robustness to cursive, touching characters, punctuation or multiple lines.','',
      'Exact CNN recognition is still only 8/40 (20%) and CER is 33.125%, despite acceptable separation in this set. Case ambiguity, O/o/0 and I/l/1 confusion, preparation and classifier errors remain. Fresh final photos were not loaded and must remain separate for the final evaluation after the configuration is frozen.','',
      '## Evidence','']
    for root,file,label in [(independent,'plan.json','Independent prospective plan'),(independent,'summary.csv','Independent and projection metrics'),(combined,'plan.json','Combined plan'),(combined,'summary.csv','Combined metrics'),(combined,'predictions.csv','All combined-run predictions'),(combined,'paired.csv','Paired gains and regressions'),(combined,'visual_review.csv','40-photo qualitative review'),(combined,'geometry.json','Boxes and component groups'),(combined,'decision.json','Adoption decision'),(combined,'number_regression.csv','20-number regression results'),(combined,'verification.json','Experiment checks'),(combined,'integration_verification.json','Production checks'),(combined,'software_tests.txt','Full regression test log')]:text.append(f'- [{label}](../../{root.as_posix()}/{file})')
    for i in range(1,9):text.append(f'- [Combined review sheet {i}](../../{combined.as_posix()}/combined_sheet_{i:02d}.png)')
    text+=['','Per-image folders contain masks, raw crops, prepared arrays, segmentation metadata and previews. Baseline source snapshots and hashes are preserved. To repeat the controlled comparison after production defaults change, run the archived source from the repository root:','',
      '```powershell',f'python {independent.as_posix()}/source/experiment_segmentation_week10.py --output evidence/segmentation_experiment/week10_repeat',f'python {combined.as_posix()}/source/experiment_segmentation_week10.py --combined --output evidence/segmentation_experiment/week10_combined_repeat','python verify_segmentation_week10.py','python -m unittest discover -s tests -v','```','']
    Path('docs/reports/segmentation-experiment-week10.md').write_text('\n'.join(text),encoding='utf-8')
    print('Segmentation report and visual review saved')


if __name__=='__main__':main()
