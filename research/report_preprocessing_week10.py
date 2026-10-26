"""Build the Week 10 preprocessing report from saved comparisons."""
import csv
import json
from pathlib import Path
from collections import Counter
from research.compare_photo_models import sha


def main():
    root=Path('evidence/preprocessing_experiment/week10_run01')
    def read(name):
        with (root/name).open() as f:return list(csv.DictReader(f))
    groups=read('summary.csv');pairs=read('paired.csv');predictions=read('predictions.csv')
    review={
        101:'Both character candidates retain broad texture contamination; original digit 2 remains unresolved.',
        104:'B removes distant noise and restores a larger centred 1; CNN changes i to 1, but MLP remains wrong (l to I).',
        113:'Both character candidates retain broad texture instead of the i; unresolved.',
        74:'B deletes the genuine detached j dot. Both models change correct lowercase j to uppercase J. Reject B.',
        96:'A changes CNN uppercase Z to lowercase z despite a retained outline; recognition regression.',
        24:'C retains the 9 geometry and restores intensity; CNN changes 7 to 9.',
        37:'C retains all three digit crops; CNN changes 140 to the correct 240.',
        4:'C brightens the existing 9 but CNN remains 7; geometry is not repaired by intensity scaling.',
        14:'C brightens the existing 9 but CNN remains 4.',
        32:'C retains the 9 outline; CNN remains wrong (4), MLP remains correct (9).',
        20:'C brightens the 0 outline without changing CNN correctness.',
        53:'Detached i dot/stem remain visible in both character candidates.',
        54:'Detached j dot and hook remain visible in both character candidates.',
        73:'Detached i dot/stem remain visible in both character candidates.',
        77:'Thin t strokes remain visible; CNN remains correct.',
        81:'Thin i and dot remain visible; CNN remains correct.',
        83:'Narrow l outline is retained; CNN remains wrong (1).',
    }
    (root/'visual_review.json').write_text(json.dumps(dict(basis='Qualitative inspection of 17 saved comparison previews, not blinded or independent annotation',observations=review),indent=2)+'\n')
    eligible={}
    for variant,collections in [('A_background',('characters','strings')),('B_foreground',('characters','strings')),('C_peak',('single_digits','numbers'))]:
        strict=False;safe=True
        for collection in collections:
            b=next(r for r in groups if (r['collection'],r['variant'],r['model'])==(collection,'baseline','cnn'))
            c=next(r for r in groups if (r['collection'],r['variant'],r['model'])==(collection,variant,'cnn'))
            safe &= float(c['accuracy'])>=float(b['accuracy']) and float(c['cer'])<=float(b['cer']) and int(c['failures'])==0
            strict |= float(c['accuracy'])>float(b['accuracy']) or float(c['cer'])<float(b['cer'])
        eligible[variant]=bool(safe and strict)
    assert eligible=={'A_background':False,'B_foreground':False,'C_peak':True}
    (root/'decision.json').write_text(json.dumps(dict(metric_gate=eligible,adopted='C_peak for Otsu digit crops only',rejected=['A_background','B_foreground'],visual_gate='C preserves geometry; B removes genuine dot',fresh_final_evaluation_pending=True),indent=2)+'\n')
    text=['# Week 10 preprocessing experiments','',
      '**Decision: adopt post-resize peak normalisation for Otsu digit crops. Retain existing character preprocessing.**','',
      '## Question and controls','',
      'Can targeted preparation changes improve recognition without sacrificing existing successes? Week 9 identified three clear isolated-character preparation defects and seven thin-stroke review cases. Three candidates were specified before scoring; no parameter sweep or label-dependent filtering was used.','',
      '| Candidate | Single change | Coverage |','|---|---|---|',
      '| A: background | Change local MaxFilter kernel from 31 to 15; retain Gaussian blur 8 and other settings | 80 characters and 40 strings |',
      '| B: foreground | Remove a surviving component only if its area is below 10% of the largest and its bounding-box gap exceeds half the largest component height | 80 characters and 40 strings |',
      '| C: stroke contrast | Divide each resized digit array by its nonzero maximum, preserving spatial support and relative intensities | 20 single digits and 20 numbers |','',
      'CNN is the primary model, MLP the secondary comparison. Both original digit and character checkpoints are fixed. Haresh\'s rejected weighted CNN is not used. Segmentation, reading order, images and supplied labels are unchanged. All 160 original prepared arrays and 320 original model predictions were reproduced before comparison. Candidate arrays, input/checkpoint/source hashes and every prediction are saved.','',
      'The prospective adoption rule requires no CNN exact-accuracy or CER regression in either relevant collection, strict improvement in at least one, no preparation failures, and visual review before integration. There is no automatic adoption from an isolated success.','',
      '## Complete comparison','',
      '| Collection | Variant | Model | Correct | Accuracy | CER | Prep ms |','|---|---|---|---:|---:|---:|---:|']
    for r in groups:text.append(f"| {r['collection']} | {r['variant']} | {r['model']} | {r['correct']}/{r['count']} | {float(r['accuracy']):.2%} | {float(r['cer']):.3%} | {float(r['mean_preparation_ms']):.2f} |")
    text+=['','CER is edit distance divided by total reference characters; lower is better. All variants completed without preparation errors. Timing is a single-pass descriptive measurement, not a speed benchmark: variants run in fixed order, baseline operations are reused for C, and regression tests ran concurrently during part of the run. No speed improvement is claimed.','',
      '## Paired gains and regressions','', '| Collection | Variant | Model | Gains | Regressions | Both correct | Both wrong |','|---|---|---|---:|---:|---:|---:|']
    for key in sorted({(r['collection'],r['variant'],r['model']) for r in pairs}):
        counts=Counter(r['outcome'] for r in pairs if (r['collection'],r['variant'],r['model'])==key)
        text.append('| '+' | '.join(key)+f" | {counts['gain']} | {counts['regression']} | {counts['both_correct']} | {counts['both_wrong']} |")
    text+=['','## Visual investigation','',
       'The following 17 comparisons were inspected, covering all priority defects, the seven thin-stroke cases, detached-dot controls, and observed gains/regressions. This qualitative review is not an independent annotation study. All 160 previews remain available.','', '| ID / comparison | Observation |','|---|---|']
    for index,note in review.items():text.append(f'| [{index:03d}](../../{root.as_posix()}/previews/{index:03d}.png) | {note} |')
    text+=['','## Interpretation and integration','',
      'A does not resolve the texture defects in 101 or 113. It loses a correct uppercase Z and increases CNN string CER. The faster observed preparation is not sufficient grounds for adoption and was not measured under benchmark controls.','',
      'B fixes the offset 1 (104) but removes Manula\'s genuine j dot (074). CNN character accuracy remains 53/80 while MLP falls from 43/80 to 42/80. A component-distance rule cannot safely distinguish every isolated dot from noise. B fails the strict-improvement gate and visual preservation check; it is retained only as rejected research code.','',
      'C improves CNN single digits from 15/20 to 16/20 and complete numbers from 18/20 to 19/20. It corrects Paarth\'s 9 (024) and Saniru\'s 240 (037), with no newly wrong examples. MLP exact accuracy and CER are unchanged, although a wrong prediction can change to another wrong prediction; see paired.csv. Normalisation restores intensity lost during downsampling; it neither thickens strokes nor repairs missing geometry. The other problematic 9s remain errors.','',
      'Only C is integrated into `segment_digits.prepare_crops` for the Otsu representation. The GUI number-recognition pipeline already calls this function. Fixed-threshold/grayscale alternatives, character preprocessing, segmentation and model checkpoints remain unchanged. No cross-pipeline combination is warranted because A and B were rejected.','',
      'Production verification regenerated all 40 digit inputs and exactly matched the saved C arrays and all 80 CNN/MLP predictions. The full regression suite passed 59 tests, including empty batches, intensity support, representation isolation and detached-dot preservation. This is code/pipeline verification; a new manual GUI acceptance session was not performed.','',
      '## Limits and remaining work','',
      'These are reused development photos from four writers, not independent final-test results. The gains are two photos in a small collection and do not establish statistical significance or generalisation. Character accuracy remains 53/80 and complete-string accuracy 7/40. Texture failures 101/113, the string segmentation defects and case/shape ambiguity remain unresolved. Fresh final photos must remain separate for Week 11 evaluation after integration choices.','',
      '## Evidence and reproduction','']
    for file,label in [('plan.json','Prospective plan'),('summary.csv','Full metric comparison'),('predictions.csv','Every prediction and input hash'),('paired.csv','Paired gains and regressions'),('diagnostics.json','Component boxes and warnings'),('visual_review.json','Visual observations'),('decision.json','Adoption decision'),('verification.json','Baseline verification'),('integration_verification.json','Production integration verification'),('software_tests_after_integration.txt','59-test regression log')]:text.append(f'- [{label}](../../{root.as_posix()}/{file})')
    text+=['','The root experiment refuses a changed baseline. To reproduce after adoption, run the archived baseline source snapshot from the repository root. The archived runner differs from the initially executed runner only in source-root/snapshot handling; the executed runner is also retained at the evidence root with its original hash in plan.json. Reproduction source hashes are listed separately.','',
       '```powershell',f'python {root.as_posix()}/source/experiment_preprocessing_week10.py --output evidence/preprocessing_experiment/week10_repeat','python verify_preprocessing_week10.py','python -m unittest discover -s tests -v','```','']
    Path('docs/reports/preprocessing-experiment-week10.md').write_text('\n'.join(text),encoding='utf-8')
    sources={p.relative_to(root/'source').as_posix():sha(p) for p in (root/'source').rglob('*.py')}
    (root/'reproduction_sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    print('Report and decision saved')


if __name__=='__main__':main()
