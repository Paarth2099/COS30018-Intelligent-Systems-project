"""Controlled geometry-only string segmentation experiments."""
import argparse
import csv
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw
from compare_photo_models import sha
from compare_segmentation import edit_distance
from hnrs_chars.segmentation_candidates import segment_characters,prepare_character_crops
from hnrs_chars.predict import CharacterPredictor
from hnrs_chars.string_experiment import save_artifacts
from hnrs_chars.photo_experiment import write_csv
from hnrs_ml.predict import DigitPredictor
from segment_digits import segment_image,prepare_crops


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--combined',action='store_true');args=p.parse_args();out=args.output
    if out.exists():p.error('Choose a new output folder')
    source=Path(__file__).resolve().parent
    frozen=Path('evidence/photo_model_comparison/week09_run01')
    old=json.loads((frozen/'summary.json').read_text())
    with (frozen/'paired_results.csv').open() as f:items=list(csv.DictReader(f))
    sources=[Path('experiment_segmentation_week10.py'),Path('segment_digits.py'),Path('preprocess_digits.py'),Path('compare_segmentation.py'),Path('compare_photo_models.py'),*Path('hnrs_chars').glob('*.py'),*Path('hnrs_ml').glob('*.py')]
    out.mkdir(parents=True)
    variants={'baseline':{},'projection':{'method':'projection'},'A_body':{'repair_bodies':True},'B_stem':{'stem_ratio':.55}}
    if args.combined:variants={'baseline':{},'combined':{'repair_bodies':True,'stem_ratio':.55}}
    plan=dict(variants=variants,body_rule='Adjacent tall fragments >=60% typical height; horizontal box gap <=4%; vertical overlap >=65% shorter height; one width <=30% typical, other >=45%; union width <=typical height. Merge at most one pair per pass.',
        stem_rule='Short-stem width/height eligibility .45 -> .55, keeping minimum height and dot assignment fixed',
        selection='Require target visual repair, no new visual separation failures, no CNN exact/CER regression, no lost components or errors. Inspect all 40; combine only independently supported changes.',
        controls='Same original CNN/MLP checkpoints, character preprocessing and development photos. No reference text/count/model output enters segmentation. Numbers use already adopted Week 10 digit preprocessing.',
        sources={str(f).replace('\\','/'):sha(source/f) for f in sources},checkpoints=old['checkpoints'],fresh_photos_used=False)
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    for f in sources:
        target=out/'source'/f;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/f,target)
    torch.set_num_threads(4);models={}
    for task,factory in [('characters',CharacterPredictor),('digits',DigitPredictor)]:
        for model in ('cnn','mlp'):
            record=old['checkpoints'][f'{task}_{model}'];assert sha(record['file'])==record['sha256']
            obj=factory(record['file']);models[task,model]=obj.predict_text if task=='characters' else obj.predict_number
    rows=[];geometry=[];number_rows=[]
    for index,item in enumerate(items):
        if item['collection'] not in ('strings','numbers'):continue
        assert sha(item['file'])==item['sha256']
        with Image.open(item['file']) as im:image=ImageOps.exif_transpose(im).convert('RGB')
        if item['collection']=='numbers':
            seg=segment_image(image,'components');a=prepare_crops(seg.crops)
            expected=np.load(Path('evidence/preprocessing_experiment/week10_run01/prepared')/f'{index:03d}_C_peak.npy')
            assert np.array_equal(a,expected)
            assert [list(b) for b in seg.boxes]==json.loads(item['boxes'])
            for model in ('cnn','mlp'):
                pred=models['digits',model](a)['text'];number_rows.append(dict(index=index,model=model,reference=item['reference'],prediction=pred,correct=int(pred==item['reference']),boxes_unchanged=True))
            continue
        base_groups=None;base_boxes=None
        for name,options in variants.items():
            start=time.perf_counter();seg=segment_characters(image,**options);seg_ms=(time.perf_counter()-start)*1000
            start=time.perf_counter();arrays=prepare_character_crops(seg.crops);prep_ms=(time.perf_counter()-start)*1000
            if name=='baseline':
                assert [list(b) for b in seg.boxes]==json.loads(item['boxes'])
                assert np.array_equal(arrays,np.load(frozen/'prepared'/f'{index:03d}.npy'))
                base_groups=seg.groups;base_boxes=seg.boxes
            assert sorted(x for g in seg.groups for x in g)==sorted(x for g in base_groups for x in g)
            assert [b[0] for b in seg.boxes]==sorted(b[0] for b in seg.boxes)
            path=out/f'{index:03d}'/name
            for model in ('cnn','mlp'):
                start=time.perf_counter();prediction=models['characters',model](arrays);infer_ms=(time.perf_counter()-start)*1000
                if name=='baseline':assert prediction['text']==item[model]
                rows.append(dict(index=index,file=item['file'],input_sha256=item['sha256'],writer=item['writer'],reference=item['reference'],variant=name,model=model,prediction=prediction['text'],correct=int(prediction['text']==item['reference']),edits=edit_distance(item['reference'],prediction['text']),crop_count=len(seg.boxes),count_correct=int(len(seg.boxes)==len(item['reference'])),warnings='; '.join(seg.warnings),segmentation_ms=seg_ms,preparation_ms=prep_ms,inference_ms=infer_ms))
                if model=='cnn':save_artifacts(path,seg,arrays,prediction,item['reference'])
            geometry.append(dict(index=index,variant=name,boxes=seg.boxes,groups=seg.groups,changed=seg.boxes!=base_boxes or seg.groups!=base_groups))
        print(f'{index:03d} {item["reference"]} '+str([(r['variant'],r['crop_count'],r['prediction']) for r in rows if r['index']==index and r['model']=='cnn']),flush=True)
    write_csv(out/'predictions.csv',rows);write_csv(out/'number_regression.csv',number_rows)
    summary=[]
    for name in variants:
        for model in ('cnn','mlp'):
            group=[r for r in rows if r['variant']==name and r['model']==model]
            assert len(group)==40
            summary.append(dict(variant=name,model=model,count=40,count_correct=sum(r['count_correct'] for r in group),correct=sum(r['correct'] for r in group),cer=sum(r['edits'] for r in group)/sum(len(r['reference']) for r in group),warning_images=sum(bool(r['warnings']) for r in group),mean_segmentation_ms=np.mean([r['segmentation_ms'] for r in group])))
    write_csv(out/'summary.csv',summary)
    (out/'geometry.json').write_text(json.dumps(geometry,indent=2)+'\n')
    (out/'verification.json').write_text(json.dumps(dict(status='PASS',baseline_strings_reproduced=40,baseline_predictions_reproduced=80,number_arrays_reproduced=20,number_boxes_unchanged=20,all_component_ids_retained=True,left_to_right_verified=True,fresh_photos_used=False),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
