"""One-shot locked photo evaluation. Does not train, tune or change GUI defaults."""
import csv
import hashlib
import json
import platform
from pathlib import Path
import shutil
import statistics
import subprocess
import time
from datetime import datetime, timezone
from collections import Counter, defaultdict
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import torch
from gui_core import resolve_checkpoint, annotated_image
from hnrs_ml.predict import DigitPredictor
from hnrs_chars.predict import CharacterPredictor
from hnrs_chars.segmentation import segment_characters, prepare_character_crops
from segment_digits import segment_image, prepare_crops
from research.compare_segmentation import edit_distance
from research.compare_photo_models import prepare as development_prepare

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/final_evaluation/run_01'
NAMES=('mlp','cnn','resnet18','mobilenet_v3_small')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
    Path(p).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def preparation(image,task,method):
    seg=segment_image(image,method) if task=='mnist' else segment_characters(image,method)
    arrays=prepare_crops(seg.crops,'otsu') if task=='mnist' else prepare_character_crops(seg.crops)
    return seg,arrays
def photograph(path):
    with Image.open(path) as im: return ImageOps.exif_transpose(im).convert('RGB')
def score(reference,prediction):
    return dict(correct=int(reference==prediction),edits=edit_distance(reference,prediction),reference_length=len(reference))
def summarise(rows):
    groups=defaultdict(list)
    for r in rows: groups[(r['collection'],r['model'],r['method'])].append(r)
    result=[]
    for (collection,model,method),items in groups.items():
        timing=[r['median_end_to_end_ms'] for r in items if r['median_end_to_end_ms'] is not None]
        result.append(dict(collection=collection,model=model,method=method,count=len(items),correct=sum(r['correct'] for r in items),accuracy=sum(r['correct'] for r in items)/len(items),cer=sum(r['edits'] for r in items)/sum(r['reference_length'] for r in items),errors=sum(bool(r['error']) for r in items),median_end_to_end_ms=statistics.median(timing) if timing else None,p95_end_to_end_ms=float(np.percentile(timing,95)) if timing else None))
    return result

def main():
    if OUT.exists(): raise SystemExit('Preserve existing final run; refusing overwrite.')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    manifest=json.loads((ROOT/'evidence/final_evaluation/protocol/collection_manifest.json').read_text())
    items=manifest['samples']
    assert len(items)==120 and len({r['sample_id'] for r in items})==120
    assert Counter(r['collection'] for r in items)==dict(single_digits=40,numbers=20,single_letters=40,strings=20)
    for r in items:
        assert r['verified_label']==r['planned_label'] and sha(ROOT/r['actual_path'])==r['sha256']
    predictors={}; checkpoints=[]
    for task,mode,factory in [('mnist','Numbers',DigitPredictor),('emnist','Alphanumeric',CharacterPredictor)]:
        for name in NAMES:
            path,digest=resolve_checkpoint(ROOT,mode,name)
            predictor=factory(path)
            predictors[task,name]=predictor.predict_number if task=='mnist' else predictor.predict_text
            checkpoints.append(dict(task=task,model=name,path=path.relative_to(ROOT).as_posix(),sha256=digest))
    # Reproduce earlier results on development inputs BEFORE final inference.
    checks=[]
    with (ROOT/'evidence/model_comparison/week11_run01/paired_photos.csv').open(newline='') as f:
        old=list(csv.DictReader(f))
    selected=[next(r for r in old if r['collection']==c) for c in ['single_digits','numbers','characters','strings']]
    for r in selected:
        assert sha(ROOT/r['file'])==r['sha256']
        image=photograph(ROOT/r['file']); arrays,_,_=development_prepare(image,r['collection'])
        task='mnist' if r['collection'] in ('single_digits','numbers') else 'emnist'
        for name in NAMES:
            text=predictors[task,name](arrays)['text']
            assert text==r[name], (r['file'],name,text,r[name])
            checks.append(dict(file=r['file'],model=name,expected=r[name],actual=text))
    # Warm the actual final route, including both segmentation methods, on development photos.
    for task,kind,method in [('mnist','numbers','components'),('emnist','strings','grouped_components')]:
        im=photograph(ROOT/next(r['file'] for r in selected if r['collection']==kind))
        for meth in [method,'projection']:
            _,a=preparation(im,task,meth)
            for name in NAMES: predictors[task,name](a)
    OUT.mkdir(parents=True)
    sources=[Path(__file__),ROOT/'gui_core.py',ROOT/'app.py',ROOT/'segment_digits.py',ROOT/'preprocess_digits.py',ROOT/'research/compare_segmentation.py',ROOT/'research/compare_photo_models.py',*sorted((ROOT/'hnrs_ml').glob('*.py')),*sorted((ROOT/'hnrs_chars').glob('*.py'))]
    source_hashes={}
    for src in sources:
        rel=src.relative_to(ROOT); target=OUT/'source'/rel; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,target); source_hashes[rel.as_posix()]=sha(src)
    manifest.update(status='locked_for_run_01',dataset_locked=True,recognition_run=False,locked_utc=datetime.now(timezone.utc).isoformat())
    manifest['user_confirmation']['prior_gui_testing']='User explicitly confirmed none tested before this run'
    save(OUT/'locked_manifest.json',manifest)
    configuration=dict(created_utc=datetime.now(timezone.utc).isoformat(),base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),working_tree_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),version_note='Base commit plus exact archived evaluator/runtime sources; uncommitted evaluation/planning files are explicitly recorded. No commit or push performed.',source_sha256=source_hashes,manifest_sha256=sha(OUT/'locked_manifest.json'),checkpoints=checkpoints,primary={'mnist':'cnn','emnist':'resnet18'},methods={'mnist':'components','emnist':'grouped_components'},representation='otsu',single_letter_route='GUI-equivalent grouped segmentation then character preparation; not direct isolated-character preprocessing used by older development comparison',acceptance={'status':'user_reports_completed_evidence_pending','user_authorized_proceeding_before_upload':True},human_label_review='Paarth collective attestation plus assistant visual inspection; separate teammate review not claimed',freshness='User confirmed newly supplied photographs and none previously tested',python=platform.python_version(),torch=str(torch.__version__),platform=platform.platform(),device='cpu',threads=4,timing='Five repeated end-to-end and model-only calls per input/config, warmed on development photos only; rotate model order; exclude disk load, model load, export and GUI. Score once, timing repeats are not samples.',post_test_tuning_allowed=False)
    save(OUT/'frozen_configuration.json',configuration);save(OUT/'development_reproduction.json',checks)
    print('FROZEN before final inference; 16 development predictions reproduced.',flush=True)
    rows=[];preps=[]
    for index,item in enumerate(items):
        sid=item['sample_id']; image=photograph(ROOT/item['actual_path'])
        task='mnist' if item['collection'] in ('single_digits','numbers') else 'emnist'
        method='components' if task=='mnist' else 'grouped_components'
        jobs=[(task,method,item['collection'],NAMES)]
        if item['collection'] in ('numbers','strings'): jobs.append((task,'projection',item['collection'],('cnn',) if task=='mnist' else ('resnet18',)))
        if item['collection']=='single_digits': jobs.append(('emnist','grouped_components','auxiliary_emnist_digits',NAMES))
        for task,method,collection,names in jobs:
            folder=OUT/'samples'/sid/f'{task}_{method}';folder.mkdir(parents=True)
            prep_error=None;seg=None;arrays=np.empty((0,28,28),np.float32)
            try:
                seg,arrays=preparation(image,task,method)
                np.save(folder/'prepared.npy',arrays);annotated_image(seg).save(folder/'segmentation.png')
                for k,crop in enumerate(seg.crops): crop.save(folder/f'crop_{k:02d}.png')
            except Exception as exc: prep_error=f'{type(exc).__name__}: {exc}'
            meta=dict(sample_id=sid,task=task,method=method,collection=collection,crop_count=len(arrays),boxes=seg.boxes if seg else [],warnings=seg.warnings if seg else [],error=prep_error,prepared_sha256=sha(folder/'prepared.npy') if not prep_error else None)
            save(folder/'preparation.json',meta);preps.append(meta)
            outputs={};errors={};timings={n:dict(model_only_ms=[],end_to_end_ms=[]) for n in names}
            for rep in range(5):
                rotation=(index+rep)%len(names);order=names[rotation:]+names[:rotation]
                for name in order:
                    if prep_error: outputs[name]='';errors[name]=prep_error;continue
                    try:
                        start=time.perf_counter();text=predictors[task,name](arrays)['text'];elapsed=(time.perf_counter()-start)*1000
                        start=time.perf_counter();_,fresh=preparation(image,task,method);e2e=predictors[task,name](fresh)['text'];elapsed2=(time.perf_counter()-start)*1000
                        assert np.array_equal(arrays,fresh) and text==e2e
                        if name in outputs: assert outputs[name]==text
                        outputs[name]=text;timings[name]['model_only_ms'].append(elapsed);timings[name]['end_to_end_ms'].append(elapsed2)
                    except Exception as exc: outputs[name]='';errors[name]=f'{type(exc).__name__}: {exc}'
            for name in names:
                prediction='' if name in errors else outputs[name]
                timing=timings[name]
                row=dict(sample_id=sid,writer=item['writer'],collection=collection,task=task,model=name,method=method,reference=item['verified_label'],prediction=prediction,**score(item['verified_label'],prediction),crop_count=len(arrays),error=errors.get(name),raw_timing=timing,median_model_only_ms=statistics.median(timing['model_only_ms']) if timing['model_only_ms'] else None,median_end_to_end_ms=statistics.median(timing['end_to_end_ms']) if timing['end_to_end_ms'] else None)
                rows.append(row)
            save(folder/'predictions.json',[r for r in rows if r['sample_id']==sid and r['task']==task and r['method']==method])
        if (index+1)%10==0:print(f'{index+1}/120 photos complete',flush=True)
        save(OUT/'predictions.json',rows)
    summary=summarise(rows);save(OUT/'summary.json',summary);save(OUT/'preparation.json',preps)
    save(OUT/'by_writer.json',{w:summarise([r for r in rows if r['writer']==w]) for w in ['Haresh','Paarth','Manula','Saniru']})
    confusion={}
    for collection in ['single_digits','single_letters','auxiliary_emnist_digits']:
        for name in NAMES:
            subset=[r for r in rows if r['collection']==collection and r['model']==name]
            confusion[f'{collection}/{name}']=[dict(reference=a,prediction=b,count=count) for (a,b),count in Counter((r['reference'],r['prediction'] if len(r['prediction'])==1 else '<empty_or_multiple>') for r in subset).items()]
    save(OUT/'confusion_counts.json',confusion)
    paired=[]
    for item in items:
        subset=[r for r in rows if r['sample_id']==item['sample_id'] and r['method']!='projection' and not r['collection'].startswith('auxiliary')]
        primary='cnn' if item['collection'] in ['single_digits','numbers'] else 'resnet18'
        baseline=next(r for r in subset if r['model']==primary)
        for r in subset:paired.append(dict(sample_id=item['sample_id'],collection=item['collection'],primary_model=primary,model=r['model'],primary_correct=baseline['correct'],model_correct=r['correct'],edit_difference_vs_primary=r['edits']-baseline['edits']))
    save(OUT/'paired_changes.json',paired)
    for r in items: assert sha(ROOT/r['actual_path'])==r['sha256']
    for r in checkpoints: assert sha(ROOT/r['path'])==r['sha256']
    for rel,digest in source_hashes.items(): assert sha(ROOT/rel)==digest
    assert len(rows)==680
    save(OUT/'verification.json',dict(status='PASS',completed_utc=datetime.now(timezone.utc).isoformat(),originals=120,prediction_rows=len(rows),primary_comparison_rows=480,projection_rows=40,auxiliary_rows=160,development_predictions_reproduced=16,all_hashes_unchanged=True,no_training_or_tuning=True,acceptance_evidence='pending_upload',manual_segmentation_review='pending'))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
