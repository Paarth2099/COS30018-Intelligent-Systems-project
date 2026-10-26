"""Independent preprocessing candidates on frozen development photos."""
import argparse
import csv
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw
from research.compare_photo_models import sha, prepare
from research.compare_segmentation import edit_distance
from hnrs_chars.preprocessing_candidates import prepare_character_variants
from hnrs_chars.segmentation import segment_characters
from hnrs_chars.predict import CharacterPredictor
from hnrs_ml.predict import DigitPredictor


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def normalise_digit_batch(arrays):
    """Preserve geometry and zero batches; rescale each nonempty crop's peak."""
    if not len(arrays): return arrays.copy()
    peaks = arrays.max(axis=(1, 2), keepdims=True)
    return (arrays / np.where(peaks > 0, peaks, 1)).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); out = args.output
    if out.exists(): p.error('Use a new output directory')
    frozen = Path('evidence/photo_model_comparison/week09_run01')
    old = json.loads((frozen/'summary.json').read_text())
    source_root=Path(__file__).resolve().parents[1]
    for file, digest in old['sources'].items(): assert sha(source_root/file) == digest, file
    with (frozen/'paired_results.csv').open() as f: items = list(csv.DictReader(f))
    out.mkdir(parents=True); (out/'previews').mkdir(); (out/'prepared').mkdir()
    plan = dict(purpose='Development comparison, not final evaluation',
        A='Character MaxFilter size 31 -> 15; all other settings fixed',
        B='Character filtering: discard surviving components smaller than 10% of largest only when bounding-box gap exceeds half its height; preserve nearby dots',
        C='Digit post-resize peak normalisation only, no thickening',
        controls='Unchanged segmentation, CNN primary/MLP secondary, checkpoint hashes, labels and photos; no fresh final photos',
        adoption='Require no CNN exact-accuracy or CER regression in either relevant collection, strict improvement in at least one, no new preparation failures, and visual review before integration. No automatic deployment.',
        checkpoints=old['checkpoints'], original_sources=old['sources'],
        experiment_sources={f:sha(source_root/f) for f in ['research/experiment_preprocessing_week10.py','hnrs_chars/preprocessing_candidates.py']})
    (out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    for f in {**old['sources'],**plan['experiment_sources']}:
        destination=out/'source'/f;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source_root/f,destination)
    models = {}
    torch.set_num_threads(4)
    for task, factory in [('digits',DigitPredictor),('characters',CharacterPredictor)]:
        for model in ('cnn','mlp'):
            record = old['checkpoints'][f'{task}_{model}']
            assert sha(record['file']) == record['sha256']
            predictor = factory(record['file'])
            models[task,model] = predictor.predict_number if task=='digits' else predictor.predict_text
            models[task,model](np.zeros((1,28,28),np.float32))
    rows=[]; diagnostics=[]
    for index,item in enumerate(items):
        assert sha(item['file']) == item['sha256']
        with Image.open(item['file']) as im: image=ImageOps.exif_transpose(im).convert('RGB')
        start=time.perf_counter(); baseline,boxes,warnings=prepare(image,item['collection'])
        base_ms=(time.perf_counter()-start)*1000
        saved=np.load(frozen/'prepared'/f'{index:03d}.npy')
        assert np.array_equal(baseline,saved), f'Baseline arrays changed: {index}'
        digits=item['collection'] in ('single_digits','numbers')
        variants={'baseline':(baseline,base_ms,'')}; details={}
        if digits:
            start=time.perf_counter(); candidate=normalise_digit_batch(baseline)
            variants['C_peak']=(candidate,base_ms+(time.perf_counter()-start)*1000,'')
        else:
            start=time.perf_counter()
            crops=[image] if item['collection']=='characters' else segment_characters(image,'grouped_components').crops
            seg_ms=(time.perf_counter()-start)*1000
            for name,kwargs in [('A_background',{'background_size':15}),('B_foreground',{'filter_distant':True})]:
                start=time.perf_counter(); ds=[]
                try:
                    arrays=[]
                    for crop in crops:
                        v,_,d=prepare_character_variants(crop,**kwargs); arrays.append(v['otsu']); ds.append(d)
                    candidate=np.stack(arrays) if arrays else np.empty((0,28,28),np.float32)
                    assert candidate.shape==baseline.shape
                    error=''
                except ValueError as exc:
                    candidate=np.empty((0,28,28),np.float32);error=str(exc)
                variants[name]=(candidate,seg_ms+(time.perf_counter()-start)*1000,error);details[name]=ds
        predictions={}
        for name,(arrays,ms,error) in variants.items():
            np.save(out/'prepared'/f'{index:03d}_{name}.npy',arrays)
            predictions[name]={}
            for model in ('cnn','mlp'):
                prediction=models['digits' if digits else 'characters',model](arrays)['text'] if not error else ''
                if name=='baseline': assert prediction==item[model], (index,model)
                predictions[name][model]=prediction
                rows.append(dict(index=index,file=item['file'],input_sha256=item['sha256'],collection=item['collection'],writer=item['writer'],reference=item['reference'],variant=name,model=model,prediction=prediction,correct=int(prediction==item['reference']),edits=edit_distance(item['reference'],prediction),preparation_ms=ms,error=error,warnings='; '.join(warnings),candidate_warning_count=sum(len(d['warnings']) for d in details.get(name,[])),array_sha256=sha(out/'prepared'/f'{index:03d}_{name}.npy')))
        canvas=Image.new('RGB',(1150,110+110*len(variants)),'white');draw=ImageDraw.Draw(canvas)
        draw.text((10,8),f'{index:03d} {item["file"]} true={item["reference"]}',fill='black')
        thumb=image.copy();thumb.thumbnail((260,330));canvas.paste(thumb,(10,40))
        for k,(name,(arrays,ms,error)) in enumerate(variants.items()):
            y=40+110*k;draw.text((290,y),f'{name}: CNN={predictions[name]["cnn"]} MLP={predictions[name]["mlp"]} {error}',fill='black')
            for j,a in enumerate(arrays):canvas.paste(Image.fromarray(np.rint(a*255).astype('uint8')).resize((70,70)),(290+j*80,y+20))
        canvas.save(out/'previews'/f'{index:03d}.png')
        diagnostics.append(dict(index=index,boxes=boxes,segmentation_warnings=warnings,candidates=details))
        print(f'{index+1}/160 {item["collection"]} {item["reference"]} {predictions}',flush=True)
    write_csv(out/'predictions.csv',rows)
    summary=[];paired=[]
    for collection in ('single_digits','numbers','characters','strings'):
        for variant in sorted({r['variant'] for r in rows if r['collection']==collection}):
            for model in ('cnn','mlp'):
                group=[r for r in rows if (r['collection'],r['variant'],r['model'])==(collection,variant,model)]
                summary.append(dict(collection=collection,variant=variant,model=model,count=len(group),correct=sum(r['correct'] for r in group),accuracy=sum(r['correct'] for r in group)/len(group),cer=sum(r['edits'] for r in group)/sum(len(r['reference']) for r in group),failures=sum(bool(r['error']) for r in group),mean_preparation_ms=np.mean([r['preparation_ms'] for r in group])))
                if variant!='baseline':
                    base={r['index']:r for r in rows if (r['collection'],r['variant'],r['model'])==(collection,'baseline',model)}
                    for r in group:
                        b=base[r['index']]
                        paired.append(dict(index=r['index'],collection=collection,variant=variant,model=model,reference=r['reference'],baseline=b['prediction'],candidate=r['prediction'],outcome='gain' if r['correct']>b['correct'] else 'regression' if r['correct']<b['correct'] else 'both_correct' if r['correct'] else 'both_wrong'))
    write_csv(out/'summary.csv',summary);write_csv(out/'paired.csv',paired)
    (out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    (out/'verification.json').write_text(json.dumps(dict(status='PASS',photos=len(items),baseline_arrays_reproduced=160,baseline_predictions_reproduced=320,prediction_rows=len(rows),fresh_photos_used=False,source_hashes_verified=True,checkpoint_hashes_verified=True),indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
