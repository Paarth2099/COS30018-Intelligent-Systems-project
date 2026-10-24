"""Frozen four-architecture evaluation on official tests and existing development photos.

No training, hyperparameter search, fresh-photo discovery or GUI promotion occurs.
Model candidates are recorded from validation before official tests are loaded.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import statistics
import time
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from torchvision.datasets import MNIST
from hnrs_chars.data import CLASSES, load_partition, sha256
from hnrs_chars.evaluate import metrics, csv_rows, confusion_chart, error_grid
from hnrs_ml.evaluate import classification_metrics, make_confusion_grid, make_failure_grid
from hnrs_ml.predict import DigitPredictor
from hnrs_chars.predict import CharacterPredictor
from compare_photo_models import discover, prepare
from compare_segmentation import edit_distance

NAMES=('mlp','cnn','resnet18','mobilenet_v3_small')


def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def registry():
    records=[]
    for task,base,factory in [('mnist',Path('experiments/baseline'),DigitPredictor),
                              ('emnist',Path('experiments/emnist_byclass/run_01'),CharacterPredictor)]:
        baseline=json.loads((base/'selection.json').read_text())
        for name in NAMES:
            if name in ('mlp','cnn'):
                entry=next(r for r in baseline['models'] if r['model']==name)
                config=baseline['config'];path=base/f'{name}.pt'
            else:
                folder=Path('experiments/week11_gpu_run01')/task/name
                entry=json.loads((folder/'selection.json').read_text());config=entry['config'];path=folder/f'{name}.pt'
            digest=sha256(path)
            if digest!=entry['checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch '+str(path))
            for key in ('train_indices_sha256','validation_indices_sha256'):
                if config[key]!=baseline['config'][key]:raise ValueError('Split mismatch '+str(path))
            if config['epochs']!=5 or config['batch_size']!=256 or config['seed']!=42:
                raise ValueError('Unexpected training budget '+str(path))
            checkpoint=torch.load(path,map_location='cpu',weights_only=True)
            if checkpoint['model_name']!=name or checkpoint['epoch']!=entry['best_epoch']:
                raise ValueError('Checkpoint selection metadata mismatch')
            if abs(checkpoint['validation_accuracy']-entry['validation_accuracy'])>1e-12:
                raise ValueError('Checkpoint accuracy metadata mismatch')
            predictor=factory(path)
            records.append(dict(task=task,model=name,checkpoint=path.as_posix(),sha256=digest,
                validation_accuracy=entry['validation_accuracy'],validation_loss=entry['validation_loss'],
                best_epoch=entry['best_epoch'],parameters=sum(p.numel() for p in predictor.model.parameters()),
                checkpoint_bytes=path.stat().st_size,training_seconds=entry['training_seconds'],
                training_device=config.get('device'),training_platform=config.get('platform',config.get('runtime',{}).get('platform')),
                train_indices_sha256=config['train_indices_sha256'],validation_indices_sha256=config['validation_indices_sha256']))
    return records


def choose_validation(records):
    return {task:max([r for r in records if r['task']==task],key=lambda r:(r['validation_accuracy'],-r['validation_loss']))['model']
            for task in ('mnist','emnist')}


def preview(image,arrays,item,predictions,destination):
    canvas=Image.new('RGB',(1100,470),'white');draw=ImageDraw.Draw(canvas)
    draw.text((12,8),f"{item['file']} | reference: {item['reference']}",fill='black')
    original=image.copy();original.thumbnail((400,400));canvas.paste(original,(12,50))
    for j,name in enumerate(NAMES):draw.text((430,35+j*22),f'{name}: {predictions[name]}',fill='black')
    for j,a in enumerate(arrays):
        tile=Image.fromarray(np.rint(a*255).astype('uint8')).resize((70,70))
        canvas.paste(tile,(430+(j%8)*80,150+(j//8)*95))
    draw.text((430,435),'Same ordered prepared crops for every architecture',fill='black');canvas.save(destination)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('evidence/model_comparison/week11_run01'))
    args=p.parse_args();out=args.output
    if out.exists():p.error('Choose a fresh output directory; preserve previous evidence')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    records=registry();candidates=choose_validation(records)
    out.mkdir(parents=True)
    source_files=[Path(__file__),Path('compare_photo_models.py'),Path('compare_segmentation.py'),
                  Path('segment_digits.py'),Path('preprocess_digits.py'),*Path('hnrs_ml').glob('*.py'),*Path('hnrs_chars').glob('*.py')]
    source_hashes={}
    for source in source_files:
        relative=Path(source.name) if source.is_absolute() else source
        destination=out/'source'/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,destination);source_hashes[relative.as_posix()]=sha256(source)
    # This plan is persisted before any official test dataset is opened.
    save(out/'plan.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),models=records,
         validation_candidates=candidates,selection_basis='Highest saved validation accuracy, then lower validation loss. Recorded before test loading.',
         benchmark_role='Reporting only; no tuning or selection on official tests',
         photo_role='Previously used development images, not independent final evaluation',
         evaluation_device='cpu',threads=4,python=platform.python_version(),torch=str(torch.__version__),platform=platform.platform(),
         source_sha256=source_hashes,fresh_photos_used=False,
         promotion_rule='Validation candidate must also have no lower exact accuracy and no higher CER than baseline CNN on either development collection for its task; otherwise retain baseline pending investigation. Official tests excluded.',
         timing='All models on this CPU, four threads. Per photo median of five warmed calls, rotating model order; excludes preparation.'))
    csv_rows(out/'models.csv',records)
    benchmarks=[]
    for task in ('mnist','emnist'):
        if task=='mnist':
            dataset=MNIST('data',train=False,download=False);images,targets=dataset.data,dataset.targets
        else:images,targets=load_partition(Path('data'),train=False)
        truth=targets.numpy()
        for record in [r for r in records if r['task']==task]:
            name=record['model'];factory=DigitPredictor if task=='mnist' else CharacterPredictor
            model=factory(record['checkpoint']).model;guesses=[]
            with torch.inference_mode():
                model(images[:256].unsqueeze(1).float()/255)
                started=time.perf_counter()
                for offset in range(0,len(targets),256):
                    x=images[offset:offset+256].unsqueeze(1).float()/255
                    guesses.extend(model(x).argmax(1).tolist())
                    if offset%25600==0:print(f'{task} {name}: {min(offset+256,len(targets))}/{len(targets)}',flush=True)
            elapsed=time.perf_counter()-started;predicted=np.array(guesses)
            report=(classification_metrics if task=='mnist' else metrics)(truth,predicted)
            folder=out/'benchmarks'/task/name;folder.mkdir(parents=True)
            save(folder/'metrics.json',report)
            csv_rows(folder/'per_class.csv',report['per_class'])
            np.savez_compressed(folder/'predictions.npz',truth=truth,prediction=predicted)
            if task=='mnist':
                make_confusion_grid(report['confusion_matrix'],folder/'confusion.png')
                make_failure_grid(images.unsqueeze(1).float()/255,truth,predicted,folder/'errors.png')
            else:
                confusion_chart(report['confusion_matrix'],folder/'confusion.png')
                error_grid(images,truth,predicted,folder/'errors.png')
            row=dict(task=task,model=name,count=report['count'],correct=report['correct'],accuracy=report['accuracy'],macro_f1=report['macro_f1'],
                     evaluation_seconds=elapsed,ms_per_image=elapsed*1000/len(truth),parameters=record['parameters'],checkpoint_bytes=record['checkpoint_bytes'])
            benchmarks.append(row);csv_rows(out/'benchmark_summary.csv',benchmarks)
            print(json.dumps(row),flush=True)
        del images,targets
    predictors={}
    for r in records:
        predictor=(DigitPredictor if r['task']=='mnist' else CharacterPredictor)(r['checkpoint'])
        fn=predictor.predict_number if r['task']=='mnist' else predictor.predict_text
        fn(np.zeros((1,28,28),np.float32));predictors[r['task'],r['model']]=fn
    items=discover();csv_rows(out/'manifest.csv',items)
    (out/'prepared').mkdir();(out/'previews').mkdir();rows=[];pairs=[];prepared_records=[]
    for index,item in enumerate(items):
        with Image.open(item['file']) as f:image=ImageOps.exif_transpose(f).convert('RGB')
        arrays,boxes,warnings=prepare(image,item['collection']);path=out/'prepared'/f'{index:03d}.npy';np.save(path,arrays)
        task='mnist' if item['collection'] in ('single_digits','numbers') else 'emnist'
        results={};durations={name:[] for name in NAMES}
        # Warm every model with this crop count before timed calls.
        for name in NAMES:predictors[task,name](arrays)
        for repeat in range(5):
            rotation=(index+repeat)%4;order=NAMES[rotation:]+NAMES[:rotation]
            for name in order:
                start=time.perf_counter();text=predictors[task,name](arrays)['text'];durations[name].append((time.perf_counter()-start)*1000)
                if name in results and results[name]!=text:raise ValueError('Nonrepeatable prediction')
                results[name]=text
        pair=dict(index=index,**item,crop_count=len(arrays),**results);pairs.append(pair)
        prepared_records.append(dict(index=index,input_sha256=item['sha256'],prepared_sha256=sha256(path),boxes=boxes,warnings=warnings))
        for name in NAMES:
            rows.append(dict(index=index,task=task,collection=item['collection'],writer=item['writer'],file=item['file'],reference=item['reference'],
                model=name,prediction=results[name],correct=int(results[name]==item['reference']),edits=edit_distance(item['reference'],results[name]),
                reference_length=len(item['reference']),crop_count=len(arrays),median_inference_ms=statistics.median(durations[name])))
        preview(image,arrays,item,results,out/'previews'/f'{index:03d}.png')
    csv_rows(out/'photo_predictions.csv',rows);csv_rows(out/'paired_photos.csv',pairs);save(out/'preparation.json',prepared_records)
    summary=[]
    for collection in ('single_digits','numbers','characters','strings'):
        for name in NAMES:
            subset=[r for r in rows if r['collection']==collection and r['model']==name]
            summary.append(dict(collection=collection,model=name,count=len(subset),correct=sum(r['correct'] for r in subset),
                accuracy=sum(r['correct'] for r in subset)/len(subset),cer=sum(r['edits'] for r in subset)/sum(r['reference_length'] for r in subset),
                mean_photo_inference_ms=statistics.mean(r['median_inference_ms'] for r in subset)))
    csv_rows(out/'photo_summary.csv',summary)
    recommendations={}
    for task,collections in [('mnist',('single_digits','numbers')),('emnist',('characters','strings'))]:
        candidate=candidates[task];gates={}
        for collection in collections:
            baseline=next(r for r in summary if r['collection']==collection and r['model']=='cnn')
            contender=next(r for r in summary if r['collection']==collection and r['model']==candidate)
            gates[collection+'_exact_no_regression']=contender['accuracy']>=baseline['accuracy']
            gates[collection+'_cer_no_regression']=contender['cer']<=baseline['cer']
        recommendations[task]=dict(validation_candidate=candidate,development_gates=gates,
            recommended_model=candidate if all(gates.values()) else 'cnn',gui_changed=False,
            note='Development recommendation; independent final evaluation and GUI acceptance still required.')
    save(out/'recommendation.json',recommendations)
    for record in records:
        if sha256(record['checkpoint'])!=record['sha256']:raise ValueError('Checkpoint changed during evaluation')
    for name,digest in source_hashes.items():
        if sha256(name)!=digest:raise ValueError('Source changed during evaluation')
    for item in items:
        if sha256(item['file'])!=item['sha256']:raise ValueError('Photo changed during evaluation')
    save(out/'verification.json',dict(status='PASS',checkpoint_count=len(records),official_test_predictions=sum(r['count'] for r in benchmarks),
        development_photos=len(items),development_predictions=len(rows),sources_and_checkpoints_unchanged=True,
        validation_candidates_recorded_before_tests=True,fresh_photos_used=False,gui_changed=False))
    print(json.dumps(dict(photos=summary,recommendations=recommendations),indent=2),flush=True)


if __name__=='__main__':main()
