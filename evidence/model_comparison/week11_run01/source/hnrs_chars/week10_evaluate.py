"""Compare validation-selected weighted CNN with baseline on validation and frozen development crops."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import statistics
import time
import numpy as np
import torch
from .data import CLASSES,load_partition,batch_images,sha256,indices_hash
from .evaluate import metrics,csv_rows,confusion_chart
from .predict import CharacterPredictor
from compare_segmentation import edit_distance


def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a new output directory')
    selection=json.loads((args.candidate/'selection.json').read_text())
    assert sha256(args.candidate/'cnn.pt')==selection['models'][0]['checkpoint_sha256']
    base=Path('experiments/emnist_byclass/run_01')
    assert sha256(base/'cnn.pt')==selection['config']['baseline_checkpoint_sha256']
    frozen=Path('evidence/photo_model_comparison/week09_run01')
    source_record=json.loads((frozen/'summary.json').read_text())
    for path,h in source_record['sources'].items():assert sha256(path)==h
    pairs=read_csv(frozen/'paired_results.csv')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    predictors={'baseline':CharacterPredictor(base/'cnn.pt'),'weighted':CharacterPredictor(args.candidate/'cnn.pt')}
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__,args.output/'week10_evaluate.py')
    images,targets=load_partition(Path('data'),train=True)
    val_ids=torch.from_numpy(np.load('evidence/emnist/preparation/split_indices.npz')['validation'])
    assert indices_hash(val_ids)==selection['config']['validation_indices_sha256']
    truth=targets[val_ids].numpy();validation={}
    for name,predictor in predictors.items():
        predictions=[]
        with torch.inference_mode():
            for ids in val_ids.split(256):
                predictions.extend(predictor.model(batch_images(images,ids)).argmax(1).tolist())
        result=metrics(truth,np.array(predictions));validation[name]=result
        np.savez_compressed(args.output/f'{name}_validation_predictions.npz',indices=val_ids.numpy(),truth=truth,prediction=predictions)
        csv_rows(args.output/f'{name}_validation_per_class.csv',result['per_class'])
        confusion_chart(result['confusion_matrix'],args.output/f'{name}_validation_confusion.png')
        print(name,'validation',result['accuracy'],'macro F1',result['macro_f1'],flush=True)
    baseline_val=next(x for x in json.loads((base/'selection.json').read_text())['models'] if x['model']=='cnn')['validation_accuracy']
    assert abs(validation['baseline']['accuracy']-baseline_val)<1e-12
    assert abs(validation['weighted']['accuracy']-selection['models'][0]['validation_accuracy'])<1e-12
    del images,targets
    for predictor in predictors.values():predictor.predict_text(np.zeros((1,28,28),np.float32))
    rows=[]
    for index,p in enumerate(pairs):
        if p['collection'] not in ('characters','strings'):continue
        assert sha256(p['file'])==p['sha256']
        path=frozen/'prepared'/f'{index:03d}.npy';a=np.load(path)
        predictions={};elapsed={name:[] for name in predictors}
        for repeat in range(5):
            order=('baseline','weighted') if (index+repeat)%2==0 else ('weighted','baseline')
            for name in order:
                t=time.perf_counter();pred=predictors[name].predict_text(a)['text']
                elapsed[name].append((time.perf_counter()-t)*1000)
                if name in predictions:assert predictions[name]==pred
                predictions[name]=pred
        assert predictions['baseline']==p['cnn']
        row=dict(index=index,file=p['file'],writer=p['writer'],collection=p['collection'],reference=p['reference'],
            input_sha256=p['sha256'],prepared_sha256=sha256(path),crop_count=len(a),
            baseline=predictions['baseline'],weighted=predictions['weighted'],
            baseline_correct=int(predictions['baseline']==p['reference']),weighted_correct=int(predictions['weighted']==p['reference']),
            baseline_edits=edit_distance(p['reference'],predictions['baseline']),weighted_edits=edit_distance(p['reference'],predictions['weighted']),
            baseline_inference_ms=statistics.median(elapsed['baseline']),weighted_inference_ms=statistics.median(elapsed['weighted']),
            preview=(frozen/p['preview']).as_posix())
        rows.append(row)
    assert len(rows)==120 and Counter(r['collection'] for r in rows)=={'characters':80,'strings':40}
    assert len({r['file'] for r in rows})==120
    csv_rows(args.output/'photo_predictions.csv',rows)
    aggregate=[]
    for collection in ('characters','strings'):
        subset=[r for r in rows if r['collection']==collection]
        for name in predictors:
            aggregate.append(dict(collection=collection,model=name,count=len(subset),correct=sum(r[f'{name}_correct'] for r in subset),
                accuracy=sum(r[f'{name}_correct'] for r in subset)/len(subset),
                character_error_rate=sum(r[f'{name}_edits'] for r in subset)/sum(len(r['reference']) for r in subset),
                mean_inference_ms=statistics.mean(r[f'{name}_inference_ms'] for r in subset)))
    paired={c:dict(improved=sum(r['weighted_correct']>r['baseline_correct'] for r in rows if r['collection']==c),
        regressed=sum(r['weighted_correct']<r['baseline_correct'] for r in rows if r['collection']==c),
        both_correct=sum(r['weighted_correct'] and r['baseline_correct'] for r in rows if r['collection']==c),
        both_wrong=sum(not r['weighted_correct'] and not r['baseline_correct'] for r in rows if r['collection']==c)) for c in ('characters','strings')}
    by_writer=[]
    for writer in sorted({r['writer'] for r in rows}):
        for collection in ('characters','strings'):
            subset=[r for r in rows if r['writer']==writer and r['collection']==collection]
            by_writer.append(dict(writer=writer,collection=collection,count=len(subset),baseline_correct=sum(r['baseline_correct'] for r in subset),weighted_correct=sum(r['weighted_correct'] for r in subset)))
    csv_rows(args.output/'by_writer.csv',by_writer)
    lookup={(r['collection'],r['model']):r for r in aggregate}
    criteria={'validation_accuracy_no_regression':validation['weighted']['accuracy']>=validation['baseline']['accuracy'],
        'character_accuracy_no_regression':lookup['characters','weighted']['accuracy']>=lookup['characters','baseline']['accuracy'],
        'string_accuracy_no_regression':lookup['strings','weighted']['accuracy']>=lookup['strings','baseline']['accuracy'],
        'string_cer_no_regression':lookup['strings','weighted']['character_error_rate']<=lookup['strings','baseline']['character_error_rate']}
    result=dict(validation=validation,photos=aggregate,paired=paired,adoption_criteria=criteria,
        recommendation='candidate eligible for later integration verification' if all(criteria.values()) else 'retain baseline',
        source_run=frozen.as_posix(),candidate=args.candidate.as_posix(),checkpoints={name:sha256(path) for name,path in [('baseline',base/'cnn.pt'),('weighted',args.candidate/'cnn.pt')]},
        validation_only=True,official_test_evaluated=False,fresh_photos_used=False,
        baseline_predictions_reproduced=120,source_snapshot_unchanged=True,
        timing='Median of five warmed CPU inference calls per photo, four threads, alternating order. Preparation reused and excluded.',
        limitation='One seed; same validation used for checkpoint selection; reused development photos; no generalisation claim.')
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.output/'verification.json').write_text(json.dumps(dict(status='PASS',photo_pairs=120,
        model_predictions=240,validation_images=len(val_ids),baseline_photo_predictions_reproduced=120,
        baseline_validation_accuracy_reproduced=True,candidate_validation_accuracy_reproduced=True,
        original_source_hashes_verified=True,checkpoint_hashes_verified=True,
        evaluator_sha256=sha256(__file__),training_plan_sha256=sha256(args.candidate/'plan.json'),
        official_test_loaded=False,fresh_photos_used=False),indent=2)+'\n')
    csv_rows(args.output/'photo_summary.csv',aggregate)
    print(json.dumps(dict(photos=aggregate,paired=paired,adoption_criteria=criteria,recommendation=result['recommendation']),indent=2),flush=True)


if __name__=='__main__':main()
