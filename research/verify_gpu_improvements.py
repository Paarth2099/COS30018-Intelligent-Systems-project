"""Verify imported paired GPU evidence and reproduce its development predictions."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from hnrs_ml.predict import DigitPredictor
from hnrs_chars.predict import CharacterPredictor
from research.evaluate_training_improvements import distance


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())


def main():
    root=Path('experiments/improvements_gpu_run01')
    output=Path('evidence/improvements/gpu_evaluation_run01')
    prepared=Path('evidence/improvements/development_run01/prepared')
    def array_path(item):return prepared/(Path(item['array']).stem+'_baseline.npy')
    protocol=load(output/'protocol.json')
    assert not protocol['smoke']
    assert sha(output/'prepared_manifest.json')==protocol['prepared_manifest_sha256']
    manifest=load(output/'prepared_manifest.json')
    assert len(manifest)==160 and len({r['sample_id'] for r in manifest})==160
    for row in manifest:assert sha(array_path(row))==row['array_sha256']
    rows=load(output/'photo_predictions.json');summary=load(output/'photo_summary.json')
    benchmarks=load(output/'benchmarks.json')
    assert len(rows)==640 and len(summary)==16 and len(benchmarks)==8
    torch.set_num_threads(4);selections={};reproduced=0
    for task in ('mnist','emnist'):
        for seed in (42,43):
            for arm in ('control','augmented'):
                key=(task,seed,arm);folder=root/task/f'seed_{seed}'/arm
                s=load(folder/'selection.json');config=s['config'];name=s['model'];checkpoint=folder/f'{name}.pt'
                assert (config['task'],config['seed'],config['arm'])==key
                assert config['smoke'] is False and config['epochs']==5 and config['batch_size']==256
                assert config['runtime']['device']=='cuda'
                assert sha(checkpoint)==s['checkpoint_sha256']
                assert load(folder/'config.json')==config
                assert next(v['selection'] for v in protocol['selections'] if (v['task'],v['seed'],v['arm'])==key)==s
                for source,digest in config['source_sha256'].items():assert sha(folder/'source'/source)==digest
                history=load(folder/'history.json');assert [r['epoch'] for r in history]==list(range(1,6))
                chosen=max(history,key=lambda r:(r['validation_accuracy'],-r['validation_loss'],-r['epoch']))
                assert chosen['epoch']==s['best_epoch'] and chosen['validation_accuracy']==s['validation_accuracy']
                assert chosen['validation_loss']==s['validation_loss']
                split=np.load(folder/'split_indices.npz')
                for field in ('train','validation'):
                    assert hashlib.sha256(split[field].astype(np.int64).tobytes()).hexdigest()==config[field+'_indices_sha256']
                assert len(set(split['train']).intersection(split['validation']))==0
                assert len(split['train'])==config['train_count'] and len(split['validation'])==config['validation_count']
                saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
                assert saved['config']==config and saved['epoch']==s['best_epoch']
                selections[key]=s
                bench=next(r for r in benchmarks if (r['task'],r['seed'],r['arm'])==key)
                arrays=np.load(output/f'{task}_{seed}_{arm}_benchmark.npz')
                truth,pred=arrays['truth'],arrays['prediction'];classes=10 if task=='mnist' else 62
                assert len(truth)==(10000 if task=='mnist' else 116323)
                confusion=np.bincount(truth*classes+pred,minlength=classes*classes).reshape(classes,classes)
                assert np.array_equal(confusion,arrays['confusion'])
                assert abs(bench['accuracy']-(truth==pred).mean())<1e-12
                tp=confusion.diagonal();support=confusion.sum(1);called=confusion.sum(0)
                precision=np.divide(tp,called,out=np.zeros(classes,float),where=called!=0)
                recall=np.divide(tp,support,out=np.zeros(classes,float),where=support!=0)
                f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(classes,float),where=precision+recall!=0)
                assert abs(bench['macro_f1']-f1.mean())<1e-12
                for i,c in enumerate(bench['per_class']):
                    assert c['support']==support[i]
                    for field,value in [('precision',precision[i]),('recall',recall[i]),('f1',f1[i])]:assert abs(c[field]-value)<1e-12
                predictor=DigitPredictor(checkpoint) if task=='mnist' else CharacterPredictor(checkpoint)
                fn=predictor.predict_number if task=='mnist' else predictor.predict_text
                subset=[r for r in rows if (r['task'],r['seed'],r['arm'])==key]
                expected=[m for m in manifest if (m['collection'] in ('single_digits','numbers'))==(task=='mnist')]
                assert len(subset)==len(expected) and {r['sample_id'] for r in subset}=={m['sample_id'] for m in expected}
                for item in expected:
                    row=next(r for r in subset if r['sample_id']==item['sample_id'])
                    assert row['reference']==item['reference'] and row['writer']==item['writer'] and row['collection']==item['collection']
                    prediction=fn(np.load(array_path(item)))['text']
                    assert prediction==row['prediction'],f'CPU reproduction differs: {key}, {item["sample_id"]}'
                    assert row['correct']==int(row['reference']==prediction)
                    assert row['edits']==distance(row['reference'],prediction)
                    reproduced+=1
                print('Verified',task,seed,arm,flush=True)
    for task in ('mnist','emnist'):
        for seed in (42,43):
            a,b=(selections[task,seed,arm]['config'] for arm in ('control','augmented'))
            for field in ('initial_state_sha256','train_indices_sha256','validation_indices_sha256','epochs','batch_size','source_sha256'):
                assert a[field]==b[field]
    for s in summary:
        subset=[r for r in rows if all(r[k]==s[k] for k in ('task','seed','arm','collection'))]
        assert len(subset)==s['count'] and sum(r['correct'] for r in subset)==s['correct']
        assert abs(sum(r['edits'] for r in subset)/sum(len(r['reference']) for r in subset)-s['cer'])<1e-12
    decision=[]
    for task in ('mnist','emnist'):
        means={arm:float(np.mean([selections[task,seed,arm]['validation_accuracy'] for seed in (42,43)])) for arm in ('control','augmented')}
        group_rows=[]
        for collection in (('single_digits','numbers') if task=='mnist' else ('characters','strings')):
            group={}
            for arm in ('control','augmented'):
                subset=[s for s in summary if (s['task'],s['arm'],s['collection'])==(task,arm,collection)]
                group[arm]=dict(mean_correct=float(np.mean([r['correct'] for r in subset])),mean_cer=float(np.mean([r['cer'] for r in subset])),images_per_seed=subset[0]['count'])
            group_rows.append(dict(collection=collection,**group))
        no_photo_regression=all(g['augmented']['mean_correct']>=g['control']['mean_correct'] and g['augmented']['mean_cer']<=g['control']['mean_cer'] for g in group_rows)
        strict_gain=any(g['augmented']['mean_correct']>g['control']['mean_correct'] or g['augmented']['mean_cer']<g['control']['mean_cer'] for g in group_rows)
        passes=means['augmented']>=means['control'] and no_photo_regression and strict_gain
        decision.append(dict(task=task,mean_validation_accuracy=means,development=group_rows,
            no_development_regression=no_photo_regression,strict_development_gain=strict_gain,
            passes_prespecified_arm_rule=passes,action='RETAIN_PRODUCTION' if not passes else 'REVIEW_REPRESENTATIVE_CHECKPOINT'))
    result=dict(status='PASS',checkpoints_verified=8,source_snapshots_verified=True,validation_selection_verified=True,
        benchmark_metrics_recalculated=8,development_predictions_reproduced_on_cpu=reproduced,
        same_prepared_development_manifest=True,fresh_photos_used=False,decisions=decision)
    (output/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
