"""Evaluate all completed paired runs; no training or checkpoint selection here."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from research.train_models_week11 import digit_model,character_model,CLASSES,sha256,select_device,batch


def distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        previous=row;row=[i]
        for j,y in enumerate(b,1):row.append(min(row[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
    return row[-1]


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,default=Path('experiments/improvements_gpu_run01'))
    p.add_argument('--prepared',type=Path,required=True,help='Prepared development arrays from the GPU transfer bundle')
    p.add_argument('--output',type=Path,default=Path('evidence/improvements/gpu_evaluation_run01'))
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43]);p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.output.exists():p.error('Choose a new output folder')
    if args.smoke and 'smoke' not in args.output.name:p.error('Smoke output name must contain smoke')
    torch.set_num_threads(4);device=select_device(args.device)
    manifest=json.loads((args.prepared/'manifest.json').read_text())
    for item in manifest:
        assert sha256(args.prepared/item['array'])==item['array_sha256']
    runs=[]
    # Verify ALL required selections before touching official test partitions.
    for task in ('mnist','emnist'):
        for seed in args.seeds:
            for arm in ('control','augmented'):
                path=args.runs/task/f'seed_{seed}'/arm
                selection=json.loads((path/'selection.json').read_text())
                config=selection['config'];name=selection['model'];checkpoint=path/f'{name}.pt'
                assert (config['task'],config['seed'],config['arm'])==(task,seed,arm)
                assert config['smoke']==args.smoke and sha256(checkpoint)==selection['checkpoint_sha256']
                runs.append(dict(task=task,seed=seed,arm=arm,path=path,selection=selection,checkpoint=checkpoint))
    for task in ('mnist','emnist'):
        for seed in args.seeds:
            pair=[r['selection']['config'] for r in runs if r['task']==task and r['seed']==seed]
            for key in ('initial_state_sha256','train_indices_sha256','validation_indices_sha256','epochs','batch_size'):
                assert pair[0][key]==pair[1][key],f'Unpaired {key}'
    args.output.mkdir(parents=True)
    write(args.output/'protocol.json',dict(smoke=args.smoke,prepared_manifest_sha256=sha256(args.prepared/'manifest.json'),
        selections=[dict(task=r['task'],seed=r['seed'],arm=r['arm'],selection=r['selection']) for r in runs],
        adoption='Official test is descriptive only. Use validation plus paired development checks; do not select using fresh final photos.',
        source_sha256=sha256(__file__)))
    metrics=[];photo_rows=[]
    for task in ('mnist','emnist'):
        if task=='mnist':
            from torchvision.datasets import MNIST
            dataset=MNIST('data',train=False,download=False);images,targets=dataset.data,dataset.targets;classes=list('0123456789')
        else:
            from hnrs_chars.data import load_partition
            images,targets=load_partition(Path('data'),train=False);classes=CLASSES
        ids=torch.arange(min(128,len(targets)) if args.smoke else len(targets))
        for run in [r for r in runs if r['task']==task]:
            saved=torch.load(run['checkpoint'],map_location='cpu',weights_only=True)
            model=(digit_model if task=='mnist' else character_model)(saved['model_name']).to(device)
            model.load_state_dict(saved['state_dict']);model.eval()
            predictions=[]
            for chunk in ids.split(256):predictions.extend(model(batch(images,chunk,device)).argmax(1).cpu().tolist())
            truth=targets[ids].numpy();pred=np.asarray(predictions);k=len(classes)
            confusion=np.bincount(truth*k+pred,minlength=k*k).reshape(k,k)
            tp=confusion.diagonal();support=confusion.sum(1);called=confusion.sum(0)
            precision=np.divide(tp,called,out=np.zeros(k,float),where=called!=0)
            recall=np.divide(tp,support,out=np.zeros(k,float),where=support!=0)
            f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(k,float),where=(precision+recall)!=0)
            key=f"{task}_{run['seed']}_{run['arm']}"
            np.savez_compressed(args.output/f'{key}_benchmark.npz',truth=truth,prediction=pred,confusion=confusion)
            metrics.append(dict(task=task,seed=run['seed'],arm=run['arm'],count=len(ids),accuracy=float((pred==truth).mean()),
                macro_f1=float(f1.mean()),validation_accuracy=run['selection']['validation_accuracy'],
                per_class=[dict(label=label,support=int(support[i]),precision=float(precision[i]),recall=float(recall[i]),f1=float(f1[i])) for i,label in enumerate(classes)]))
            for item in manifest:
                if (item['collection'] in ('single_digits','numbers'))!=(task=='mnist'):continue
                array=np.load(args.prepared/item['array'])
                labels=model(torch.from_numpy(array).unsqueeze(1).to(device)).argmax(1).cpu().tolist() if len(array) else []
                prediction=''.join(classes[i] for i in labels)
                photo_rows.append(dict(sample_id=item['sample_id'],collection=item['collection'],writer=item['writer'],
                    reference=item['reference'],prediction=prediction,task=task,seed=run['seed'],arm=run['arm'],
                    correct=int(prediction==item['reference']),edits=distance(item['reference'],prediction)))
            print(f'Evaluated {key}',flush=True)
    summary=[]
    for task in ('mnist','emnist'):
        for seed in args.seeds:
            for arm in ('control','augmented'):
                for collection in (('single_digits','numbers') if task=='mnist' else ('characters','strings')):
                    subset=[r for r in photo_rows if (r['task'],r['seed'],r['arm'],r['collection'])==(task,seed,arm,collection)]
                    summary.append(dict(task=task,seed=seed,arm=arm,collection=collection,count=len(subset),
                        correct=sum(r['correct'] for r in subset),cer=sum(r['edits'] for r in subset)/sum(len(r['reference']) for r in subset)))
    write(args.output/'benchmarks.json',metrics);write(args.output/'photo_predictions.json',photo_rows);write(args.output/'photo_summary.json',summary)
    write(args.output/'verification.json',dict(status='PASS',smoke=args.smoke,runs=len(runs),paired_initialisation_and_split=True,
        development_images=len(manifest),photo_predictions=len(photo_rows),fresh_photos_used=False))


if __name__=='__main__':main()
