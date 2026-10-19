"""Train both character models; select only using stratified validation data."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random
import subprocess
import time
import numpy as np
import torch
from torch import nn
import torchvision
from .data import CLASSES, ORIENTATION, load_partition, batch_images, sha256, indices_hash
from .models import build_model


@torch.inference_mode()
def score(model, images, targets, ids, batch_size):
    model.eval(); total_loss=0.; correct=0
    for selected in ids.split(batch_size):
        logits=model(batch_images(images,selected))
        total_loss+=nn.functional.cross_entropy(logits,targets[selected],reduction='sum').item()
        correct+=int((logits.argmax(1)==targets[selected]).sum())
    return total_loss/len(ids),correct/len(ids)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('data'))
    parser.add_argument('--preparation',type=Path,default=Path('evidence/emnist/preparation'))
    parser.add_argument('--output',type=Path,default=Path('experiments/emnist_byclass/run_01'))
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--learning-rate',type=float,default=.001)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    if min(args.epochs,args.batch_size,args.threads)<=0 or args.learning_rate<=0:
        parser.error('Training parameters must be positive')
    if args.output.exists():parser.error('Choose a new experiment folder; never overwrite a run')
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    images,targets=load_partition(args.data_dir,train=True)
    report=json.loads((args.preparation/'dataset_report.json').read_text())
    splits=np.load(args.preparation/'split_indices.npz')
    train_ids=torch.from_numpy(splits['train']);val_ids=torch.from_numpy(splits['validation'])
    joined=torch.cat([train_ids,val_ids])
    if len(joined)!=len(targets) or not torch.equal(joined.sort().values,torch.arange(len(targets))):
        raise ValueError('Training and validation indices must be disjoint and cover the dataset')
    for key,indices in [('train',train_ids),('validation',val_ids)]:
        if indices_hash(indices)!=report[f'{key}_indices_sha256']:raise ValueError('Split hash mismatch')
    for name,expected in report['files'].items():
        if '-train-' in name and sha256(args.data_dir/'EMNIST/raw'/name)!=expected:
            raise ValueError('Training data hash differs from preparation')
    seed=report['seed'];args.output.mkdir(parents=True)
    source_dir=args.output/'source';source_dir.mkdir()
    sources={}
    for path in sorted(Path(__file__).parent.glob('*.py')):
        (source_dir/path.name).write_bytes(path.read_bytes());sources[path.name]=sha256(path)
    config={'created_utc':datetime.now(timezone.utc).isoformat(),'dataset':'EMNIST ByClass','classes':CLASSES,
            'orientation':ORIENTATION,'normalisation':'uint8 / 255','augmentation':'none','seed':seed,
            'epochs':args.epochs,'batch_size':args.batch_size,'learning_rate':args.learning_rate,
            'optimizer':'Adam','weight_decay':0,'loss':'unweighted cross entropy','device':'cpu','threads':args.threads,
            'training_count':len(train_ids),'validation_count':len(val_ids),'test_data_used':False,
            'train_indices_sha256':indices_hash(train_ids),'validation_indices_sha256':indices_hash(val_ids),
            'preparation_report_sha256':sha256(args.preparation/'dataset_report.json'),
            'python':platform.python_version(),'torch':str(torch.__version__),'torchvision':str(torchvision.__version__),
            'platform':platform.platform(),'source_sha256':sources,
            'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'selection_rule':'Highest validation accuracy; then lowest validation loss; retain earlier epoch on tie'}
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    history=[];summaries=[]
    for name in ('mlp','cnn'):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        model=build_model(name);optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
        generator=torch.Generator().manual_seed(seed+1);best=(-1.,float('-inf'));start=time.perf_counter()
        for epoch in range(1,args.epochs+1):
            epoch_start=time.perf_counter();model.train();loss_sum=0.;correct=0
            order=train_ids[torch.randperm(len(train_ids),generator=generator)]
            for batch_id,selected in enumerate(order.split(args.batch_size)):
                optimizer.zero_grad(set_to_none=True)
                logits=model(batch_images(images,selected));loss=nn.functional.cross_entropy(logits,targets[selected])
                loss.backward();optimizer.step()
                loss_sum+=float(loss.detach())*len(selected)
                correct+=int((logits.detach().argmax(1)==targets[selected]).sum())
                if batch_id and batch_id % 500 == 0:
                    print(f'{name} epoch {epoch}: {min((batch_id+1)*args.batch_size,len(order))}/{len(order)} images',flush=True)
            val_loss,val_acc=score(model,images,targets,val_ids,args.batch_size)
            row={'model':name,'epoch':epoch,'train_loss':loss_sum/len(order),'train_accuracy':correct/len(order),
                 'validation_loss':val_loss,'validation_accuracy':val_acc,'seconds':time.perf_counter()-epoch_start}
            history.append(row);key=(val_acc,-val_loss)
            if key>best:
                best=key;best_epoch=epoch
                torch.save({'format_version':1,'task':'emnist_byclass_62','classes':CLASSES,
                            'model_name':name,'state_dict':model.state_dict(),'epoch':epoch,
                            'config':config,'validation_accuracy':val_acc,'validation_loss':val_loss},args.output/f'{name}.pt')
            with (args.output/'history.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerows(history)
            print(json.dumps(row),flush=True)
        summaries.append({'model':name,'best_epoch':best_epoch,'validation_accuracy':best[0],
                          'validation_loss':-best[1],'training_seconds':time.perf_counter()-start,
                          'parameters':sum(p.numel() for p in model.parameters()),'checkpoint_sha256':sha256(args.output/f'{name}.pt')})
    selected=max(summaries,key=lambda r:(r['validation_accuracy'],-r['validation_loss']))
    selection={'selected_model':selected['model'],'selected_before_test_evaluation':True,'models':summaries,
               'config':config,'note':'Single seed and fixed budget; no test-driven tuning.'}
    (args.output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    print('Selected using validation:',selected['model'],flush=True)


if __name__=='__main__':main()
