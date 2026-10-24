"""Controlled five-epoch CNN experiment: square-root inverse class-frequency loss."""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import platform
import random
import shutil
import subprocess
import time
import numpy as np
import torch
from torch import nn
from .data import CLASSES,ORIENTATION,load_partition,batch_images,sha256,indices_hash
from .models import build_model
from .train import score


def class_weights(targets, train_ids):
    counts=torch.bincount(targets[train_ids],minlength=len(CLASSES)).float()
    if (counts==0).any():raise ValueError('Every class must occur in training')
    weights=counts.rsqrt()
    return weights/weights.mean(),counts.to(torch.int64)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Use a new output directory')
    base=Path('experiments/emnist_byclass/run_01')
    config=json.loads((base/'config.json').read_text())
    preparation=Path('evidence/emnist/preparation')
    dataset=json.loads((preparation/'dataset_report.json').read_text())
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    images,targets=load_partition(Path('data'),train=True)
    split=np.load(preparation/'split_indices.npz')
    train_ids=torch.from_numpy(split['train']);val_ids=torch.from_numpy(split['validation'])
    assert torch.equal(torch.cat([train_ids,val_ids]).sort().values,torch.arange(len(targets)))
    for name,ids in [('train',train_ids),('validation',val_ids)]:
        assert indices_hash(ids)==dataset[f'{name}_indices_sha256']==config[f'{name}_indices_sha256']
    for name,h in dataset['files'].items():
        if '-train-' in name:assert sha256(Path('data/EMNIST/raw')/name)==h
    baseline_record=next(x for x in json.loads((base/'selection.json').read_text())['models'] if x['model']=='cnn')
    assert sha256(base/'cnn.pt')==baseline_record['checkpoint_sha256']
    weights,counts=class_weights(targets,train_ids)
    args.output.mkdir(parents=True)
    sources={}
    for path in [*Path('hnrs_chars').glob('*.py'),Path('hnrs_ml/predict.py')]:
        dest=args.output/'source'/path;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest);sources[path.as_posix()]=sha256(path)
    plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),purpose='Week 10 controlled model experiment',
        hypothesis='Moderate inverse-frequency weighting may reduce majority-digit bias and improve minority-letter recognition; overall accuracy may regress.',
        single_change='Cross-entropy class weights = 1/sqrt(training class count), normalised to mean 1',
        architecture='unchanged cnn',seed=config['seed'],epochs=5,batch_size=256,learning_rate=.001,
        optimizer='Adam',weight_decay=0,augmentation='none',device='cpu',threads=4,
        orientation=ORIENTATION,classes=CLASSES,training_count=len(train_ids),validation_count=len(val_ids),
        train_indices_sha256=indices_hash(train_ids),validation_indices_sha256=indices_hash(val_ids),
        selection_rule='Highest unweighted validation accuracy; then lowest unweighted validation loss; earlier epoch on tie',
        adoption_rule='Do not deploy automatically. Prefer candidate only if validation accuracy and frozen-photo exact accuracy do not regress, with string CER no worse. Otherwise retain baseline and report tradeoffs.',
        test_data_used=False,fresh_photo_data_used=False,baseline_checkpoint_sha256=sha256(base/'cnn.pt'),
        source_sha256=sources,python=platform.python_version(),torch=str(torch.__version__),
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    with (args.output/'class_weights.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['class_id','character','train_count','loss_weight'])
        w.writerows((i,c,int(counts[i]),float(weights[i])) for i,c in enumerate(CLASSES))
    seed=config['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=build_model('cnn');optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    generator=torch.Generator().manual_seed(seed+1)
    history=[];best=(-1.,float('-inf'));started=time.perf_counter()
    log=(args.output/'training_log.txt').open('w',encoding='utf-8')
    def emit(message):
        print(message,flush=True);log.write(message+'\n');log.flush()
    for epoch in range(1,6):
        t=time.perf_counter();model.train();correct=0;loss_sum=0
        order=train_ids[torch.randperm(len(train_ids),generator=generator)]
        for batch_id,ids in enumerate(order.split(256)):
            optimizer.zero_grad(set_to_none=True)
            logits=model(batch_images(images,ids))
            loss=nn.functional.cross_entropy(logits,targets[ids],weight=weights)
            loss.backward();optimizer.step()
            loss_sum+=float(loss.detach())*len(ids)
            correct+=int((logits.detach().argmax(1)==targets[ids]).sum())
            if batch_id and batch_id%500==0:emit(f'epoch {epoch}: {min((batch_id+1)*256,len(order))}/{len(order)}')
        val_loss,val_acc=score(model,images,targets,val_ids,256)
        row=dict(epoch=epoch,train_weighted_batch_mean_loss=loss_sum/len(order),train_accuracy=correct/len(order),
            validation_unweighted_loss=val_loss,validation_accuracy=val_acc,seconds=time.perf_counter()-t)
        history.append(row)
        if (val_acc,-val_loss)>best:
            best=(val_acc,-val_loss);best_epoch=epoch
            torch.save(dict(format_version=1,task='emnist_byclass_62',classes=CLASSES,model_name='cnn',
                state_dict=model.state_dict(),epoch=epoch,config=plan,validation_accuracy=val_acc,validation_loss=val_loss),args.output/'cnn.pt')
        with (args.output/'history.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(history)
        emit(json.dumps(row))
    record=dict(model='cnn',best_epoch=best_epoch,validation_accuracy=best[0],validation_loss=-best[1],
        training_seconds=time.perf_counter()-started,checkpoint_sha256=sha256(args.output/'cnn.pt'))
    (args.output/'selection.json').write_text(json.dumps(dict(selected_model='cnn',selected_before_test_evaluation=True,
        models=[record],config=plan,note='Validation-selected candidate only; no application default change'),indent=2)+'\n')
    emit('COMPLETE '+json.dumps(record));log.close()


if __name__=='__main__':main()
