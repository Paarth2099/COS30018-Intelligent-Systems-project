"""Paired augmentation study; no official test or fresh-photo inputs are loaded."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from research.train_models_week11 import (data, batch, score, select_device, synchronise,
    runtime, digit_model, character_model, CLASSES, sha256, indices_hash, write_json)


def augment(images,generator):
    """Independent modest affine/gain variation for each image, never flips.

    Randomness is CPU generated with its own seed; augmentation cannot change
    initialization or minibatch order. Grid sampling has no gradient inputs.
    """
    n=len(images)
    values=torch.rand(n,5,generator=generator)
    angle=(values[:,0]*2-1)*(10*np.pi/180)
    scale=.9+.2*values[:,1]
    theta=torch.zeros(n,2,3)
    theta[:,0,0]=angle.cos()/scale;theta[:,0,1]=-angle.sin()/scale
    theta[:,1,0]=angle.sin()/scale;theta[:,1,1]=angle.cos()/scale
    theta[:,:,2]=(values[:,2:4]*2-1)*.12
    theta=theta.to(images.device)
    grid=torch.nn.functional.affine_grid(theta,images.shape,align_corners=False)
    result=torch.nn.functional.grid_sample(images,grid,mode='bilinear',padding_mode='zeros',align_corners=False)
    gain=(.75+.25*values[:,4]).to(images.device).view(-1,1,1,1)
    return (result*gain).clamp(0,1)


def train(task,arm,seed,args,device):
    name='cnn' if task=='mnist' else 'resnet18'
    output=args.output/task/f'seed_{seed}'/arm
    if (output/'selection.json').exists():
        saved=json.loads((output/'selection.json').read_text())
        assert saved['checkpoint_sha256']==sha256(output/f'{name}.pt')
        assert saved['config']['epochs']==args.epochs and saved['config']['batch_size']==args.batch_size
        assert saved['config']['smoke']==args.smoke
        print(f'Completed run verified: {output}',flush=True);return
    if output.exists():raise ValueError(f'Incomplete run exists: {output}. Preserve it and choose a new --output.')
    images,targets,train_ids,val_ids=data(task)
    if args.smoke:train_ids=train_ids[:128];val_ids=val_ids[:64]
    output.mkdir(parents=True)
    sources=['research/train_improvements.py','research/train_models_week11.py','hnrs_ml/models.py',
             'hnrs_ml/architectures.py','hnrs_chars/models.py','hnrs_ml/data.py','hnrs_chars/data.py']
    source_root=Path(__file__).resolve().parents[1]
    for source in sources:
        target=output/'source'/source;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source_root/source,target)
    config=dict(task=task,model=name,arm=arm,seed=seed,epochs=args.epochs,batch_size=args.batch_size,
        optimizer='Adam',learning_rate=.001,loss='unweighted cross entropy',pretrained=False,
        augmentation='none' if arm=='control' else 'per-image rotation +/-10 deg, scale .9-1.1, translation +/-6% canvas, gain .75-1; bilinear; black fill; no flips',
        train_count=len(train_ids),validation_count=len(val_ids),train_indices_sha256=indices_hash(train_ids),
        validation_indices_sha256=indices_hash(val_ids),runtime=runtime(device),smoke=args.smoke,
        official_test_loaded=False,photo_data_loaded=False,source_sha256={s:sha256(source_root/s) for s in sources},
        selection='Highest validation accuracy, then lowest validation loss, then earlier epoch')
    write_json(output/'config.json',config)
    np.savez_compressed(output/'split_indices.npz',train=train_ids.numpy(),validation=val_ids.numpy())
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=(digit_model if task=='mnist' else character_model)(name).to(device)
    # Save a hash of initial weights for independent pairing verification.
    import hashlib
    digest=hashlib.sha256()
    for key,tensor in model.state_dict().items():digest.update(key.encode());digest.update(tensor.cpu().numpy().tobytes())
    config['initial_state_sha256']=digest.hexdigest();write_json(output/'config.json',config)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    order_rng=torch.Generator().manual_seed(seed+1)
    augmentation_rng=torch.Generator().manual_seed(seed+1000)
    history=[];best=(-1.,float('-inf'));start=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        epoch_start=time.perf_counter();model.train();total_loss=0.;correct=0;processed=0
        order=train_ids[torch.randperm(len(train_ids),generator=order_rng)]
        chunks=list(order.split(args.batch_size))
        if len(chunks)>1 and len(chunks[-1])==1:chunks[-2]=torch.cat((chunks[-2],chunks[-1]));chunks.pop()
        for step,ids in enumerate(chunks):
            x=batch(images,ids,device);y=targets[ids].to(device)
            if arm=='augmented':x=augment(x,augmentation_rng)
            optimizer.zero_grad(set_to_none=True);logits=model(x)
            loss=torch.nn.functional.cross_entropy(logits,y)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();optimizer.step()
            total_loss+=float(loss.detach())*len(ids);correct+=int((logits.detach().argmax(1)==y).sum());processed+=len(ids)
            if step%100==0:print(json.dumps(dict(task=task,arm=arm,seed=seed,epoch=epoch,processed=processed,total=len(order))),flush=True)
        val_loss,val_acc=score(model,images,targets,val_ids,args.batch_size,device);synchronise(device)
        row=dict(epoch=epoch,train_loss=total_loss/len(order),train_accuracy=correct/len(order),
            validation_loss=val_loss,validation_accuracy=val_acc,seconds=time.perf_counter()-epoch_start)
        history.append(row);write_json(output/'history.json',history)
        if (val_acc,-val_loss)>best:
            best=(val_acc,-val_loss);best_epoch=epoch
            checkpoint=dict(format_version=1,model_name=name,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                epoch=epoch,validation_accuracy=val_acc,validation_loss=val_loss,config=config)
            if task=='emnist':checkpoint.update(task='emnist_byclass_62',classes=CLASSES)
            torch.save(checkpoint,output/f'{name}.pt')
        print(json.dumps(dict(task=task,arm=arm,seed=seed,**row)),flush=True)
    write_json(output/'selection.json',dict(model=name,best_epoch=best_epoch,validation_accuracy=best[0],
        validation_loss=-best[1],checkpoint_sha256=sha256(output/f'{name}.pt'),config=config,
        training_seconds=time.perf_counter()-start,selected_before_test_evaluation=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('experiments/improvements_gpu_run01'))
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--task',choices=['all','mnist','emnist'],default='all')
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43])
    p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--smoke',action='store_true',help='Tiny plumbing test, NOT performance evidence')
    args=p.parse_args()
    if args.epochs<1 or args.batch_size<2 or len(set(args.seeds))!=len(args.seeds):p.error('Invalid run parameters')
    if args.smoke and 'smoke' not in args.output.name:p.error('Smoke output name must contain smoke')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    device=select_device(args.device);print(json.dumps(runtime(device)),flush=True)
    for task in ('mnist','emnist') if args.task=='all' else (args.task,):
        for seed in args.seeds:
            for arm in ('control','augmented'):train(task,arm,seed,args,device)


if __name__=='__main__':main()
