"""Train additional architectures on the original splits; official tests stay unloaded.

Each completed model has selection.json. Interrupted model runs must use a new
directory, so partial histories cannot silently be combined with a fresh run.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import time
# Set before importing Torch/initialising CUDA for deterministic cuBLAS operations.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torchvision
from hnrs_chars.data import CLASSES, sha256, indices_hash
from hnrs_ml.models import build_model as digit_model
from hnrs_chars.models import build_model as character_model

NAMES = ('resnet18', 'mobilenet_v3_small')
ROOT = Path('experiments/week11_four_models')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def data(task):
    if task == 'mnist':
        from torchvision.datasets import MNIST
        from hnrs_ml.data import development_split
        dataset=MNIST('data',train=True,download=False)
        images,targets=dataset.data,dataset.targets
        train,val=development_split()
    else:
        from hnrs_chars.data import load_partition
        images,targets=load_partition(Path('data'),train=True)
        split=np.load('results/datasets/emnist_split_indices.npz')
        train,val=(torch.from_numpy(split[key]) for key in ('train','validation'))
    joined=torch.cat((train,val))
    if not torch.equal(joined.sort().values,torch.arange(len(targets))):
        raise ValueError('Split must be disjoint and cover training partition')
    base=Path('experiments/baseline' if task=='mnist' else 'experiments/emnist_byclass/run_01')
    config=json.loads((base/'config.json').read_text())
    for key,ids in [('train',train),('validation',val)]:
        assert indices_hash(ids)==config[key+'_indices_sha256']
    return images,targets,train,val


def select_device(requested):
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable. Check the NVIDIA driver and CUDA-enabled PyTorch installation.')
    return torch.device('cuda' if requested == 'cuda' or (requested == 'auto' and torch.cuda.is_available()) else 'cpu')


def synchronise(device):
    if device.type == 'cuda': torch.cuda.synchronize(device)


def runtime(device):
    return dict(device=str(device),device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor(),
                cuda_runtime=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),
                deterministic_algorithms=True,precision='float32; AMP and TF32 disabled',
                platform=platform.platform(),torch=str(torch.__version__),torchvision=str(torchvision.__version__))


def batch(images,ids,device):
    # Keep the full dataset in CPU RAM; only transfer a minibatch to the GPU.
    return images[ids].unsqueeze(1).to(device=device,dtype=torch.float32).div_(255)


@torch.inference_mode()
def score(model,images,targets,ids,batch_size,device):
    model.eval();loss=0.;correct=0
    for chunk in ids.split(batch_size):
        labels=targets[chunk].to(device)
        logits=model(batch(images,chunk,device))
        loss+=torch.nn.functional.cross_entropy(logits,labels,reduction='sum').item()
        correct+=int((logits.argmax(1)==labels).sum())
    return loss/len(ids),correct/len(ids)


def pilot(output,device,batch_size):
    if output.exists(): raise ValueError('Pilot output already exists')
    rows=[]
    for name in NAMES:
        torch.manual_seed(42);model=character_model(name).to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        x=torch.rand(batch_size,1,28,28,device=device);y=torch.arange(batch_size,device=device)%62
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        times=[]
        for step in range(4):
            synchronise(device);start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(x),y);loss.backward();optimizer.step()
            synchronise(device)
            times.append(time.perf_counter()-start)
        if not torch.isfinite(loss):raise ValueError('Pilot produced nonfinite loss')
        model.eval()
        with torch.inference_mode():
            logits=model(x[:1])
            if logits.shape!=(1,62) or not torch.isfinite(logits).all():raise ValueError('Invalid pilot prediction')
        rows.append(dict(model=name,classes=62,batch_size=batch_size,runtime=runtime(device),parameters=sum(p.numel() for p in model.parameters()),
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                         warm_training_batch_seconds=float(np.median(times[1:])),
                         all_batch_seconds=times,finite_loss=bool(torch.isfinite(loss)),
                         note='Synthetic timing only; not accuracy evidence. Validation and IO add time.'))
        print(json.dumps(rows[-1]),flush=True)
        del model,optimizer,x,y,logits,loss
        if device.type=='cuda':torch.cuda.empty_cache()
    output.parent.mkdir(parents=True,exist_ok=True);write_json(output,rows)


def train(task,name,root,epochs,device,batch_size):
    output=root/task/name
    if (output/'selection.json').exists():
        selection=json.loads((output/'selection.json').read_text())
        assert sha256(output/f'{name}.pt')==selection['checkpoint_sha256']
        assert selection['config']['epochs']==epochs
        assert selection['config']['batch_size']==batch_size
        print('Verified completed run:',output,flush=True);return
    if output.exists():raise ValueError(f'Incomplete run exists: {output}. Preserve and choose a new --output.')
    images,targets,train_ids,val_ids=data(task)
    output.mkdir(parents=True)
    source_files=[Path(__file__),Path('hnrs_ml/architectures.py'),Path('hnrs_ml/models.py'),
                  Path('hnrs_chars/models.py'),Path('hnrs_ml/data.py'),Path('hnrs_chars/data.py')]
    sources={}
    for source in source_files:
        relative=source.resolve().relative_to(Path.cwd().resolve()) if source.is_absolute() else source
        destination=output/'source'/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,destination);sources[str(relative)]=sha256(source)
    try:
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):
        base_commit=None
    config=dict(task=task,model=name,epochs=epochs,seed=42,batch_size=batch_size,learning_rate=.001,
                optimizer='Adam',weight_decay=0,loss='unweighted cross entropy',augmentation='none',
                pretrained=False,input='float32 [N,1,28,28], uint8/255; no resizing',
                architecture='Torchvision default width/strides/pooling; first convolution changed to one channel; 10 or 62 outputs',
                train_count=len(train_ids),validation_count=len(val_ids),
                train_indices_sha256=indices_hash(train_ids),validation_indices_sha256=indices_hash(val_ids),
                device=str(device),runtime=runtime(device),threads=torch.get_num_threads(),torch=str(torch.__version__),torchvision=str(torchvision.__version__),
                python=platform.python_version(),source_sha256=sources,
                base_commit=base_commit,
                comparison_limitations='Single seed/fixed budget; historical baselines trained on CPU. GPU numerical results need not be bit-identical; historical training times are not a controlled speed comparison.',
                test_data_used=False,fresh_photos_used=False,
                selection_rule='Highest validation accuracy, then lowest validation loss, earlier epoch on tie')
    write_json(output/'config.json',config)
    np.savez_compressed(output/'split_indices.npz',train=train_ids.numpy(),validation=val_ids.numpy())
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    model=(digit_model if task=='mnist' else character_model)(name).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    generator=torch.Generator().manual_seed(43)
    best=(-1.,float('-inf'));history=[];started=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train();epoch_start=time.perf_counter();loss_sum=0.;correct=0
        order=train_ids[torch.randperm(len(train_ids),generator=generator)]
        # BatchNorm needs >1 image at the final 1x1 feature map in training.
        chunks=list(order.split(batch_size))
        if len(chunks[-1])==1:
            chunks[-2]=torch.cat((chunks[-2],chunks[-1]));chunks.pop()
        processed=0
        for i,chunk in enumerate(chunks):
            optimizer.zero_grad(set_to_none=True)
            labels=targets[chunk].to(device)
            logits=model(batch(images,chunk,device));loss=torch.nn.functional.cross_entropy(logits,labels)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();optimizer.step()
            loss_sum+=loss.item()*len(chunk);correct+=int((logits.detach().argmax(1)==labels).sum())
            processed+=len(chunk)
            if i%100==0:
                progress=dict(task=task,model=name,device=str(device),epoch=epoch,processed=processed,total=len(order),elapsed_seconds=time.perf_counter()-epoch_start)
                write_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
        val_loss,val_acc=score(model,images,targets,val_ids,batch_size,device)
        synchronise(device)
        row=dict(epoch=epoch,train_loss=loss_sum/len(order),train_accuracy=correct/len(order),validation_loss=val_loss,
                 validation_accuracy=val_acc,seconds=time.perf_counter()-epoch_start)
        history.append(row)
        with (output/'history.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerows(history)
        if (val_acc,-val_loss)>best:
            best=(val_acc,-val_loss);best_epoch=epoch
            saved=dict(format_version=1,model_name=name,state_dict={key:value.detach().cpu() for key,value in model.state_dict().items()},epoch=epoch,
                       validation_accuracy=val_acc,validation_loss=val_loss,config=config)
            if task=='emnist':saved.update(task='emnist_byclass_62',classes=CLASSES)
            torch.save(saved,output/f'{name}.pt')
        print(json.dumps(dict(task=task,model=name,**row)),flush=True)
    write_json(output/'selection.json',dict(model=name,best_epoch=best_epoch,validation_accuracy=best[0],validation_loss=-best[1],
               training_seconds=time.perf_counter()-started,parameters=sum(p.numel() for p in model.parameters()),
               checkpoint_sha256=sha256(output/f'{name}.pt'),selected_before_test_evaluation=True,config=config))
    write_json(output/'progress.json',dict(status='completed',task=task,model=name,device=str(device),epochs=epochs,best_epoch=best_epoch))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=('mnist','emnist','all'),default='all')
    parser.add_argument('--models',nargs='+',choices=NAMES,default=list(NAMES))
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--output',type=Path,default=ROOT)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    parser.add_argument('--batch-size',type=int,default=256)
    args=parser.parse_args()
    if args.epochs<1:parser.error('Epochs must be positive')
    if args.batch_size<2:parser.error('Batch size must be at least two for BatchNorm')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    device=select_device(args.device)
    print(json.dumps(runtime(device)),flush=True)
    if args.pilot:pilot(args.output/f'pilot_{device.type}.json',device,args.batch_size);return
    for task in ('mnist','emnist') if args.task=='all' else (args.task,):
        for name in args.models:train(task,name,args.output,args.epochs,device,args.batch_size)


if __name__=='__main__':main()
