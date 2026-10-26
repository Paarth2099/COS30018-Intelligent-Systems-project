"""Run from the PC project root; verify transfer, splits and CUDA before training."""
import argparse,json,hashlib
from pathlib import Path
import torch
from research.train_models_week11 import data
from research.train_improvements import augment


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cpu-check',action='store_true')
    p.add_argument('--package',type=Path,help='Optional original GPU transfer bundle to verify')
    args=p.parse_args()
    if args.package:
        package=args.package.resolve()
        digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for name,expected in json.loads((package/'package_hashes.json').read_text()).items():
            assert digest(package/name)==expected,f'Transfer differs: {name}'
        for name,expected in json.loads((package/'required_project_files.json').read_text()).items():
            assert Path(name).is_file(),f'Missing in PC project: {name}'
            assert digest(name)==expected,f'PC split/config differs: {name}'
        print('Original transfer package hashes verified.')
    if not args.cpu_check and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; use the working GPU environment from Week 11')
    for task in ('mnist','emnist'):
        images,targets,training,validation=data(task)
        print(f'{task}: {len(training)} training / {len(validation)} validation; split verified',flush=True)
        del images,targets,training,validation
    device='cpu' if args.cpu_check else 'cuda'
    x=torch.zeros(8,1,28,28,device=device);x[:,:,6:22,12:16]=1
    y=augment(x,torch.Generator().manual_seed(42))
    assert y.shape==x.shape and torch.isfinite(y).all()
    print(f'PASS: project splits, datasets and augmentation on {device}.')
    if args.cpu_check:print('CPU preflight does not verify CUDA on the RTX PC.')


if __name__=='__main__':main()
