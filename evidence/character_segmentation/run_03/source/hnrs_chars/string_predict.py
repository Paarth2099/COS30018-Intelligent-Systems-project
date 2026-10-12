"""Run a complete photo-to-string demonstration without reference labels."""
import argparse
import json
from pathlib import Path
from PIL import Image
import torch
from .data import sha256
from .segmentation import METHODS, DEFAULT_METHOD, recognise_text
from .string_experiment import load_predictor, save_artifacts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--method',choices=METHODS,default=DEFAULT_METHOD)
    parser.add_argument('--experiment',type=Path,default=Path('experiments/emnist_byclass/run_01'))
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new output directory')
    torch.set_num_threads(4);predictor,checkpoint=load_predictor(args.experiment)
    with Image.open(args.image) as image:result=recognise_text(image,predictor,args.method)
    save_artifacts(args.output,result['segmentation'],result['prepared'],result['prediction'])
    output={'input':args.image.as_posix(),'input_sha256':sha256(args.image),
            'checkpoint_sha256':sha256(checkpoint),'method':args.method,
            'prediction':result['prediction'],'detected_characters':len(result['segmentation'].boxes),
            'warnings':result['segmentation'].warnings,'preview':(args.output/'preview.png').as_posix()}
    (args.output/'demo.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
