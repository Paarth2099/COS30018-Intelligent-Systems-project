"""Demonstrate preparation and prediction of one complete character photo."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from .data import sha256
from .predict import CharacterPredictor
from .preprocessing import METHODS, DEFAULT_METHOD, prepare_character_variants
from preprocess_digits import save_preview


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--method',choices=METHODS,default=DEFAULT_METHOD)
    parser.add_argument('--experiment',type=Path,default=Path('experiments/emnist_byclass/run_01'))
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new output folder to preserve prior evidence')
    torch.set_num_threads(4)
    selection=json.loads((args.experiment/'selection.json').read_text())
    checkpoint=args.experiment/(selection['selected_model']+'.pt')
    record=next(r for r in selection['models'] if r['model']==selection['selected_model'])
    if sha256(checkpoint)!=record['checkpoint_sha256']:raise ValueError('Checkpoint changed after model selection')
    with Image.open(args.image) as image:
        original=ImageOps.exif_transpose(image).convert('RGB')
    variants,crop,diagnostics=prepare_character_variants(original)
    array=variants[args.method]
    if not np.any(array):raise ValueError('Empty prepared character')
    prediction=CharacterPredictor(checkpoint).predict_text(array[None])
    args.output.mkdir(parents=True)
    save_preview(original,crop,variants,args.output/'comparison.png')
    np.save(args.output/'prepared.npy',array)
    result={'input':args.image.as_posix(),'input_sha256':sha256(args.image),
            'method':args.method,'checkpoint_sha256':sha256(checkpoint),
            'prediction':prediction,'diagnostics':diagnostics}
    (args.output/'prediction.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
