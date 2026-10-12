"""Classify upright prepared crops; does not segment images or infer spaces."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from hnrs_ml.predict import DigitPredictor
from .data import CLASSES
from .models import build_model


class CharacterPredictor:
    def __init__(self,checkpoint):
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if saved.get('format_version')!=1 or saved.get('task')!='emnist_byclass_62' or saved.get('classes')!=CLASSES:
            raise ValueError('Expected a 62-class ByClass character checkpoint with the verified mapping')
        self.model_name=saved['model_name'];self.model=build_model(self.model_name)
        self.model.load_state_dict(saved['state_dict']);self.model.eval()

    @torch.inference_mode()
    def predict_characters(self,crops,batch_size=256):
        if batch_size<=0:raise ValueError('Batch size must be positive')
        batch=DigitPredictor.prepare_batch(crops);results=[]
        for chunk in batch.split(batch_size):
            if len(chunk)==0:continue
            scores,ids=self.model(chunk).softmax(1).max(1)
            results.extend({'class_id':i,'character':CLASSES[i],'confidence':s}
                           for i,s in zip(ids.tolist(),scores.tolist()))
        return results

    def predict_text(self,ordered_crops):
        characters=self.predict_characters(ordered_crops)
        return {'text':''.join(item['character'] for item in characters),'characters':characters,
                'model':self.model_name,'task':'emnist_byclass_62'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--crops',type=Path,nargs='+',required=True)
    args=parser.parse_args();torch.set_num_threads(4);arrays=[]
    for path in args.crops:
        with Image.open(path) as image:
            if image.size!=(28,28):parser.error('Supply prepared upright 28x28 crops with light ink on black')
            arrays.append(np.asarray(image.convert('L'),dtype=np.float32)/255.)
    print(json.dumps(CharacterPredictor(args.checkpoint).predict_text(np.stack(arrays)),indent=2))


if __name__=='__main__':main()
