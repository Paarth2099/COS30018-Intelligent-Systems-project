"""Recognise one photo, without requiring a label in its filename."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from hnrs_ml.predict import DigitPredictor
from segment_digits import recognise_number
from compare_segmentation import preview


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--method',choices=['components','projection'],default='components')
    parser.add_argument('--checkpoint',type=Path,default=Path(__file__).resolve().parent/'experiments/baseline/cnn.pt')
    args=parser.parse_args()
    if args.output.exists(): parser.error('Output exists: use a new folder name')
    predictor=DigitPredictor(args.checkpoint)
    with Image.open(args.input) as im:
        result=recognise_number(ImageOps.exif_transpose(im).convert('RGB'),predictor,method=args.method)
    args.output.mkdir(parents=True)
    seg=result['segmentation']; arrays=result['prepared']; prediction=result['prediction']
    for index,crop in enumerate(seg.crops): crop.save(args.output/f'crop_{index:02d}.png')
    np.save(args.output/'prepared_crops.npy',arrays)
    preview(seg,arrays,prediction['text'] or '(no digits detected)',args.output/'preview.png')
    record={**prediction,'method':args.method,'boxes':seg.boxes,'box_coordinate_space':'resized grayscale image',
            'resized_image_size':seg.gray.size,'warnings':seg.warnings}
    (args.output/'prediction.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))
    print('Saved preview and ordered crops to',args.output)


if __name__=='__main__': main()
