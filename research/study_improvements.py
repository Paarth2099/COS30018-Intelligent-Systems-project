"""Bounded development-only study. Output is immutable; GUI remains unchanged."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image, ImageDraw
import torch
from research.compare_photo_models import discover, sha
from research.compare_segmentation import edit_distance
from segment_digits import segment_image, prepare_crops
from hnrs_chars.segmentation import segment_characters, prepare_character_crops
from hnrs_chars.predict import CharacterPredictor
from hnrs_ml.predict import DigitPredictor
from research.improvement_candidates import padded_character, segment_repaired_digits, mask_characters


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mask-study',action='store_true')
    args=parser.parse_args()
    if args.output.exists(): parser.error('Choose a new output directory')
    items=discover(); torch.set_num_threads(4)
    paths={'digits':Path('experiments/baseline/cnn.pt'),
           'characters':Path('experiments/week11_gpu_run01/emnist/resnet18/resnet18.pt')}
    args.output.mkdir(parents=True)
    # Record hypotheses and adoption rules before inference.
    write(args.output/'protocol.json',dict(
        purpose='Development experiment after v1 final error analysis, not fresh-test evidence',
        hypotheses={'padded':'15% pale margin may stabilise character illumination estimation',
                    'merged':'Bounded complementary fragment merging may avoid digit over-segmentation',
                    'mask':'Reuse segmentation foreground to avoid a second background estimate' if args.mask_study else 'not tested'},
        allowed_data_roots=['data/segmentation_photos','data/own_characters/development','data/own_strings/development'],
        final_photos_used=False,
        baseline='Current production routing, including segmentation of single letters',
        adoption='No exact-match or CER regression in either relevant collection; at least one strict gain; inspect changed crops before adoption',
        checkpoints={k:dict(path=str(v),sha256=sha(v)) for k,v in paths.items()},manifest=items))
    sources=['research/study_improvements.py','research/improvement_candidates.py','preprocess_digits.py','segment_digits.py',
             'hnrs_chars/preprocessing.py','hnrs_chars/segmentation.py']
    for source in sources:
        target=args.output/'source'/source;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    write(args.output/'source_hashes.json',{p:sha(p) for p in sources})
    predictors={'digits':DigitPredictor(paths['digits']).predict_number,
                'characters':CharacterPredictor(paths['characters']).predict_text}
    rows=[]
    for index,item in enumerate(items):
        digit=item['collection'] in ('single_digits','numbers')
        with Image.open(item['file']) as source: image=source.convert('RGB')
        segment=segment_image if digit else segment_characters
        base=segment(image)
        arrays=prepare_crops(base.crops) if digit else prepare_character_crops(base.crops)
        candidates={'baseline':(arrays,base.boxes)}
        if digit:
            new=segment_repaired_digits(image)
            candidates['merged']=(prepare_crops(new.crops),new.boxes)
        else:
            candidates['padded']=(np.stack([padded_character(c) for c in base.crops])
                if base.crops else np.empty((0,28,28),np.float32),base.boxes)
            if args.mask_study:candidates={'baseline':(arrays,base.boxes),'mask':(mask_characters(base),base.boxes)}
        canvas=Image.new('RGB',(1000,350),'white'); draw=ImageDraw.Draw(canvas)
        original=image.copy();original.thumbnail((300,280));canvas.paste(original,(5,55))
        draw.text((5,5),f"{item['file']} | reference: {item['reference']}",fill='black')
        for j,(name,(prepared,boxes)) in enumerate(candidates.items()):
            prediction=predictors['digits' if digit else 'characters'](prepared)['text']
            rows.append(dict(item,variant=name,prediction=prediction,correct=int(prediction==item['reference']),
                             edits=edit_distance(item['reference'],prediction),crop_count=len(prepared),boxes=boxes))
            draw.text((315,45+j*140),f'{name}: {prediction}',fill='black')
            for k,array in enumerate(prepared):
                canvas.paste(Image.fromarray(np.rint(array*255).astype('uint8')).resize((65,65)),(315+k*70,70+j*140))
            dest=args.output/'prepared'/f'{index:03d}_{name}.npy';dest.parent.mkdir(exist_ok=True)
            np.save(dest,prepared)
        dest=args.output/'previews'/f'{index:03d}.png';dest.parent.mkdir(exist_ok=True);canvas.save(dest)
        print(f'{index+1}/{len(items)} {item["collection"]}',flush=True)
    write(args.output/'predictions.json',rows)
    groups=defaultdict(list)
    for row in rows:groups[row['collection'],row['variant']].append(row)
    summary=[]
    for (collection,variant),subset in groups.items():
        summary.append(dict(collection=collection,variant=variant,count=len(subset),
            correct=sum(r['correct'] for r in subset),
            cer=sum(r['edits'] for r in subset)/sum(len(r['reference']) for r in subset),
            count_matches=sum(r['crop_count']==len(r['reference']) for r in subset)))
    write(args.output/'summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
