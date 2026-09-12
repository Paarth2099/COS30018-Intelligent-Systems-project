"""Run both segmentation methods on labelled development photos.

Labels are used only after segmentation for scoring. Count agreement is not
proof of correct segmentation. No ground-truth bounding boxes are available.
"""
import argparse
import csv
import hashlib
import json
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from segment_digits import METHODS, segment_image, prepare_crops


def edit_distance(a,b):
    prev=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        row=[i]
        for j,cb in enumerate(b,1):
            row.append(min(row[-1]+1,prev[j]+1,prev[j-1]+(ca!=cb)))
        prev=row
    return prev[-1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preview(seg,prepared,prediction,path):
    image=seg.gray.convert('RGB'); draw=ImageDraw.Draw(image)
    for i,box in enumerate(seg.boxes,1):
        draw.rectangle((box[0],box[1],box[2]-1,box[3]-1),outline='red',width=2)
        draw.text((box[0],max(0,box[1]-15)),str(i),fill='red')
    image.thumbnail((1000,450))
    width=max(640,image.width,len(prepared)*90)
    out=Image.new('RGB',(width,640),'white'); draw=ImageDraw.Draw(out)
    draw.text((10,8),'Detected regions in left-to-right order',fill='black')
    out.paste(image,(0,30))
    draw.text((10,490),'Prepared crops | Prediction: '+prediction,fill='black')
    for i,array in enumerate(prepared):
        im=Image.fromarray(np.rint(array*255).astype('uint8')).resize((84,84),Image.Resampling.NEAREST)
        out.paste(im,(i*90,520)); draw.text((i*90,610),str(i+1),fill='black')
    out.save(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--include-single',action='store_true')
    args=parser.parse_args()
    if not args.input.exists(): parser.error('Input does not exist')
    files=[args.input] if args.input.is_file() else sorted(args.input.rglob('*'))
    inputs=[]; skipped=[]
    for path in files:
        if not path.is_file() or path.suffix.lower() not in ('.jpg','.jpeg','.png','.bmp','.tif','.tiff'): continue
        match=re.search(r'label-([0-9]+)$',path.stem)
        if not match or (len(match[1])==1 and not args.include_single):
            skipped.append(str(path)); continue
        inputs.append((path,match[1]))
    if not inputs: parser.error('No eligible labelled photos found')
    if args.output.exists(): parser.error('Output already exists; choose a new run name')
    predictor=None
    if args.checkpoint:
        from hnrs_ml.predict import DigitPredictor
        predictor=DigitPredictor(args.checkpoint)
    args.output.mkdir(parents=True)
    rows=[]; manifest=[]
    for index,(path,label) in enumerate(inputs,1):
        manifest.append({'file':str(path.resolve()),'label':label,'sha256':digest(path)})
        for method in METHODS:
            out=args.output/f'{index:03d}_{path.stem}'/method; out.mkdir(parents=True)
            row={'file':str(path),'label':label,'method':method,'expected_count':len(label),
                 'detected_count':0,'count_correct':0,'prediction':'','exact_correct':0 if predictor else '',
                 'edit_distance':len(label) if predictor else '', 'segmentation_ms':'','warnings':'','error':''}
            try:
                with Image.open(path) as src: image=ImageOps.exif_transpose(src).convert('RGB')
                start=time.perf_counter(); seg=segment_image(image,method)
                row['segmentation_ms']=round(1000*(time.perf_counter()-start),3)
                row['detected_count']=len(seg.boxes); row['count_correct']=int(len(seg.boxes)==len(label))
                row['warnings']='; '.join(seg.warnings)
                Image.fromarray(seg.mask.astype('uint8')*255).save(out/'mask.png')
                (out/'boxes.json').write_text(json.dumps({'coordinate_space':'resized grayscale image',
                    'size':seg.gray.size,'boxes':seg.boxes,'removed_components':seg.removed_components},indent=2))
                for i,crop in enumerate(seg.crops): crop.save(out/f'crop_{i:02d}.png')
                arrays=prepare_crops(seg.crops,'otsu'); np.save(out/'prepared_crops.npy',arrays)
                for i,array in enumerate(arrays): Image.fromarray(np.rint(array*255).astype('uint8')).save(out/f'prepared_{i:02d}.png')
                if predictor:
                    result=predictor.predict_number(arrays)
                    (out/'prediction.json').write_text(json.dumps(result,indent=2))
                    row['prediction']=result['text']; row['exact_correct']=int(result['text']==label)
                    row['edit_distance']=edit_distance(label,result['text'])
                preview(seg,arrays,row['prediction'] or '(not predicted)',out/'preview.png')
            except (ValueError,OSError) as error:
                row['error']=str(error)
            rows.append(row)
            print(f'{path.name} | {method}: {row["detected_count"]}/{len(label)} crops; prediction={row["prediction"]}; error={row["error"]}',flush=True)
    with (args.output/'results.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary={}
    for method in METHODS:
        group=[r for r in rows if r['method']==method]
        summary[method]={'images':len(group),'count_matches':sum(r['count_correct'] for r in group),
            'count_accuracy':sum(r['count_correct'] for r in group)/len(group),
            'exact_matches':sum(r['exact_correct'] for r in group) if predictor else None,
            'exact_number_accuracy':sum(r['exact_correct'] for r in group)/len(group) if predictor else None,
            'character_error_rate':sum(r['edit_distance'] for r in group)/sum(len(r['label']) for r in group) if predictor else None,
            'errors':sum(bool(r['error']) for r in group),
            'mean_segmentation_ms':sum(r['segmentation_ms'] for r in group if r['segmentation_ms']!='')/len(group)}
    record={'created_utc':datetime.now(timezone.utc).isoformat(),'purpose':'Development comparison, not final evaluation',
        'summary':summary,'inputs':manifest,'skipped':skipped,'configuration':{'max_side':1000,'min_area_fraction':.01,'min_height_fraction':.1,
        'shared_mask':'autocontrast plus Otsu; small connected components removed','representation':'otsu'},
        'checkpoint':str(args.checkpoint) if args.checkpoint else None,'checkpoint_sha256':digest(args.checkpoint) if args.checkpoint else None,
        'source_sha256':{name:digest(Path(__file__).parent/name) for name in ['segment_digits.py','compare_segmentation.py','preprocess_digits.py']},
        'python':platform.python_version(),'platform':platform.platform(),
        'metric_note':'Count agreement is only a proxy. CER is Levenshtein edits divided by reference characters; may exceed 1. Inspect previews for actual crop quality.'}
    (args.output/'summary.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
