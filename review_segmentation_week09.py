"""Recreate selected segmentation for visual review against frozen Week 9 evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from segment_digits import segment_image
from hnrs_chars.segmentation import segment_characters


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('evidence/photo_model_comparison/week09_run01'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a new output directory')
    summary=json.loads((args.run/'summary.json').read_text())
    for path,digest in summary['sources'].items():
        if sha(path)!=digest:raise ValueError(f'Baseline source changed: {path}')
    with (args.run/'paired_results.csv').open(newline='',encoding='utf-8') as f:pairs=list(csv.DictReader(f))
    args.output.mkdir(parents=True)
    rows=[];sheets=[]
    for index,p in enumerate(pairs):
        if p['collection'] not in ('numbers','strings'):continue
        assert sha(p['file'])==p['sha256']
        with Image.open(p['file']) as src:image=ImageOps.exif_transpose(src).convert('RGB')
        seg=segment_image(image,'components') if p['collection']=='numbers' else segment_characters(image,'grouped_components')
        assert [list(b) for b in seg.boxes]==json.loads(p['boxes'])
        arrays=np.load(args.run/'prepared'/f'{index:03d}.npy')
        assert len(arrays)==len(seg.crops)==int(p['crop_count'])
        folder=args.output/f'{index:03d}';folder.mkdir()
        annotated=seg.gray.convert('RGB');draw=ImageDraw.Draw(annotated)
        for i,b in enumerate(seg.boxes):
            draw.rectangle((b[0],b[1],b[2]-1,b[3]-1),outline='red',width=2)
            draw.text((b[0],max(0,b[1]-12)),str(i+1),fill='red')
        annotated.save(folder/'annotated.png')
        Image.fromarray(seg.mask.astype('uint8')*255).save(folder/'mask.png')
        for i,crop in enumerate(seg.crops):crop.save(folder/f'crop_{i:02d}.png')
        (folder/'segmentation.json').write_text(json.dumps(dict(boxes=seg.boxes,warnings=seg.warnings,
            groups=getattr(seg,'groups',None),coordinate_space='upright resized grayscale',size=seg.gray.size),indent=2)+'\n')
        panel=Image.new('RGB',(1000,570),'white');d=ImageDraw.Draw(panel)
        d.text((10,5),f"ID {index:03d} {p['file']}\nTrue={p['reference']} | MLP={p['mlp']} CNN={p['cnn']} | crops {len(seg.crops)}/{len(p['reference'])}",fill='black')
        a=annotated.copy();a.thumbnail((420,480));panel.paste(a,(10,65))
        d.text((450,55),'Raw crops in returned order',fill='black')
        d.text((450,310),'Frozen prepared crops (same inputs to both models)',fill='black')
        for i,(crop,array) in enumerate(zip(seg.crops,arrays)):
            tile=crop.convert('RGB');tile.thumbnail((80,205))
            panel.paste(tile,(450+i*88,90));d.text((450+i*88,75),str(i+1),fill='black')
            tile=Image.fromarray(np.rint(array*255).astype('uint8')).resize((80,80),Image.Resampling.NEAREST)
            panel.paste(tile,(450+i*88,340))
        panel.save(folder/'review.png')
        position=len(rows)%6
        if position==0:sheets.append(Image.new('RGB',(1400,1002),'#eeeeee'))
        small=panel.resize((700,334))
        sheets[-1].paste(small,((position%2)*700,(position//2)*334))
        rows.append(dict(index=index,file=p['file'],writer=p['writer'],collection=p['collection'],reference=p['reference'],
            mlp=p['mlp'],cnn=p['cnn'],expected_count=len(p['reference']),detected_count=len(seg.crops),
            count_match=int(len(seg.crops)==len(p['reference'])),left_to_right=int(seg.boxes==sorted(seg.boxes,key=lambda b:(b[0],b[1]))),
            warnings='; '.join(seg.warnings),input_sha256=p['sha256'],prepared_sha256=sha(args.run/'prepared'/f'{index:03d}.npy'),
            preview=f'{index:03d}/review.png',sheet=f'sheet_{len(rows)//6+1:02d}.png'))
        print(f'Review evidence {len(rows)}/60: {index:03d}',flush=True)
    assert len(rows)==60
    for i,sheet in enumerate(sheets,1):sheet.save(args.output/f'sheet_{i:02d}.png')
    with (args.output/'diagnostics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (args.output/'provenance.json').write_text(json.dumps(dict(source_run=args.run.as_posix(),
        paired_csv_sha256=sha(args.run/'paired_results.csv'),sources=summary['sources'],
        checkpoint_records=summary['checkpoints'],review_script_sha256=sha(__file__),
        purpose='60-photo visual review; recreated boxes exactly match saved baseline; predictions and prepared arrays reused'),indent=2)+'\n')
    shutil.copyfile(__file__,args.output/'review_segmentation_week09.py')


if __name__=='__main__':main()
