"""Build diagnostic sheets from frozen Week 9 arrays; no pipeline changes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('evidence/photo_model_comparison/week09_run01'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a new output directory')
    with (args.run/'paired_results.csv').open(newline='',encoding='utf-8') as f:
        pairs=list(csv.DictReader(f))
    args.output.mkdir(parents=True)
    records=[];sheets=[]
    for index,p in enumerate(pairs):
        if p['collection'] not in ('single_digits','characters'):continue
        path=Path(p['file'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==p['sha256']
        array_path=args.run/'prepared'/f'{index:03d}.npy'
        batch=np.load(array_path)
        assert batch.shape==(1,28,28)
        a=batch[0];ys,xs=np.nonzero(a>.2)
        assert np.isfinite(a).all() and a.min()>=0 and a.max()<=1
        record=dict(index=index,file=p['file'],collection=p['collection'],writer=p['writer'],reference=p['reference'],
                    mlp=p['mlp'],cnn=p['cnn'],mlp_correct=p['mlp_correct'],cnn_correct=p['cnn_correct'],
                    input_sha256=p['sha256'],prepared_sha256=hashlib.sha256(array_path.read_bytes()).hexdigest(),
                    ink_width=int(xs.max()-xs.min()+1) if len(xs) else 0,
                    ink_height=int(ys.max()-ys.min()+1) if len(ys) else 0,
                    ink_fraction=float((a>.2).mean()),
                    centroid_x=float((a*np.arange(28)[None,:]).sum()/a.sum()),
                    centroid_y=float((a*np.arange(28)[:,None]).sum()/a.sum()),
                    sheet=f'sheet_{len(records)//10+1:02d}.png')
        records.append(record)
        position=(len(records)-1)%10
        if position==0:sheets.append(Image.new('RGB',(1000,1000),'#f5f5f5'))
        sheet=sheets[-1];draw=ImageDraw.Draw(sheet)
        x=(position%2)*500;y=(position//2)*200
        draw.text((x+8,y+4),f"ID {index:03d} {p['writer']} {p['collection']} true={p['reference']} MLP={p['mlp']} CNN={p['cnn']}",fill='black')
        with Image.open(path) as src: original=ImageOps.exif_transpose(src).convert('RGB')
        original.thumbnail((200,165));sheet.paste(original,(x+8,y+26))
        tile=Image.fromarray(np.rint(a*255).astype('uint8')).resize((140,140),Image.Resampling.NEAREST)
        sheet.paste(tile,(x+240,y+32))
        draw.text((x+390,y+45),f"ink >0.2\n{record['ink_width']}x{record['ink_height']}\n{record['ink_fraction']:.2f}",fill='black')
    for i,sheet in enumerate(sheets,1):sheet.save(args.output/f'sheet_{i:02d}.png')
    with (args.output/'diagnostics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (args.output/'provenance.json').write_text(json.dumps(dict(
        purpose='Descriptive visual preprocessing review, not a new accuracy experiment',
        reviewed_images=len(records),source_run=args.run.as_posix(),
        paired_csv_sha256=hashlib.sha256((args.run/'paired_results.csv').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        diagnostic_note='Ink geometry uses intensity >0.2 for description only; not a quality threshold or an algorithm change. Centroid uses intensity weights. Predictions are copied unchanged from the source run.'),indent=2)+'\n')
    (args.output/'review_preprocessing_week09.py').write_bytes(Path(__file__).read_bytes())
    print(f'Created {len(sheets)} sheets for {len(records)} photos')


if __name__=='__main__':main()
