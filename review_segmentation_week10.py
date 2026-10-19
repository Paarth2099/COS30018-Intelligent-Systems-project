"""Create five-photo sheets for human review of complete ordered crops."""
import argparse
import csv
import json
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--variant',required=True);a=p.parse_args()
    with (a.run/'predictions.csv').open() as f:rows=[r for r in csv.DictReader(f) if r['variant']==a.variant and r['model']=='cnn']
    for page in range(0,len(rows),5):
        sheet=Image.new('RGB',(1250,1100),'white');draw=ImageDraw.Draw(sheet)
        for row,r in enumerate(rows[page:page+5]):
            y=row*220;folder=a.run/f'{int(r["index"]):03d}'/a.variant
            details=json.loads((folder/'segmentation.json').read_text())
            draw.text((5,y+3),f"{r['index']} {r['writer']} true={r['reference']} CNN={r['prediction']} crops={r['crop_count']}",fill='black')
            with Image.open(r['file']) as original:im=ImageOps.exif_transpose(original).convert('RGB')
            im.thumbnail((1200,1200));d=ImageDraw.Draw(im)
            for box in details['boxes']:d.rectangle(box,outline='red',width=2)
            im.thumbnail((410,190));sheet.paste(im,(5,y+22))
            for i in range(len(details['boxes'])):
                with Image.open(folder/f'crop_{i:02d}.png') as raw:
                    crop=raw.convert('RGB');crop.thumbnail((90,105));sheet.paste(crop,(430+i*105,y+22))
                with Image.open(folder/f'prepared_{i:02d}.png') as prepared:
                    sheet.paste(prepared.resize((70,70)),(430+i*105,y+137))
        sheet.save(a.run/f'{a.variant}_sheet_{page//5+1:02d}.png')
    print(f'{len(rows)} rows; {(len(rows)+4)//5} sheets')


if __name__=='__main__':main()
