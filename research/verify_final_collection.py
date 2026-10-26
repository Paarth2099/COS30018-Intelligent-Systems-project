"""Read-only photo integrity audit and review sheets; never runs recognition."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
from datetime import datetime, timezone
from PIL import Image, ImageOps, ImageDraw
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/final_evaluation/collection_review'
EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def inspect(path):
    with Image.open(path) as im:
        im.verify()
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert('RGB')
        pixel_hash = hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
        small = np.asarray(ImageOps.grayscale(im).resize((9, 8)), dtype=np.int16)
        dhash = (small[:, 1:] > small[:, :-1]).ravel()
        return im.size, pixel_hash, dhash

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT/'evidence/final_evaluation/protocol/collection_manifest.json').read_text())
    records=[]; vectors={}; matched=set(); groups=defaultdict(list)
    for row in manifest['samples']:
        candidates=[p for p in (ROOT/row['relative_folder']).iterdir() if p.is_file() and p.stem==row['sample_id'] and p.suffix.lower() in EXT]
        entry=dict(row)
        entry['status']='missing_or_multiple_files'
        if len(candidates)==1:
            path=candidates[0]; matched.add(path.resolve())
            entry.update(filename=path.name, sha256=digest(path), actual_path=path.relative_to(ROOT).as_posix())
            try:
                size,pixel_hash,vector=inspect(path)
                entry.update(width=size[0],height=size[1],pixel_sha256=pixel_hash,status='readable_pending_visual_review')
                vectors[entry['actual_path']]=vector
                groups[(row['writer'],row['collection'])].append(entry)
            except Exception as exc:
                entry.update(status='unreadable',error=str(exc))
        records.append(entry)
    raw=list((ROOT/'data/final_evaluation/raw').rglob('*'))
    unexpected=[p.relative_to(ROOT).as_posix() for p in raw if p.is_file() and p.resolve() not in matched]
    development=[]; errors=[]
    for folder in ['own_characters','own_strings','segmentation_photos']:
        for p in sorted((ROOT/'data'/folder).rglob('*')):
            if p.is_file() and p.suffix.lower() in EXT:
                try:
                    size,pixel_hash,vector=inspect(p)
                    rel=p.relative_to(ROOT).as_posix()
                    development.append(dict(actual_path=rel,sha256=digest(p),pixel_sha256=pixel_hash))
                    vectors[rel]=vector
                except Exception as exc: errors.append({'file':str(p),'error':str(exc)})
    duplicates=[]; near=[]
    valid=[r for r in records if 'pixel_sha256' in r]
    for i,a in enumerate(valid):
        for b in valid[i+1:]+development:
            scope='development' if b in development else 'final_collection'
            if a['sha256']==b['sha256'] or a['pixel_sha256']==b['pixel_sha256']:
                duplicates.append({'a':a['actual_path'],'b':b['actual_path'],'scope':scope})
            else:
                distance=int(np.count_nonzero(vectors[a['actual_path']]!=vectors[b['actual_path']]))
                if distance<=4: near.append({'a':a['actual_path'],'b':b['actual_path'],'scope':scope,'dhash_distance':distance})
    for (person,category),entries in groups.items():
        # Thumbnails are review aids; original bytes are never changed.
        sheet=Image.new('RGB',(1500, ((len(entries)+4)//5)*430),'white'); draw=ImageDraw.Draw(sheet)
        for i,r in enumerate(entries):
            x=(i%5)*300;y=(i//5)*430
            with Image.open(ROOT/r['actual_path']) as im:
                im=ImageOps.exif_transpose(im).convert('RGB'); im.thumbnail((290,365))
                sheet.paste(im,(x+(300-im.width)//2,y+45+(365-im.height)//2))
            draw.text((x+6,y+5),r['sample_id'],fill='black')
            draw.text((x+6,y+23),'Expected: '+r['planned_label'],fill='black')
        sheet.save(OUT/f'{person}_{category}.jpg',quality=95)
    output={'created_utc':datetime.now(timezone.utc).isoformat(),'recognition_run':False,'expected':120,'matched':len(matched),'readable':len(valid),'unexpected_files':unexpected,'development_images_checked':len(development),'development_scan_errors':errors,'exact_duplicate_pairs':duplicates,'perceptual_candidates':near,'perceptual_note':'dHash is only a weak screening heuristic; similar glyphs can collide and rephotographed writing can evade it. Visual inspection and writer confirmation are required.','samples':records}
    (OUT/'integrity_audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ['samples','perceptual_candidates']},indent=2))
    print('Perceptual candidates:',len(near))

if __name__=='__main__': main()
