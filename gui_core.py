"""Image acquisition and evidence export for Saniru's desktop interface."""
import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from preprocess_digits import prepare_variants

IMAGE_TYPES = {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}


def open_photo(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert('RGB')


def digit_catalog(folder):
    """Index single-digit labels only; never use number photos as digit tiles."""
    folder=Path(folder)
    if not folder.is_dir(): raise ValueError('Select a folder containing individual digit images.')
    catalog={str(i):[] for i in range(10)}
    for path in sorted(folder.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in IMAGE_TYPES: continue
        match=re.search(r'(?:^|_label-)([0-9])$',path.stem)
        if match: catalog[match[1]].append(path)
    return catalog


def generate_number(folder,text,seed=42):
    """Assemble actual handwriting samples. Requested text is NOT a prediction."""
    if not re.fullmatch(r'[0-9]{1,12}',text):
        raise ValueError('Enter 1 to 12 digits (0-9). Leading zeros are allowed.')
    catalog=digit_catalog(folder)
    missing=sorted(set(text)-{digit for digit,paths in catalog.items() if paths})
    if missing: raise ValueError('No single-digit image for: '+', '.join(missing)+'. Use filenames such as Name_01_label-7.jpg or 7.png.')
    rng=random.Random(seed)
    canvas=Image.new('RGB',(32+len(text)*112+(len(text)-1)*16,144),'white')
    sources=[]; warnings=[]
    for i,digit in enumerate(text):
        path=rng.choice(catalog[digit])
        try:
            variants,_,warning=prepare_variants(open_photo(path))
        except (ValueError,OSError) as exc:
            raise ValueError(f'Cannot prepare {path.name}: {exc}') from exc
        tile=Image.fromarray(255-np.rint(variants['otsu']*255).astype('uint8')).resize((112,112),Image.Resampling.NEAREST)
        canvas.paste(tile,(16+i*128,16))
        sources.append({'digit':digit,'file':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        if warning: warnings.append(f'{path.name}: {warning}')
    return canvas,{'route':'generated','requested_text':text,'seed':seed,'sources':sources,'warnings':warnings,
                   'note':'Composed from labelled handwriting crops; requested text was not supplied to the recogniser.'}


def annotated_image(segmented):
    image=segmented.gray.convert('RGB'); draw=ImageDraw.Draw(image)
    for i,(left,top,right,bottom) in enumerate(segmented.boxes,1):
        draw.rectangle((left,top,right-1,bottom-1),outline='#d32828',width=2)
        draw.text((left,max(0,top-14)),str(i),fill='#d32828')
    return image


def export_result(parent,image,origin,result,settings):
    """Create a fresh evidence directory; never replace previous experiments."""
    destination=Path(parent)/('gui_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    destination.mkdir(parents=True,exist_ok=False)
    image.save(destination/'input.png')
    seg=result['segmentation']; arrays=result['prepared']
    annotated_image(seg).save(destination/'detected_digits.png')
    Image.fromarray(seg.mask.astype('uint8')*255).save(destination/'foreground_mask.png')
    np.save(destination/'prepared_crops.npy',arrays)
    for index,(crop,array) in enumerate(zip(seg.crops,arrays)):
        crop.save(destination/f'crop_{index:02d}.png')
        Image.fromarray(np.rint(array*255).astype('uint8')).save(destination/f'prepared_{index:02d}.png')
    record={'created_utc':datetime.now(timezone.utc).isoformat(),'input':origin,'settings':settings,
            'prediction':result['prediction'],'boxes':seg.boxes,'coordinate_space':'resized grayscale image',
            'resized_size':seg.gray.size,'warnings':seg.warnings,
            'score_note':'Per-digit confidence is an uncalibrated model score, not a verified probability.'}
    (destination/'result.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    return destination
