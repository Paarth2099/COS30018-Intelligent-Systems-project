"""Compare three representations on labelled character DEVELOPMENT photos."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re
import shutil
import time
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from .data import sha256
from .predict import CharacterPredictor
from .preprocessing import METHODS, prepare_character_variants
from preprocess_digits import save_preview

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def label_from_path(path):
    match = re.fullmatch(r'([A-Za-z]+)_(\d+)_(digit|upper|lower)-([A-Za-z0-9])', path.stem)
    if not match:
        raise ValueError(f'Invalid label filename: {path.name}')
    writer, _, category, label = match.groups()
    permitted = {'digit': '0123456789', 'upper': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                 'lower': 'abcdefghijklmnopqrstuvwxyz'}
    if label not in permitted[category]:
        raise ValueError(f'Category/case mismatch: {path.name}')
    if path.parent.name != writer:
        raise ValueError(f'Writer folder does not match filename: {path}')
    return writer, category, label


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def aggregate(rows):
    correct = sum(r['correct'] for r in rows)
    return {'total': len(rows), 'correct': correct, 'accuracy': correct / len(rows),
            'preprocessing_failures': sum(bool(r['error']) for r in rows),
            'warning_images': sum(bool(r['warnings']) for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/own_characters/development'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--experiment', type=Path, default=Path('experiments/emnist_byclass/run_01'))
    args = parser.parse_args()
    if args.output.exists(): parser.error('Choose a new output folder; previous evidence is preserved')
    paths = sorted(p for p in args.input.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not paths: parser.error('No development photos found')
    manifest = []
    for path in paths:
        writer, category, label = label_from_path(path)
        with Image.open(path) as image:
            image.load()
            width, height = ImageOps.exif_transpose(image).size
        manifest.append({'file': path.relative_to(args.input).as_posix(), 'writer': writer,
                         'category': category, 'label': label, 'sha256': sha256(path),
                         'width': width, 'height': height})
    selection = json.loads((args.experiment / 'selection.json').read_text())
    name = selection['selected_model']; checkpoint = args.experiment / f'{name}.pt'
    selected_record = next(r for r in selection['models'] if r['model'] == name)
    if sha256(checkpoint) != selected_record['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from frozen validation-selected model')
    torch.set_num_threads(4)
    predictor = CharacterPredictor(checkpoint)
    predictor.predict_characters(np.zeros((1,28,28), np.float32))
    args.output.mkdir(parents=True)
    write_csv(args.output / 'manifest.csv', manifest)
    source_dir = args.output / 'source'; source_dir.mkdir()
    sources = {}
    for path in [Path(__file__), Path(__file__).with_name('preprocessing.py'),
                 Path(__file__).with_name('predict.py'), Path('preprocess_digits.py')]:
        shutil.copyfile(path, source_dir / path.name); sources[path.name] = sha256(path)
    rows = []; galleries = {}
    for path, sample in zip(paths, manifest):
        directory = args.output / sample['writer'] / path.stem; directory.mkdir(parents=True)
        original = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
        start = time.perf_counter()
        try:
            variants, crop, diagnostics = prepare_character_variants(original)
            elapsed = time.perf_counter() - start
            save_preview(original, crop, variants, directory / 'comparison.png')
            (directory / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2)+'\n')
            errors = {m: '' if np.any(a) else 'Empty prepared character' for m,a in variants.items()}
        except (ValueError, OSError) as error:
            elapsed = time.perf_counter() - start
            variants = {}; diagnostics = {'warnings': []}; errors = {m:str(error) for m in METHODS}
        for method in METHODS:
            error = errors[method]; prediction = ''; confidence = ''; inference = 0.
            if not error:
                array = variants[method]
                np.save(directory / f'{method}.npy', array)
                Image.fromarray(np.rint(array*255).astype('uint8')).save(directory / f'{method}.png')
                start = time.perf_counter()
                item = predictor.predict_characters(array[None])[0]
                inference = time.perf_counter() - start
                prediction, confidence = item['character'], item['confidence']
            rows.append({**{k:sample[k] for k in ('file','writer','category','label')},
                         'method':method, 'prediction':prediction, 'confidence':confidence,
                         'correct':int(prediction == sample['label']), 'error':error,
                         'warnings':'; '.join(diagnostics['warnings']),
                         'shared_preprocessing_seconds':elapsed, 'inference_seconds':inference})
        galleries.setdefault(sample['writer'], []).append((sample, original.copy(), variants, rows[-3:]))
    write_csv(args.output / 'results.csv', rows)
    scores = {m:aggregate([r for r in rows if r['method']==m]) for m in METHODS}
    # Declared tie preference: adaptive threshold, fixed threshold, then grayscale.
    preference = ('otsu','fixed','grayscale')
    winner = max(preference, key=lambda m:scores[m]['correct'])
    breakdown = []
    for field in ('writer','category','label'):
        for group in sorted({r[field] for r in rows}):
            for method in METHODS:
                breakdown.append({'group_type':field,'group':group,'method':method,
                                  **aggregate([r for r in rows if r[field]==group and r['method']==method])})
    write_csv(args.output / 'breakdown.csv', breakdown)
    confusions = Counter((r['label'],r['prediction']) for r in rows if r['method']==winner and not r['correct'] and not r['error'])
    result = {'created_utc':datetime.now(timezone.utc).isoformat(),
              'purpose':'Development method selection; not an untouched test or end-to-end string benchmark',
              'design':'Common autocontrast/Otsu bounding box and 20px aspect-preserving resize on 28px canvas; three pixel representations',
              'sample_count':len(paths), 'writers':dict(Counter(r['writer'] for r in manifest)),
              'labels':dict(sorted(Counter(r['label'] for r in manifest).items())),
              'duplicate_hash_groups':[[r['file'] for r in manifest if r['sha256']==h] for h,c in Counter(r['sha256'] for r in manifest).items() if c>1],
              'methods':scores, 'selected_method':winner,
              'selection_rule':'Highest correct count on all development photos; ties prefer otsu, fixed, grayscale',
              'checkpoint':checkpoint.as_posix(), 'checkpoint_sha256':sha256(checkpoint),
              'model':name, 'source_sha256':sources, 'python':platform.python_version(),
              'numpy':np.__version__, 'torch':str(torch.__version__),
              'selected_confusions':[{'true':a,'predicted':b,'count':c} for (a,b),c in confusions.most_common()],
              'timing_note':'Preprocessing computes all three shared variants once per photo; inference is warmed single-image CPU time; excludes file loading and evidence export'}
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for writer, samples in galleries.items():
        canvas = Image.new('RGB',(1000,60+len(samples)*110),'white'); draw=ImageDraw.Draw(canvas)
        draw.text((10,10),f'{writer} | reference and original | grayscale | fixed | otsu',fill='black')
        for i,(sample,original,variants,records) in enumerate(samples):
            y=50+i*110
            draw.text((5,y),f"true: {sample['label']}",fill='black')
            original.thumbnail((120,95));canvas.paste(original,(90,y))
            for j,row in enumerate(records):
                x=250+j*245
                if row['method'] in variants:
                    tile=Image.fromarray(np.rint(variants[row['method']]*255).astype('uint8')).resize((84,84),Image.Resampling.NEAREST)
                    canvas.paste(tile,(x,y))
                draw.text((x+90,y+10),f"{row['method']}\npred: {row['prediction']}\n{'OK' if row['correct'] else 'ERROR'}",fill='green' if row['correct'] else 'red')
        canvas.save(args.output/f'{writer}_overview.png')
    print(json.dumps({'methods':scores,'selected_method':winner},indent=2))
    print('Evidence:',args.output)


if __name__ == '__main__': main()
