"""Paired MLP/CNN evaluation on existing development photos (not final tests)."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch

from compare_segmentation import edit_distance
from hnrs_chars.photo_experiment import EXTENSIONS, label_from_path, write_csv
from hnrs_chars.preprocessing import prepare_character
from hnrs_chars.predict import CharacterPredictor
from hnrs_chars.segmentation import segment_characters, prepare_character_crops
from hnrs_chars.string_experiment import parse_label
from hnrs_ml.predict import DigitPredictor
from segment_digits import segment_image, prepare_crops


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def discover():
    items = []
    for root, kind in [('data/segmentation_photos', 'digits'),
                       ('data/own_characters/development', 'characters'),
                       ('data/own_strings/development', 'strings')]:
        for path in sorted(Path(root).rglob('*')):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            if kind == 'digits':
                match = re.fullmatch(r'([A-Za-z]+)_\d+_label-([0-9]+)', path.stem)
                if not match or match[1] != path.parent.name:
                    raise ValueError(f'Invalid digit filename: {path}')
                writer, label = match.groups()
                collection = 'single_digits' if len(label) == 1 else 'numbers'
            elif kind == 'characters':
                writer, _, label = label_from_path(path)
                collection = kind
            else:
                writer, label = parse_label(path)
                collection = kind
            items.append(dict(file=path.as_posix(), writer=writer, reference=label,
                              collection=collection, sha256=sha(path)))
    counts = Counter(x['collection'] for x in items)
    if counts != dict(single_digits=20, numbers=20, characters=80, strings=40):
        raise ValueError(f'Unexpected development collection counts: {counts}')
    return items


def prepare(image, collection):
    if collection == 'characters':
        arrays = prepare_character(image)[None]
        return arrays, [], []
    seg = (segment_image(image, 'components') if collection in ('single_digits', 'numbers')
           else segment_characters(image, 'grouped_components'))
    arrays = (prepare_crops(seg.crops, 'otsu') if collection in ('single_digits', 'numbers')
              else prepare_character_crops(seg.crops))
    return arrays, seg.boxes, seg.warnings


def panel(image, arrays, item, predictions, path):
    canvas = Image.new('RGB', (1050, 440), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), f"{item['file']} | true={item['reference']} | MLP={predictions['mlp']} | CNN={predictions['cnn']}", fill='black')
    original = image.copy(); original.thumbnail((420, 380))
    canvas.paste(original, (10, 40))
    for i, array in enumerate(arrays):
        tile = Image.fromarray(np.rint(array*255).astype('uint8')).resize((70, 70))
        canvas.paste(tile, (450+(i % 7)*82, 80+(i//7)*100))
    draw.text((450, 45), 'Identical prepared crops supplied to both models', fill='black')
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1:
        parser.error('Use a new output directory and positive repeat count')
    items = discover()
    torch.set_num_threads(4)
    predictors, checkpoints = {}, {}
    for task, folder, factory in [('digits', 'experiments/baseline', DigitPredictor),
                                   ('characters', 'experiments/emnist_byclass/run_01', CharacterPredictor)]:
        for model in ('mlp', 'cnn'):
            path = Path(folder)/f'{model}.pt'
            checkpoints[f'{task}_{model}'] = {'file':path.as_posix(), 'sha256':sha(path)}
            predictor = factory(path)
            fn = predictor.predict_number if task == 'digits' else predictor.predict_text
            fn(np.zeros((1, 28, 28), np.float32))
            predictors[task, model] = fn
    args.output.mkdir(parents=True)
    (args.output/'previews').mkdir()
    (args.output/'prepared').mkdir()
    sources = {}
    for path in [Path(__file__).resolve(), *Path('hnrs_ml').glob('*.py'),
                 *Path('hnrs_chars').glob('*.py'), Path('segment_digits.py'),
                 Path('preprocess_digits.py'), Path('compare_segmentation.py')]:
        relative = Path(path.name) if path.is_absolute() else path
        target = args.output/'source'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target); sources[relative.as_posix()] = sha(path)
    write_csv(args.output/'manifest.csv', items)
    rows, paired = [], []
    for index, item in enumerate(items):
        with Image.open(item['file']) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        start = time.perf_counter()
        arrays, boxes, warnings = prepare(image, item['collection'])
        preparation_ms = (time.perf_counter()-start)*1000
        np.save(args.output/'prepared'/f'{index:03d}.npy', arrays)
        task = 'digits' if item['collection'] in ('single_digits', 'numbers') else 'characters'
        predictions, elapsed = {}, {'mlp':[], 'cnn':[]}
        # Alternate execution order to reduce systematic first-model timing bias.
        for repeat in range(args.repeats):
            order = ('mlp','cnn') if (index+repeat)%2 == 0 else ('cnn','mlp')
            for model in order:
                start = time.perf_counter()
                prediction = predictors[task, model](arrays)['text']
                elapsed[model].append((time.perf_counter()-start)*1000)
                if model in predictions and predictions[model] != prediction:
                    raise ValueError('Non-repeatable inference')
                predictions[model] = prediction
        preview = f'previews/{index:03d}.png'
        panel(image, arrays, item, predictions, args.output/preview)
        pair = dict(item, crop_count=len(arrays), boxes=json.dumps(boxes), warnings='; '.join(warnings),
                    preview=preview, preparation_ms=preparation_ms)
        for model in ('mlp','cnn'):
            prediction = predictions[model]
            ms = statistics.median(elapsed[model])
            row = dict(item, model=model, prediction=prediction, correct=int(prediction==item['reference']),
                       edit_distance=edit_distance(item['reference'], prediction), crop_count=len(arrays),
                       preparation_ms=preparation_ms, inference_ms=ms, pipeline_ms=preparation_ms+ms,
                       inference_samples_ms=json.dumps(elapsed[model]), warnings='; '.join(warnings), preview=preview)
            rows.append(row)
            pair[model] = prediction; pair[f'{model}_correct'] = row['correct']
        pair['review_category'] = ('count_mismatch' if len(arrays)!=len(item['reference']) else
                                  'recognition_error_review_crops' if not(pair['mlp_correct'] and pair['cnn_correct']) else 'both_correct')
        paired.append(pair)
        print(f"{index+1}/{len(items)} {item['collection']} {item['reference']} MLP={predictions['mlp']} CNN={predictions['cnn']}", flush=True)
    write_csv(args.output/'results.csv', rows)
    write_csv(args.output/'paired_results.csv', paired)
    groups = []
    for collection in ('single_digits','numbers','characters','strings'):
        for model in ('mlp','cnn'):
            subset = [r for r in rows if r['collection']==collection and r['model']==model]
            groups.append(dict(collection=collection, model=model, images=len(subset),
                correct=sum(r['correct'] for r in subset), accuracy=sum(r['correct'] for r in subset)/len(subset),
                character_error_rate=sum(r['edit_distance'] for r in subset)/sum(len(r['reference']) for r in subset),
                mean_inference_ms=statistics.mean(r['inference_ms'] for r in subset),
                mean_pipeline_ms=statistics.mean(r['pipeline_ms'] for r in subset)))
    summary = dict(purpose='Paired development comparison; not independent final evaluation', results=groups,
        checkpoints=checkpoints, sources=sources, repeats=args.repeats, torch=torch.__version__, python=platform.python_version(),
        platform=platform.platform(), threads=4, git_base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        timing='Shared preparation once per image; median of warmed repeated inference. Pipeline = preparation + inference, excludes file I/O, model load and export. Single machine, no timing significance claim.',
        settings={'digits':'components + otsu', 'characters':'prepare_character defaults', 'strings':'grouped_components + prepare_character defaults'},
        duplicate_hash_groups=[v for v in Counter(i['sha256'] for i in items).values() if v>1])
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    write_csv(args.output/'summary.csv', groups)


if __name__ == '__main__':
    main()
