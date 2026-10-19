"""Download, inspect and split EMNIST ByClass without inspecting test images."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from .data import CLASSES, URL, ARCHIVE_MD5, ORIENTATION, download_byclass, load_partition, sha256, stratified_split, indices_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('evidence/emnist/preparation'))
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.output.exists():
        parser.error('Choose a new output folder to preserve earlier evidence')
    if args.download:
        download_byclass(args.data_dir)
    images, targets = load_partition(args.data_dir)
    train_ids, val_ids = stratified_split(targets, args.seed)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'split_indices.npz', train=train_ids.numpy(), validation=val_ids.numpy())
    rows = []
    grid = Image.new('RGB', (8*150, 8*158), '#f0f4f8')
    draw = ImageDraw.Draw(grid)
    orient = Image.new('RGB', (8*150, 180), 'white'); od = ImageDraw.Draw(orient)
    for label, char in enumerate(CLASSES):
        ids = torch.where(targets == label)[0]
        x, y = (label % 8)*150, (label//8)*158
        draw.text((x+8, y+6), f'{label}: {char}', fill='black')
        for j in range(2):
            tile = Image.fromarray(images[ids[j]].numpy()).resize((64,64), Image.Resampling.NEAREST)
            grid.paste(tile, (x+6+j*70, y+30))
        count_train = int((targets[train_ids] == label).sum())
        rows.append({'class_id':label,'character':char,'codepoint':ord(char),'training':count_train,
                     'validation':len(ids)-count_train,'original_training':len(ids)})
    grid.save(args.output/'all_classes_upright.png')
    for j, char in enumerate('2FLRabgj'):
        idx = int(torch.where(targets == CLASSES.index(char))[0][0])
        tile = images[idx].numpy()
        od.text((j*150+8,4), f'{char}: raw / upright',fill='black')
        orient.paste(Image.fromarray(tile.T).resize((64,64)),(j*150+6,30))
        orient.paste(Image.fromarray(tile).resize((64,64)),(j*150+76,30))
        od.text((j*150+8,104), f'train index {idx}', fill='black')
    orient.save(args.output/'orientation_check.png')
    with (args.output/'class_counts.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    raw=args.data_dir/'EMNIST/raw'
    record={'dataset':'EMNIST ByClass','source':URL,'official_archive_md5':ARCHIVE_MD5,
            'classes':CLASSES,'orientation':ORIENTATION,'seed':args.seed,'training_count':len(train_ids),
            'validation_count':len(val_ids),'official_training_count':len(targets),'official_test_count_expected':116323,
            'test_images_inspected':False,'train_indices_sha256':indices_hash(train_ids),
            'validation_indices_sha256':indices_hash(val_ids),
            'files':{p.name:sha256(p) for p in raw.glob('emnist-byclass-*') if not p.name.endswith('.gz')},
            'note':'Inspect the asymmetric raw/upright examples before training. Class counts are imbalanced.'}
    (args.output/'dataset_report.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ['dataset','training_count','validation_count','test_images_inspected']}),flush=True)


if __name__=='__main__':main()
