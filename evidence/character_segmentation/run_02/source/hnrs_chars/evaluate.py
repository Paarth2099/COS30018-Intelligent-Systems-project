"""Evaluate frozen checkpoints on the official test set, without selecting on it."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from .data import CLASSES, load_partition, batch_images, sha256
from .predict import CharacterPredictor


def metrics(truth,predicted):
    truth=np.asarray(truth);predicted=np.asarray(predicted)
    if truth.shape!=predicted.shape or truth.ndim!=1 or not len(truth):
        raise ValueError('Metrics require equal nonempty one-dimensional label arrays')
    if np.any(truth<0) or np.any(truth>=62) or np.any(predicted<0) or np.any(predicted>=62):
        raise ValueError('Class IDs must be between 0 and 61')
    confusion=np.zeros((62,62),dtype=np.int64);np.add.at(confusion,(truth,predicted),1)
    per_class=[]
    for i,char in enumerate(CLASSES):
        tp=int(confusion[i,i]);support=int(confusion[i].sum());guesses=int(confusion[:,i].sum())
        precision=tp/guesses if guesses else 0.;recall=tp/support if support else 0.
        per_class.append({'class_id':i,'character':char,'support':support,'precision':precision,'recall':recall,
                          'f1':2*precision*recall/(precision+recall) if precision+recall else 0.})
    groups={}
    for name,start,end in [('digits',0,10),('uppercase',10,36),('lowercase',36,62)]:
        mask=(truth>=start)&(truth<end)
        groups[name]={'count':int(mask.sum()),'accuracy':float((truth[mask]==predicted[mask]).mean()) if mask.any() else None}
    confused=[{'true':CLASSES[i],'predicted':CLASSES[j],'count':int(confusion[i,j])}
              for i in range(62) for j in range(62) if i!=j and confusion[i,j]]
    confused.sort(key=lambda row:(-row['count'],row['true'],row['predicted']))
    return {'count':len(truth),'correct':int(np.trace(confusion)),'accuracy':float((truth==predicted).mean()),
            'macro_f1':float(np.mean([r['f1'] for r in per_class])),'groups':groups,'per_class':per_class,
            'top_confusions':confused[:20],'confusion_matrix':confusion.tolist()}


def csv_rows(path,rows):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def error_grid(images,truth,predicted,path):
    indices=np.flatnonzero(truth!=predicted)[:36]
    canvas=Image.new('RGB',(6*170,50+6*132),'#f3f6fb');draw=ImageDraw.Draw(canvas)
    draw.text((12,12),'First 36 official-test errors in dataset order | true -> predicted',fill='black')
    for k,idx in enumerate(indices):
        x,y=(k%6)*170,(k//6)*132+44
        canvas.paste(Image.fromarray(images[idx].numpy()).resize((84,84)),(x+8,y))
        draw.text((x+8,y+89),f'{int(idx)}: {CLASSES[truth[idx]]} -> {CLASSES[predicted[idx]]}',fill='black')
    canvas.save(path)


def confusion_chart(matrix,path):
    # Row-normalised shading makes rare and common classes visible together.
    matrix=np.asarray(matrix);counts=matrix.sum(1,keepdims=True)
    values=np.divide(matrix,counts,out=np.zeros_like(matrix,dtype=float),where=counts!=0)
    size=18;margin=45
    canvas=Image.new('RGB',(margin+62*size+8,margin+62*size+30),'white');draw=ImageDraw.Draw(canvas)
    draw.text((margin,3),'Predicted class across columns; true class down rows; darker = higher fraction',fill='black')
    for i,char in enumerate(CLASSES):
        draw.text((margin+i*size+3,25),char,fill='black');draw.text((23,margin+i*size+3),char,fill='black')
        for j in range(62):
            value=values[i,j];color=tuple(int(255*(1-value)+channel*value) for channel in (24,77,160))
            draw.rectangle((margin+j*size,margin+i*size,margin+(j+1)*size-1,margin+(i+1)*size-1),fill=color)
    canvas.save(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('data'))
    parser.add_argument('--experiment',type=Path,default=Path('experiments/emnist_byclass/run_01'))
    parser.add_argument('--output',type=Path,default=Path('evidence/emnist/evaluation_run_01'))
    parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    if min(args.batch_size,args.threads)<=0:parser.error('Batch size and threads must be positive')
    if args.output.exists():parser.error('Choose a new output folder to preserve prior evaluation')
    selection=json.loads((args.experiment/'selection.json').read_text())
    if not selection['selected_before_test_evaluation']:raise ValueError('Validation selection must precede testing')
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    images,targets=load_partition(args.data_dir,train=False);truth=targets.numpy()
    args.output.mkdir(parents=True);all_results=[];predictions={};predictors={}
    source_dir=args.output/'source';source_dir.mkdir()
    source_hashes={}
    for path in sorted(Path(__file__).parent.glob('*.py')):
        (source_dir/path.name).write_bytes(path.read_bytes());source_hashes[path.name]=sha256(path)
    for record in selection['models']:
        name=record['model'];checkpoint=args.experiment/f'{name}.pt'
        if sha256(checkpoint)!=record['checkpoint_sha256']:raise ValueError('Checkpoint changed after selection')
        predictor=CharacterPredictor(checkpoint);predictors[name]=predictor
        predictor.predict_characters(batch_images(images,torch.arange(args.batch_size)))
        results=[];started=time.perf_counter()
        for ids in torch.arange(len(targets)).split(args.batch_size):
            results.extend(predictor.predict_characters(batch_images(images,ids),args.batch_size))
        elapsed=time.perf_counter()-started
        guessed=np.array([r['class_id'] for r in results]);predictions[name]=guessed
        report=metrics(truth,guessed)
        report.update({'model':name,'checkpoint_sha256':record['checkpoint_sha256'],
                       'best_epoch':record['best_epoch'],'validation_accuracy':record['validation_accuracy'],
                       'warmed_batch_prediction_seconds':elapsed,'seconds_per_character':elapsed/len(targets),
                       'timing_scope':'CPU batches including uint8 conversion, softmax and Python result construction; excludes loading, segmentation and GUI'})
        all_results.append(report)
        csv_rows(args.output/f'{name}_per_class.csv',report['per_class'])
        with (args.output/f'{name}_confusion.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['true/predicted']+CLASSES)
            for char,row in zip(CLASSES,report['confusion_matrix']):writer.writerow([char]+row)
        csv_rows(args.output/f'{name}_predictions.csv',[
            {'test_index':i,'true_id':int(truth[i]),'true':CLASSES[truth[i]],'predicted_id':int(guessed[i]),
             'predicted':CLASSES[guessed[i]],'score':results[i]['confidence'],'correct':int(truth[i]==guessed[i])}
            for i in range(len(truth))])
        error_grid(images,truth,guessed,args.output/f'{name}_errors.png')
        confusion_chart(report['confusion_matrix'],args.output/f'{name}_confusion.png')
        print(json.dumps({k:report[k] for k in ['model','accuracy','macro_f1','warmed_batch_prediction_seconds']}),flush=True)
    # Benchmark string reconstruction uses known boundaries, not a segmentation
    # pass. Share exactly the same source indices between the two models.
    rng=np.random.default_rng(2026);order=rng.permutation(len(truth));cursor=0;strings=[]
    for length in (2,3,4,5):
        for example in range(250):
            ids=order[cursor:cursor+length];cursor+=length
            actual=''.join(CLASSES[i] for i in truth[ids])
            for name in predictions:
                text=''.join(CLASSES[i] for i in predictions[name][ids])
                strings.append({'model':name,'length':length,'example':example,'source_test_indices':' '.join(map(str,ids)),
                                'true':actual,'predicted':text,'exact_correct':int(actual==text)})
    csv_rows(args.output/'known_boundary_strings.csv',strings)
    string_results={name:{str(length):{'count':250,'correct':sum(r['exact_correct'] for r in strings if r['model']==name and r['length']==length),
                                     'accuracy':sum(r['exact_correct'] for r in strings if r['model']==name and r['length']==length)/250}
                         for length in (2,3,4,5)} for name in predictions}
    # Provide a fixed public-interface example without picking by prediction.
    request='007Ab12';ids=[];used=set()
    for char in request:
        idx=next(int(i) for i in np.flatnonzero(truth==CLASSES.index(char)) if int(i) not in used)
        ids.append(idx);used.add(idx)
    demo_dir=args.output/'example_crops';demo_dir.mkdir()
    arrays=batch_images(images,torch.tensor(ids)).squeeze(1).numpy()
    np.save(demo_dir/'prepared.npy',arrays)
    for i,array in enumerate(arrays):Image.fromarray(np.rint(array*255).astype('uint8')).save(demo_dir/f'character_{i}.png')
    demos={}
    for name,predictor in predictors.items():
        demo=predictor.predict_text(arrays)
        assert demo['text']==''.join(CLASSES[i] for i in predictions[name][ids])
        demos[name]=demo
    (args.output/'interface_example.json').write_text(json.dumps({'true_text':request,'test_indices':ids,'predictions':demos},indent=2)+'\n')
    final={'created_utc':datetime.now(timezone.utc).isoformat(),'selected_model':selection['selected_model'],
           'selection_sha256':sha256(args.experiment/'selection.json'),'models':all_results,
           'known_boundary_strings':string_results,'string_evaluation_note':'1000 constructed strings per model, known boundaries, shared subset of official character test set; not independent and not end-to-end photo evaluation',
           'test_files_sha256':{p.name:sha256(p) for p in (args.data_dir/'EMNIST/raw').glob('*byclass-test-*')},
           'source_sha256':source_hashes,
           'limits':[f"Single seed and fixed {selection['config']['epochs']}-epoch run",'Class imbalance and ambiguous case forms',
                     'No photo, character segmentation or GUI evaluation in this stage','Softmax scores are uncalibrated']}
    (args.output/'results.json').write_text(json.dumps(final,indent=2)+'\n')
    print('Evaluation complete; validation-selected model remains',selection['selected_model'],flush=True)


if __name__=='__main__':main()
