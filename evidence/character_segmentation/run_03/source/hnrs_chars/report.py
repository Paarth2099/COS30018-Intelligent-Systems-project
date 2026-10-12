"""Build a readable report and learning curves from saved, measured results."""
import argparse
import csv
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def learning_curves(history,destination):
    canvas=Image.new('RGB',(1200,520),'white');draw=ImageDraw.Draw(canvas)
    try:
        font=ImageFont.load_default(size=18)
        small=ImageFont.load_default(size=14)
    except (ImportError, OSError):
        # Deep Windows environment paths can prevent FreeType DLL loading.
        font=small=ImageFont.load_default()
    draw.text((30,16),'EMNIST ByClass learning curves',font=font,fill='#172b45')
    for panel,name in enumerate(('mlp','cnn')):
        left=65+panel*590;top=100;width=470;height=320
        rows=[r for r in history if r['model']==name];last=max(int(r['epoch']) for r in rows)
        draw.text((left,60),name.upper(),font=font,fill='#172b45')
        for tick in range(0,101,20):
            y=top+height*(1-tick/100)
            draw.line((left,y,left+width,y),fill='#e2e8f0')
            draw.text((left-42,y-8),str(tick)+'%',font=small,fill='#475569')
        for field,color,label in [('train_accuracy','#1d4ed8','Training'),('validation_accuracy','#d97706','Validation')]:
            points=[(left+(int(r['epoch'])-1)*width/max(1,last-1),top+height*(1-float(r[field]))) for r in rows]
            if len(points)>1:draw.line(points,fill=color,width=3)
            for x,y in points:draw.ellipse((x-4,y-4,x+4,y+4),fill=color)
            draw.text((left+(0 if field=='train_accuracy' else 135),465),label,font=small,fill=color)
        for epoch in range(1,last+1):
            x=left+(epoch-1)*width/max(1,last-1)
            draw.text((x-5,top+height+12),str(epoch),font=small,fill='#475569')
        draw.text((left+200,top+height+32),'Epoch',font=small,fill='#475569')
    canvas.save(destination)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,default=Path('experiments/emnist_byclass/run_01'))
    parser.add_argument('--evaluation',type=Path,default=Path('evidence/emnist/evaluation_run_01'))
    parser.add_argument('--output',type=Path,default=Path('EMNIST_RESULTS.md'))
    args=parser.parse_args()
    result=json.loads((args.evaluation/'results.json').read_text())
    selection=json.loads((args.experiment/'selection.json').read_text())
    examples=json.loads((args.evaluation/'interface_example.json').read_text())
    with (args.experiment/'history.csv').open() as stream:history=list(csv.DictReader(stream))
    learning_curves(history,args.evaluation/'learning_curves.png')
    lines=['# EMNIST ByClass baseline experiment results','',
           'Owner: Haresh model component. Implementation, execution and this report used Codex assistance.',
           'These are measured character-classifier results, not a completed alphanumeric GUI or photo benchmark.','',
           '## Method','',
           f"The saved run uses {selection['config']['training_count']:,} training and {selection['config']['validation_count']:,} validation images from the original EMNIST training partition. The official 116,323-image test partition was evaluated only after validation selected a model.",
           'The verified 62 labels preserve digits, uppercase and lowercase separately. Raw EMNIST image axes were transposed once to upright; samples were visually inspected before training.',
           f"Both architectures used {selection['config']['epochs']} epochs, seed {selection['config']['seed']}, Adam at learning rate {selection['config']['learning_rate']}, batch size {selection['config']['batch_size']}, unweighted cross entropy and no augmentation. The split and minibatch order are shared. See HARESH_EXTENSION.md for the architectures and reproduction commands.",'',
           '## Model comparison','',
           '| Model | Parameters | Best epoch | Validation accuracy | Official test correct | Test accuracy | Macro F1 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for record,measured in zip(selection['models'],result['models']):
        assert record['model']==measured['model']
        lines.append(f"| {record['model'].upper()} | {record['parameters']:,} | {record['best_epoch']} | {record['validation_accuracy']:.2%} | {measured['correct']:,}/{measured['count']:,} | {measured['accuracy']:.2%} | {measured['macro_f1']:.4f} |")
    lines+=['',f"**Selected model: {result['selected_model'].upper()}**, chosen from validation performance before testing. Test scores did not select the model.",'',
            'Macro F1 gives every class equal weight; overall accuracy gives every example equal weight. ByClass is imbalanced, so the two measures describe different aspects of performance.','',
            '## Character groups','', '| Model | Digits | Uppercase | Lowercase |','|---|---:|---:|---:|']
    for measured in result['models']:
        lines.append('| '+measured['model'].upper()+' | '+' | '.join(f"{measured['groups'][group]['accuracy']:.2%} (n={measured['groups'][group]['count']:,})" for group in ('digits','uppercase','lowercase'))+' |')
    lines+=['','## Timing','', '| Model | Training plus validation | Warmed test prediction |','|---|---:|---:|']
    for record,measured in zip(selection['models'],result['models']):
        lines.append(f"| {record['model'].upper()} | {record['training_seconds']:.2f} s | {measured['warmed_batch_prediction_seconds']:.2f} s / {measured['count']:,} characters |")
    lines+=['','These are single-run local CPU timings. Prediction includes pixel conversion, batched inference, softmax and Python result construction; it excludes dataset loading, segmentation, photo preprocessing and GUI work. Timing is not a portable performance guarantee.','',
            '## Known boundary string reconstruction','',
            'For each model, 1,000 strings were constructed from 3,500 distinct official-test characters: 250 strings at each length. Both models use identical indices. Boundaries were already known, so these results do not test character segmentation, words or complete photo recognition. They reuse a subset of the character test set and are not independent evidence.','',
            '| Model | 2 characters | 3 characters | 4 characters | 5 characters | Overall |', '|---|---:|---:|---:|---:|---:|']
    for name,groups in result['known_boundary_strings'].items():
        lines.append('| '+name.upper()+' | '+' | '.join(f"{groups[str(length)]['correct']}/250 ({groups[str(length)]['accuracy']:.1%})" for length in (2,3,4,5))+f" | {sum(r['correct'] for r in groups.values())}/1000 |")
    lines+=['','## Retained errors and interface example','']
    for measured in result['models']:
        text=', '.join(f"`{r['true']}` to `{r['predicted']}` ({r['count']})" for r in measured['top_confusions'][:5])
        lines.append(f"{measured['model'].upper()} frequent confusions: {text}.")
    lines+=['','Similar handwriting shapes and class imbalance are plausible contributors; confusion counts alone do not establish a causal explanation. Lowercase/uppercase labels remain distinct even where isolated handwriting looks alike.','',
            f"The fixed interface example has reference text `{examples['true_text']}`. Samples were selected by label and dataset order, not by whether the model predicted them correctly.",'']
    for name,prediction in examples['predictions'].items():lines.append(f"- {name.upper()} predicted `{prediction['text']}`.")
    lines+=['','The public predictor was checked against the batched test predictions for those same crops. It never receives the reference text. Leading zeros and case are preserved in the output format, not guaranteed to be recognised correctly.','',
            '## Verification and limits','',
            '- All 24 software tests passed: 18 existing tests and six new character tests. The original digit source and model files were unchanged.',
            '- Dataset mapping, class counts, split isolation, checkpoint hashes, source snapshots and per-image predictions are retained.',
            '- One training seed and one configuration per model do not establish optimal settings or run-to-run variability.',
            '- Confidence values are uncalibrated model scores, not correctness guarantees.',
            '- Character segmentation, photographed strings, spacing, dotted letters and GUI character-mode integration remain team work.',
            '- Keep later design decisions on validation/development inputs. Reserve new handwriting for final system evaluation.','',
            '## Evidence','',
            '- `evidence/emnist/preparation/`: sample grids, orientation comparison, class counts, dataset hashes and split indices.',
            '- `experiments/emnist_byclass/run_01/`: both checkpoints, configuration, history, source snapshot and validation selection.',
            '- `evidence/emnist/evaluation_run_01/`: results, per-image CSVs, per-class metrics, confusion matrices, first-error grids, string results and interface example.',
            '- `evidence/emnist/unit_tests.txt` and `evidence/emnist/training_log.txt`.',
            '- `HARESH_EXTENSION.md`: source references, commands, prediction contract and team handover.','']
    args.output.write_text('\n'.join(lines),encoding='utf-8')
    print(args.output)


if __name__=='__main__':main()
