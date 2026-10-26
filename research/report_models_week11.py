"""Build the Week 11 comparison report from verified saved results, without inference."""
import argparse
import csv
import json
from pathlib import Path
from hnrs_chars.evaluate import csv_rows


def read_csv(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(str(x) for x in row)+' |' for row in rows]])


def percent(x):return f'{float(x)*100:.2f}%'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',type=Path,default=Path('evidence/model_comparison/week11_run01'))
    p.add_argument('--report',type=Path,default=Path('docs/reports/model-comparison-week11.md'))
    args=p.parse_args();root=args.evaluation
    verification=json.loads((root/'verification.json').read_text())
    if verification['status']!='PASS':raise ValueError('Evaluation incomplete')
    plan=json.loads((root/'plan.json').read_text());recommend=json.loads((root/'recommendation.json').read_text())
    models=read_csv(root/'models.csv');bench=read_csv(root/'benchmark_summary.csv')
    photos=read_csv(root/'photo_summary.csv');pairs=read_csv(root/'paired_photos.csv')
    photo_lookup={(r['collection'],r['model']):r for r in photos}
    def result(collection,model):
        row=photo_lookup[collection,model]
        return f"{row['correct']}/{row['count']}"
    by_writer=[];paired=[]
    for collection in ('single_digits','numbers','characters','strings'):
        subset=[r for r in pairs if r['collection']==collection]
        for name in ('mlp','resnet18','mobilenet_v3_small'):
            paired.append(dict(collection=collection,model=name,
                gained=sum(r[name]==r['reference'] and r['cnn']!=r['reference'] for r in subset),
                lost=sum(r[name]!=r['reference'] and r['cnn']==r['reference'] for r in subset),
                both_correct=sum(r[name]==r['reference'] and r['cnn']==r['reference'] for r in subset),
                both_wrong=sum(r[name]!=r['reference'] and r['cnn']!=r['reference'] for r in subset)))
        for writer in sorted({r['writer'] for r in subset}):
            group=[r for r in subset if r['writer']==writer]
            for name in ('mlp','cnn','resnet18','mobilenet_v3_small'):
                by_writer.append(dict(collection=collection,writer=writer,model=name,count=len(group),correct=sum(r[name]==r['reference'] for r in group)))
    csv_rows(root/'paired_changes.csv',paired);csv_rows(root/'by_writer.csv',by_writer)
    link='../../'+root.as_posix()
    lines=['# Week 11: four-architecture comparison','',
      '**Scope:** Haresh\'s model investigation following tutor feedback. Four architectures are now implemented and evaluated for both digit and alphanumeric recognition. This report does not claim completion of the whole project, fresh final evaluation or GUI integration of the new models.','',
      '## Outcome','',
      *[f"- **{task.upper()}:** validation candidate `{r['validation_candidate']}`; development-gated recommendation **`{r['recommended_model']}`**." for task,r in recommend.items()],
      '', 'The recommendation follows the recorded validation ranking and development no-regression gates. Official-test results were excluded from these decisions. The GUI continues to use its existing MLP/CNN checkpoints until an explicit integration and acceptance step.','',
      f"ResNet-18 improves single-digit photo recognition from {result('single_digits','cnn')} to {result('single_digits','resnet18')}, but exact multi-digit recognition falls from {result('numbers','cnn')} to {result('numbers','resnet18')}. The number system therefore retains the baseline CNN under the recorded no-regression rule. The better MNIST benchmark alone does not justify replacement.",'',
      f"For alphanumerics, ResNet-18 improves isolated characters from {result('characters','cnn')} to {result('characters','resnet18')} and complete strings from {result('strings','cnn')} to {result('strings','resnet18')}. String CER changes from {percent(photo_lookup['strings','cnn']['cer'])} to {percent(photo_lookup['strings','resnet18']['cer'])}. This supports integration as the alphanumeric candidate, but most complete strings are still wrong; it is not evidence of reliable unrestricted text recognition.",'',
      '## Motivation and models','',
      'The tutor requested two more architectures in addition to MLP and the baseline CNN. ResNet-18 adds residual blocks, and MobileNetV3-Small uses a compact convolutional design. Both are CNN architectures; the original custom network is therefore named baseline CNN throughout this comparison. The Week 10 class-weighted CNN is a training variant, not a fifth architecture.','',
      'The new models use Torchvision 0.21 with no pretrained weights. Only the first convolution is adapted to one channel and the classifier to 10 or 62 classes. Default widths, strides and pooling remain intact. Inputs stay 28x28, scaled to [0,1]. This is an adapted handwriting experiment, not a reproduction of published ImageNet performance.','',
      '## Experimental controls and provenance','',
      '- MNIST: 54,000 training and 6,000 validation examples. EMNIST ByClass: 628,142 training and 69,790 validation examples. Split hashes match the existing baselines.',
      '- Five epochs, seed 42, batch 256, Adam learning rate 0.001, no weight decay, no augmentation, unweighted cross entropy. All original baseline checkpoints are preserved.',
      '- Checkpoints selected by highest validation accuracy, then lower validation loss; earlier epoch retained on exact ties. EMNIST MobileNet can therefore retain epoch 4 even though five epochs completed.',
      '- New runs trained on the RTX 3070 using float32 and deterministic settings, without AMP/TF32. Historical baselines trained on CPU; device and OS differ. Do not interpret historical training times as a controlled speed comparison.',
      '- The imported ZIP, all four checkpoint hashes, source snapshots, five-row histories and best-epoch choices were checked. The PC run had no Git commit identifier; source hashes match the local files.',
      '- Current Week 10 preprocessing and segmentation prepare each development photo once. All four models receive the same ordered arrays. These are previously used development images, not new held-out photos.',
      f"- Evaluation uses {plan['evaluation_device']} with {plan['threads']} threads on one computer. Photo latency is the mean of per-photo medians over five warmed calls, with rotating model order and preparation excluded.",
      '', '## Validation and model size','',
      table(['Dataset','Architecture','Best epoch','Validation accuracy','Parameters','Checkpoint MiB'],
       [[r['task'],r['model'],r['best_epoch'],percent(r['validation_accuracy']),f"{int(r['parameters']):,}",f"{int(r['checkpoint_bytes'])/1048576:.2f}"] for r in models]),
      '', '## Official test benchmarks','',
      'These results evaluate frozen checkpoints. They were not used for tuning or promotion. Macro F1 gives equal weight to each class; overall accuracy weights each example equally.','',
      table(['Dataset','Architecture','Correct / total','Accuracy','Macro F1'],
       [[r['task'],r['model'],f"{r['correct']} / {r['count']}",percent(r['accuracy']),f"{float(r['macro_f1']):.4f}"] for r in bench]),
      '', '## Existing handwriting development photos','',
      'Exact accuracy requires the entire predicted string to match, including case and leading zeros. Character error rate (CER) is total edit distance divided by total reference characters. A correct crop count does not imply correct recognition.','',
      table(['Collection','Architecture','Exact correct / total','Exact accuracy','CER','Photo inference ms'],
       [[r['collection'],r['model'],f"{r['correct']} / {r['count']}",percent(r['accuracy']),percent(r['cer']),f"{float(r['mean_photo_inference_ms']):.3f}"] for r in photos]),
      '', '## Paired changes against the baseline CNN','',
      table(['Collection','Architecture','Newly correct','Newly wrong','Both correct','Both wrong'],
       [[r['collection'],r['model'],r['gained'],r['lost'],r['both_correct'],r['both_wrong']] for r in paired]),
      '', '## Representative changes','',
      'Examples below are the first two gains and first two losses in dataset order for ResNet-18 in each task, rather than hand-picked favourable cases. All images remain in aggregate results.','']
    examples=[]
    for task,collections in [('mnist',('single_digits','numbers')),('emnist',('characters','strings'))]:
        subset=[r for r in pairs if r['collection'] in collections]
        for outcome in ('gain','loss'):
            chosen=[r for r in subset if (r['resnet18']==r['reference'] and r['cnn']!=r['reference'] if outcome=='gain' else r['resnet18']!=r['reference'] and r['cnn']==r['reference'])][:2]
            for r in chosen:examples.append([task,outcome,r['reference'],r['cnn'],r['resnet18'],f"[Image {r['index']}]({link}/previews/{int(r['index']):03d}.png)"])
    lines.extend([table(['Task','Change','Reference','Baseline CNN','ResNet-18','Evidence'],examples),'',
      '## Selection gates','',
      'For each task the candidate is the highest-validation model. Promotion additionally requires no reduction in exact accuracy and no increase in CER versus the baseline CNN on either development collection for that task. If any gate fails, retain the baseline while documenting the candidate and its limitations. This deliberately avoids choosing a different model after inspecting official-test scores.','',
      table(['Task','Gate','Passed'],[[task,gate,str(value)] for task,r in recommend.items() for gate,value in r['development_gates'].items()]),
      '', '## Limitations and remaining work','',
      '- One seed and five epochs per architecture. These results establish this fixed-budget comparison, not the maximum attainable accuracy of each architecture.',
      '- GPU versus historical CPU training introduces a hardware/numerical difference. Deterministic settings do not guarantee identical results across hardware or PyTorch builds.',
      '- Model size and default downsampling differ greatly. The default large-image strides may lose information on small handwriting images; this is a plausible limitation, not a separately tested explanation.',
      '- Four team writers and reused development photos limit generalisation. Case ambiguity, similar symbols and segmentation errors remain possible. Do not describe these photos as an independent final test.',
      '- Benchmark recognition operates on individual characters; photographed numbers/strings exercise the complete preparation and segmentation pipeline. Report these results separately.',
      '- Complete selected-model GUI integration/acceptance, freeze configuration, then evaluate fresh handwriting without repeated tuning on that final set. Complete installation checks, team report and video separately.',
      '', '## Evidence and reproduction','',
      f'- [Import verification](../../evidence/week11_gpu_setup/import_verification.json)',
      f'- [Plan and validation candidates]({link}/plan.json)',f'- [Model metadata]({link}/models.csv)',
      f'- [Benchmark summary]({link}/benchmark_summary.csv)',f'- [Photo summary]({link}/photo_summary.csv)',
      f'- [Every photo prediction]({link}/photo_predictions.csv)',f'- [Paired predictions]({link}/paired_photos.csv)',
      f'- [Per-writer results]({link}/by_writer.csv)',f'- [Recommendation]({link}/recommendation.json)',f'- [Verification]({link}/verification.json)',
      f'- [Independent recalculation and historical reproduction]({link}/independent_verification.json)',
      '- Training records: `experiments/week11_gpu_run01/{mnist,emnist}/{resnet18,mobilenet_v3_small}/`.',
      '- Each benchmark model directory contains predictions, per-class metrics, confusion matrix and first errors. Prepared photo arrays and 160 previews are retained with source snapshots.',
      '', '```powershell','python evaluate_models_week11.py --output evidence/model_comparison/week11_repeat01',
      'python report_models_week11.py --evaluation evidence/model_comparison/week11_repeat01 --report docs/reports/model-comparison-week11-repeat.md','```','',
      '## References','',
      '- [Torchvision 0.21 ResNet-18](https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.resnet18.html)',
      '- [Torchvision 0.21 MobileNetV3-Small](https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.mobilenet_v3_small.html)',''])
    args.report.write_text('\n'.join(lines),encoding='utf-8')
    print(args.report)


if __name__=='__main__':main()
