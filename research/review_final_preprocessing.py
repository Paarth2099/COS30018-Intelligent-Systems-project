"""Inspect saved final-run arrays and make review panels; no inference or tuning."""
from pathlib import Path
import hashlib,json
from collections import defaultdict
import numpy as np
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'evidence/final_evaluation/run_01'
OUT=ROOT/'evidence/final_evaluation/preprocessing_review'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=True)
    manifest=json.loads((RUN/'locked_manifest.json').read_text())
    predictions=json.loads((RUN/'predictions.json').read_text())
    records=[];groups=defaultdict(list)
    for item in manifest['samples']:
        task='mnist' if item['collection'] in ['single_digits','numbers'] else 'emnist'
        method='components' if task=='mnist' else 'grouped_components'
        model='cnn' if task=='mnist' else 'resnet18'
        folder=RUN/'samples'/item['sample_id']/f'{task}_{method}'
        meta=json.loads((folder/'preparation.json').read_text())
        arrays=np.load(folder/'prepared.npy')
        assert sha(ROOT/item['actual_path'])==item['sha256'] and sha(folder/'prepared.npy')==meta['prepared_sha256']
        pred=next(r for r in predictions if r['sample_id']==item['sample_id'] and r['task']==task and r['method']==method and r['model']==model)
        measures=[]
        for k,a in enumerate(arrays):
            ys,xs=np.nonzero(a>.1)
            measures.append(dict(index=k,minimum=float(a.min()),maximum=float(a.max()),finite=bool(np.isfinite(a).all()),nonzero_pixels=int(np.count_nonzero(a)),pixels_above_point1=int((a>.1).sum()),edge_above_point1=bool((a[0]>.1).any() or (a[-1]>.1).any() or (a[:,0]>.1).any() or (a[:,-1]>.1).any()),ink_bbox_above_point1=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1] if len(xs) else None))
        record=dict(sample_id=item['sample_id'],writer=item['writer'],collection=item['collection'],reference=item['verified_label'],prediction=pred['prediction'],correct=pred['correct'],crop_count=len(arrays),expected_length=len(item['verified_label']),count_matches=len(arrays)==len(item['verified_label']),warnings=meta['warnings'],metrics=measures,original_sha256=item['sha256'],prepared_sha256=meta['prepared_sha256'],prepared_path=(folder/'prepared.npy').relative_to(ROOT).as_posix())
        records.append(record)
        panel=Image.new('RGB',(760,245),'white');draw=ImageDraw.Draw(panel)
        draw.text((6,5),f"{item['sample_id']} | true={item['verified_label']} | {model}={pred['prediction']}",fill='black')
        draw.text((6,23),'Original | each saved crop ABOVE its prepared 28x28 input',fill='black')
        with Image.open(ROOT/item['actual_path']) as im:
            im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((205,180));panel.paste(im,(5,50))
        for k,a in enumerate(arrays):
            with Image.open(folder/f'crop_{k:02d}.png') as im:
                im=im.convert('RGB');im.thumbnail((100,85));panel.paste(im,(220+k*115,48))
            prepared=Image.fromarray(np.rint(a*255).astype('uint8')).resize((100,100),Image.Resampling.NEAREST)
            panel.paste(prepared,(220+k*115,138))
        panel.save(OUT/f"{item['sample_id']}.png")
        groups[item['writer'],item['collection']].append(panel)
    for (writer,collection),panels in groups.items():
        sheet=Image.new('RGB',(1520,245*((len(panels)+1)//2)), '#dddddd')
        for i,p in enumerate(panels):sheet.paste(p,((i%2)*760,(i//2)*245))
        sheet.save(OUT/f'sheet_{writer}_{collection}.jpg',quality=95)
    summary=dict(samples=len(records),crops=sum(r['crop_count'] for r in records),selected_errors=sum(not r['correct'] for r in records),empty_arrays=sum(m['nonzero_pixels']==0 for r in records for m in r['metrics']),nonfinite_arrays=sum(not m['finite'] for r in records for m in r['metrics']),prepared_edge_contacts=sum(m['edge_above_point1'] for r in records for m in r['metrics']),crop_count_mismatches=[r['sample_id'] for r in records if not r['count_matches']],recognition_run=False,preprocessing_rerun=False,threshold_point1='Diagnostic measurement only, not a changed preprocessing threshold',note='Array checks cannot establish semantic correctness or causal attribution; visual review required.')
    (OUT/'array_audit.json').write_text(json.dumps(dict(summary=summary,samples=records),indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
