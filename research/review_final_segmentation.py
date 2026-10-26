"""Audit saved segmentation evidence; never reruns segmentation or recognition."""
from pathlib import Path
import hashlib,json
from collections import defaultdict
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'evidence/final_evaluation/run_01'
OUT=ROOT/'evidence/final_evaluation/segmentation_review'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=True)
    manifest=json.loads((RUN/'locked_manifest.json').read_text())
    predictions=json.loads((RUN/'predictions.json').read_text())
    records=[];sheets=defaultdict(list)
    for item in manifest['samples']:
        if item['collection'] not in ['numbers','strings']:continue
        task='mnist' if item['collection']=='numbers' else 'emnist'
        primary='components' if task=='mnist' else 'grouped_components'
        model='cnn' if task=='mnist' else 'resnet18'
        assert sha(ROOT/item['actual_path'])==item['sha256']
        panels=[]
        for method in [primary,'projection']:
            folder=RUN/'samples'/item['sample_id']/f'{task}_{method}'
            meta=json.loads((folder/'preparation.json').read_text())
            pred=next(r for r in predictions if r['sample_id']==item['sample_id'] and r['method']==method and r['model']==model and r['task']==task)
            boxes=meta['boxes']; crops=sorted(folder.glob('crop_*.png'))
            assert len(crops)==meta['crop_count']==len(boxes)
            assert sha(folder/'prepared.npy')==meta['prepared_sha256']
            ordered=all(boxes[i][0]<=boxes[i+1][0] for i in range(len(boxes)-1))
            rec=dict(sample_id=item['sample_id'],writer=item['writer'],collection=item['collection'],reference=item['verified_label'],method=method,model=model,prediction=pred['prediction'],correct=pred['correct'],edits=pred['edits'],reference_length=pred['reference_length'],crop_count=len(crops),expected_count=len(item['verified_label']),count_matches=len(crops)==len(item['verified_label']),boxes_left_to_right=ordered,boxes=boxes,warnings=meta['warnings'],saved_artifacts={p.relative_to(ROOT).as_posix():sha(p) for p in [folder/'preparation.json',folder/'segmentation.png',folder/'prepared.npy',*crops]})
            records.append(rec)
            panel=Image.new('RGB',(900,275),'white');draw=ImageDraw.Draw(panel)
            draw.text((5,5),f"{item['sample_id']} | {method} | true={item['verified_label']} | output={pred['prediction']}",fill='black')
            draw.text((5,23),'Saved annotation left; crops right in prediction order (before preprocessing)',fill='black')
            with Image.open(folder/'segmentation.png') as im:
                im=im.convert('RGB');im.thumbnail((310,220));panel.paste(im,(5,45))
            for k,p in enumerate(crops):
                with Image.open(p) as im:
                    im=im.convert('RGB');im.thumbnail((95,180));panel.paste(im,(325+k*105,65))
                draw.text((325+k*105,45),str(k+1),fill='black')
            panel.save(OUT/f"{item['sample_id']}_{method}.png")
            panels.append(panel)
        joined=Image.new('RGB',(1800,275),'white');joined.paste(panels[0],(0,0));joined.paste(panels[1],(900,0));sheets[item['writer'],item['collection']].append(joined)
    for (writer,collection),panels in sheets.items():
        sheet=Image.new('RGB',(1800,275*len(panels)),'white')
        for i,p in enumerate(panels):sheet.paste(p,(0,i*275))
        sheet.save(OUT/f'sheet_{writer}_{collection}.jpg',quality=96)
    summary=dict(images=40,routes=len(records),count_mismatches=[{'sample_id':r['sample_id'],'method':r['method'],'count':r['crop_count'],'expected':r['expected_count']} for r in records if not r['count_matches']],order_violations=[r['sample_id'] for r in records if not r['boxes_left_to_right']],note='Counts and left-edge sorting do not establish complete glyphs, correct dot grouping or semantic order. Visual review follows.')
    (OUT/'audit.json').write_text(json.dumps(dict(summary=summary,samples=records),indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
