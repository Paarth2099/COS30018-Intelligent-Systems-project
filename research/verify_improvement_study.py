"""Independently check development study coverage, saved arrays and arithmetic."""
import hashlib,json
from pathlib import Path
import numpy as np
from research.compare_segmentation import edit_distance


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    base=Path('evidence/improvements');baseline=[]
    for run in ('development_run01','development_run02'):
        root=base/run;protocol=json.loads((root/'protocol.json').read_text())
        rows=json.loads((root/'predictions.json').read_text());summary=json.loads((root/'summary.json').read_text())
        assert len(rows)==320 and len(protocol['manifest'])==160 and not protocol['final_photos_used']
        for name,digest in json.loads((root/'source_hashes.json').read_text()).items():assert sha(root/'source'/name)==digest
        for model in protocol['checkpoints'].values():assert sha(model['path'])==model['sha256']
        for i,item in enumerate(protocol['manifest']):
            assert sha(item['file'])==item['sha256'] and 'final_evaluation' not in item['file']
            pair=rows[2*i:2*i+2]
            for row in pair:
                assert row['file']==item['file'] and row['reference']==item['reference']
                assert row['correct']==int(row['reference']==row['prediction'])
                assert row['edits']==edit_distance(row['reference'],row['prediction'])
                array=np.load(root/'prepared'/f"{i:03d}_{row['variant']}.npy")
                assert array.shape==(row['crop_count'],28,28) and np.isfinite(array).all()
                assert array.size==0 or (array.min()>=0 and array.max()<=1)
        for group in summary:
            subset=[r for r in rows if (r['collection'],r['variant'])==(group['collection'],group['variant'])]
            assert group['count']==len(subset) and group['correct']==sum(r['correct'] for r in subset)
            assert abs(group['cer']-sum(r['edits'] for r in subset)/sum(len(r['reference']) for r in subset))<1e-12
        baseline.append([r for r in rows if r['variant']=='baseline'])
    assert baseline[0]==baseline[1]
    for i in range(160):assert sha(base/'development_run01'/'prepared'/f'{i:03d}_baseline.npy')==sha(base/'development_run02'/'prepared'/f'{i:03d}_baseline.npy')
    result=dict(status='PASS',runs=2,development_images=160,predictions_recalculated=640,
        saved_array_contracts_checked=640,baseline_arrays_identical=True,development_sources_unchanged=True,
        decision='No candidate passed the prespecified strict-improvement rule; production stays unchanged')
    (base/'study_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
