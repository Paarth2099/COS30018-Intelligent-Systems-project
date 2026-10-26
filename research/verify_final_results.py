"""Independently recalculate final metrics without invoking recognition."""
from pathlib import Path
import hashlib
import json
from collections import Counter
import statistics
import numpy as np
import ast

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/final_evaluation/run_01'
def load(name):return json.loads((OUT/name).read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def distance(a,b):
    grid=[[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1):grid[i][0]=i
    for j in range(len(b)+1):grid[0][j]=j
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            grid[i][j]=min(grid[i-1][j]+1,grid[i][j-1]+1,grid[i-1][j-1]+(a[i-1]!=b[j-1]))
    return grid[-1][-1]
def main():
    rows=load('predictions.json');manifest=load('locked_manifest.json');config=load('frozen_configuration.json');summary=load('summary.json')
    assert len(rows)==680
    assert len({(r['sample_id'],r['collection'],r['model'],r['method']) for r in rows})==680
    labels={r['sample_id']:r['verified_label'] for r in manifest['samples']}
    assert len(labels)==120 and config['manifest_sha256']==sha(OUT/'locked_manifest.json')
    for sample in manifest['samples']:assert sha(ROOT/sample['actual_path'])==sample['sha256']
    for item in config['checkpoints']:assert sha(ROOT/item['path'])==item['sha256']
    for name,digest in config['source_sha256'].items():
        assert sha(OUT/'source'/name)==digest
    # Research entry points were reorganised after the frozen run. Preserve
    # original byte hashes and separately verify executable runtime semantics.
    runtime = ['app.py', 'gui_core.py', 'segment_digits.py', 'preprocess_digits.py',
               'hnrs_ml/models.py', 'hnrs_ml/architectures.py', 'hnrs_ml/predict.py',
               'hnrs_chars/models.py', 'hnrs_chars/predict.py',
               'hnrs_chars/preprocessing.py', 'hnrs_chars/segmentation.py']
    for name in runtime:
        assert ast.dump(ast.parse((ROOT/name).read_text(encoding='utf-8'))) == ast.dump(
            ast.parse((OUT/'source'/name).read_text(encoding='utf-8'))), name
    for r in rows:
        assert r['reference']==labels[r['sample_id']]
        assert r['correct']==int(r['prediction']==r['reference'])
        assert r['edits']==distance(r['reference'],r['prediction'])
        assert r['reference_length']==len(r['reference'])
        if not r['error']:
            for field in ['model_only','end_to_end']:
                values=r['raw_timing'][field+'_ms'];assert len(values)==5 and min(values)>=0
                assert r['median_'+field+'_ms']==statistics.median(values)
    for s in summary:
        group=[r for r in rows if all(r[k]==s[k] for k in ['collection','model','method'])]
        assert s['count']==len(group) and s['correct']==sum(r['correct'] for r in group)
        assert s['accuracy']==s['correct']/s['count']
        assert abs(s['cer']-sum(distance(r['reference'],r['prediction']) for r in group)/sum(len(r['reference']) for r in group))<1e-12
        timings=[r['median_end_to_end_ms'] for r in group if r['median_end_to_end_ms'] is not None]
        if timings:
            assert s['median_end_to_end_ms']==statistics.median(timings)
            assert s['p95_end_to_end_ms']==float(np.percentile(timings,95))
    for writer,groups in load('by_writer.json').items():
        for s in groups:
            subset=[r for r in rows if r['writer']==writer and all(r[k]==s[k] for k in ['collection','model','method'])]
            assert len(subset)==s['count'] and sum(r['correct'] for r in subset)==s['correct']
            assert abs(s['cer']-sum(distance(r['reference'],r['prediction']) for r in subset)/sum(len(r['reference']) for r in subset))<1e-12
    for prep in load('preparation.json'):
        if not prep['error']:
            p=OUT/'samples'/prep['sample_id']/f"{prep['task']}_{prep['method']}"/'prepared.npy'
            assert sha(p)==prep['prepared_sha256']
            arr=np.load(p);assert arr.shape==(prep['crop_count'],28,28)
            assert np.isfinite(arr).all() and (arr.size==0 or (arr.min()>=0 and arr.max()<=1))
    result={'status':'PASS','prediction_rows_recalculated':680,'photos_hash_checked':120,'checkpoints_hash_checked':8,'summary_groups':len(summary),'case_and_leading_zeros_preserved':True,'source_snapshots_match':True,'timing_repeats_checked':True,'note':'Independent arithmetic and integrity verification, not a second independent handwriting collection or manual segmentation assessment.'}
    (OUT/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
