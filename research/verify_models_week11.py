"""Independently recalculate counts and compare baseline predictions/preparation to saved evidence."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from research.compare_segmentation import edit_distance
from hnrs_chars.data import sha256
from hnrs_chars.evaluate import metrics
from hnrs_ml.evaluate import classification_metrics


def rows(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',type=Path,default=Path('evidence/model_comparison/week11_run01'))
    args=p.parse_args();root=args.evaluation
    plan=json.loads((root/'plan.json').read_text());bench=rows(root/'benchmark_summary.csv')
    count=0;historical_predictions=0
    for row in bench:
        task,name=row['task'],row['model'];folder=root/'benchmarks'/task/name
        predictions=np.load(folder/'predictions.npz')
        truth,guess=predictions['truth'],predictions['prediction']
        report=(classification_metrics if task=='mnist' else metrics)(truth,guess)
        assert len(truth)==(10000 if task=='mnist' else 116323)
        assert int(row['correct'])==report['correct']
        assert abs(float(row['accuracy'])-report['accuracy'])<1e-12
        assert abs(float(row['macro_f1'])-report['macro_f1'])<1e-12
        assert json.loads((folder/'metrics.json').read_text())==report
        if name in ('mlp','cnn'):
            old=Path('experiments/baseline/evaluation' if task=='mnist' else 'evidence/emnist/evaluation_run_01')/f'{name}_predictions.csv'
            previous=rows(old)
            truth_key='true_digit' if task=='mnist' else 'true_id'
            guess_key='predicted_digit' if task=='mnist' else 'predicted_id'
            assert np.array_equal(truth,np.array([int(r[truth_key]) for r in previous]))
            assert np.array_equal(guess,np.array([int(r[guess_key]) for r in previous]))
            historical_predictions+=len(truth)
        count+=len(truth)
    old_prep=rows(Path('evidence/preprocessing_experiment/week10_run01/predictions.csv'))
    old_seg=rows(Path('evidence/segmentation_experiment/week10_combined01/predictions.csv'))
    old_lookup={(int(r['index']),r['model']):r['prediction'] for r in old_prep
        if (int(r['index'])<40 and r['variant']=='C_peak') or (40<=int(r['index'])<120 and r['variant']=='baseline')}
    old_lookup.update({(int(r['index']),r['model']):r['prediction'] for r in old_seg if r['variant']=='combined'})
    pairs=rows(root/'paired_photos.csv');photos=rows(root/'photo_predictions.csv')
    assert len(pairs)==160 and len(photos)==640
    assert len({(r['index'],r['model']) for r in photos})==640
    for pair in pairs:
        i=int(pair['index']);a=np.load(root/'prepared'/f'{i:03d}.npy')
        old_array=(Path('evidence/preprocessing_experiment/week10_run01/prepared')/f'{i:03d}_C_peak.npy' if i<40 else
                   Path('evidence/preprocessing_experiment/week10_run01/prepared')/f'{i:03d}_baseline.npy' if i<120 else
                   Path('evidence/segmentation_experiment/week10_combined01')/f'{i:03d}'/'combined/prepared.npy')
        assert np.array_equal(a,np.load(old_array)),str(old_array)
        assert sha256(pair['file'])==pair['sha256']
        for name in ('mlp','cnn'):assert pair[name]==old_lookup[i,name],(i,name)
    lookup={(r['index'],r['model']):r for r in photos}
    for pair in pairs:
        for name in ('mlp','cnn','resnet18','mobilenet_v3_small'):
            r=lookup[pair['index'],name]
            assert r['prediction']==pair[name] and r['reference']==pair['reference']
            assert int(r['correct'])==int(r['prediction']==r['reference'])
            assert int(r['edits'])==edit_distance(r['reference'],r['prediction'])
    for summary in rows(root/'photo_summary.csv'):
        selected=[r for r in photos if r['collection']==summary['collection'] and r['model']==summary['model']]
        assert int(summary['count'])==len(selected)
        assert int(summary['correct'])==sum(int(r['correct']) for r in selected)
        assert abs(float(summary['cer'])-sum(int(r['edits']) for r in selected)/sum(len(r['reference']) for r in selected))<1e-12
    for record in plan['models']:assert sha256(record['checkpoint'])==record['sha256']
    for name,digest in plan['source_sha256'].items():assert sha256(name)==digest
    result=dict(status='PASS',benchmark_predictions_recalculated=count,historical_baseline_test_predictions_reproduced=historical_predictions,
        prepared_arrays_equal_to_week10=160,historical_development_predictions_reproduced=320,photo_predictions_checked=640,
        checkpoint_and_source_hashes_unchanged=True,scope='Independent saved-result recalculation and historical reproduction; no new training or final-photo evaluation')
    (root/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
