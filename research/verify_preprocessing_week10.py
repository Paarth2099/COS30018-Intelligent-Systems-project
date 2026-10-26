"""Verify adopted digit preparation against independently recorded candidate arrays."""
import json
import csv
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
from research.compare_photo_models import prepare, sha
from hnrs_ml.predict import DigitPredictor


def main():
    out=Path('evidence/preprocessing_experiment/week10_run01')
    plan=json.loads((out/'plan.json').read_text())
    # Segmentation module contains the sole intended production change.
    for f,h in plan['original_sources'].items():
        if f!='segment_digits.py':assert sha(f)==h,f
    with (out/'predictions.csv').open() as f:rows=list(csv.DictReader(f))
    torch.set_num_threads(4)
    models={m:DigitPredictor(plan['checkpoints'][f'digits_{m}']['file']) for m in ('cnn','mlp')}
    count=0
    for r in rows:
        if r['variant']!='C_peak' or r['model']!='cnn':continue
        assert sha(r['file'])==r['input_sha256']
        with Image.open(r['file']) as im:im=ImageOps.exif_transpose(im).convert('RGB')
        arrays,_,_=prepare(im,r['collection'])
        expected=np.load(out/'prepared'/f'{int(r["index"]):03d}_C_peak.npy')
        np.testing.assert_array_equal(arrays,expected)
        for model,predictor in models.items():
            target=next(x for x in rows if x['index']==r['index'] and x['variant']=='C_peak' and x['model']==model)
            assert predictor.predict_number(arrays)['text']==target['prediction']
        count+=1
    assert count==40
    record=dict(status='PASS',digit_images=40,arrays_exactly_match_candidate=40,predictions_reproduced=80,
                original_sources_unchanged_except_segment_digits=True,production_sha256=sha('segment_digits.py'),
                scope='Real production preparation and saved model inference; no manual GUI acceptance claimed')
    (out/'integration_verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
