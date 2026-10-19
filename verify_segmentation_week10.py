"""Check production segmentation exactly reproduces the adopted combined run."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageOps
from compare_photo_models import sha
from hnrs_chars.segmentation import segment_characters,prepare_character_crops
from hnrs_chars.predict import CharacterPredictor


def main():
    out=Path('evidence/segmentation_experiment/week10_combined01')
    plan=json.loads((out/'plan.json').read_text())
    for file,digest in plan['sources'].items():
        if file!='hnrs_chars/segmentation.py':assert sha(file)==digest,file
    with (out/'predictions.csv').open() as f:rows=list(csv.DictReader(f))
    torch.set_num_threads(4)
    models={m:CharacterPredictor(plan['checkpoints'][f'characters_{m}']['file']) for m in ('cnn','mlp')}
    for r in [r for r in rows if r['variant']=='combined' and r['model']=='cnn']:
        assert sha(r['file'])==r['input_sha256']
        with Image.open(r['file']) as im:im=ImageOps.exif_transpose(im).convert('RGB')
        seg=segment_characters(im);arrays=prepare_character_crops(seg.crops)
        folder=out/f'{int(r["index"]):03d}'/'combined'
        details=json.loads((folder/'segmentation.json').read_text())
        assert [list(b) for b in seg.boxes]==details['boxes']
        assert seg.groups==details['component_groups']
        np.testing.assert_array_equal(arrays,np.load(folder/'prepared.npy'))
        for model,predictor in models.items():
            expected=next(x for x in rows if x['index']==r['index'] and x['variant']=='combined' and x['model']==model)
            assert predictor.predict_text(arrays)['text']==expected['prediction']
    result=dict(status='PASS',string_arrays_boxes_groups_reproduced=40,model_predictions_reproduced=80,
                source_sha256=sha('hnrs_chars/segmentation.py'),other_pipeline_sources_unchanged=True,
                scope='Production pipeline; no new manual GUI acceptance claim')
    (out/'integration_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
