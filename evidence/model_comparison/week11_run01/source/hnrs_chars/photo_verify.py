"""Audit saved photo experiments against current inputs and public API."""
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
import torch
from .data import sha256
from .preprocessing import DEFAULT_METHOD, prepare_character
from .predict import CharacterPredictor


def read_rows(path):
    with path.open(encoding='utf-8-sig') as stream:return list(csv.DictReader(stream))


def main():
    base=Path('evidence/character_preprocessing')
    inputs=Path('data/own_characters/development')
    reference=None; result=[]
    for number in range(1,5):
        directory=base/f'run_{number:02d}'
        summary=json.loads((directory/'summary.json').read_text())
        manifest=read_rows(directory/'manifest.csv');rows=read_rows(directory/'results.csv')
        assert len(manifest)==80 and len(rows)==240
        assert Counter(r['writer'] for r in manifest)==dict.fromkeys(('Haresh','Manula','Paarth','Saniru'),20)
        assert len({r['label'] for r in manifest})==20
        assert set(Counter(r['label'] for r in manifest).values())=={4}
        if reference is None:reference=manifest
        assert manifest==reference, 'Inputs or labels changed between runs'
        for item in manifest:assert sha256(inputs/item['file'])==item['sha256']
        assert sha256(Path(summary['checkpoint']))==summary['checkpoint_sha256']
        for name,digest in summary['source_sha256'].items():
            assert sha256(directory/'source'/name)==digest
        for method in ('grayscale','fixed','otsu'):
            group=[r for r in rows if r['method']==method]
            assert len(group)==80 and len({r['file'] for r in group})==80
            assert sum(r['label']==r['prediction'] for r in group)==summary['methods'][method]['correct']
            assert sum(bool(r['error']) for r in group)==summary['methods'][method]['preprocessing_failures']
            for row in group:
                assert int(row['correct'])==int(row['label']==row['prediction'])
        result.append({'run':number,'summary_sha256':sha256(directory/'summary.json'),
                       'manifest_sha256':sha256(directory/'manifest.csv'),
                       'results_sha256':sha256(directory/'results.csv')})
    final=base/'run_04';summary=json.loads((final/'summary.json').read_text())
    assert DEFAULT_METHOD==summary['selected_method']
    torch.set_num_threads(4)
    model=CharacterPredictor(summary['checkpoint'])
    arrays=[];expected=[]
    for row in read_rows(final/'results.csv'):
        if row['method']!=DEFAULT_METHOD:continue
        path=inputs/row['file']
        with Image.open(path) as image:array=prepare_character(image)
        stored=np.load(final/row['writer']/path.stem/f'{DEFAULT_METHOD}.npy',allow_pickle=False)
        np.testing.assert_array_equal(array,stored)
        arrays.append(array);expected.append(row['prediction'])
    predicted=model.predict_characters(np.stack(arrays))
    assert [r['character'] for r in predicted]==expected
    demo=json.loads((base/'demo_verified/prediction.json').read_text())
    demo_row=next(r for r in read_rows(final/'results.csv') if r['file']=='Paarth/Paarth_02_lower-i.jpg' and r['method']==DEFAULT_METHOD)
    assert demo['prediction']['text']==demo_row['prediction']
    log=(base/'unit_tests.txt').read_text(encoding='utf-8-sig')
    assert 'Ran 33 tests' in log and log.rstrip().endswith('OK')
    subprocess.run(['git','diff','--exit-code','4229e8d','--','app.py','gui_core.py',
                    'preprocess_digits.py','segment_digits.py','hnrs_ml','experiments',
                    'hnrs_chars/data.py','hnrs_chars/models.py','hnrs_chars/predict.py'],check=True)
    audit={'verified_utc':datetime.now(timezone.utc).isoformat(),'runs':result,
           'all_80_default_api_arrays_match_saved_results':True,
           'all_80_default_api_predictions_match_saved_results':True,
           'demo_matches_experiment':True,'software_tests':33,
           'baseline_and_model_files_unchanged_from':'4229e8d',
           'current_source_sha256':{p.as_posix():sha256(p) for p in sorted(Path('hnrs_chars').glob('*.py'))},
           'shared_dependencies_sha256':{p:sha256(Path(p)) for p in ('preprocess_digits.py','segment_digits.py','hnrs_ml/predict.py')}}
    (base/'verification.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('Audit passed: four matching 80-photo manifests, 960 result rows, all 80 public-API outputs, CLI demo, 33 tests and unchanged baseline/models.')


if __name__=='__main__':main()
