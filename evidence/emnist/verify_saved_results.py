"""Audit saved experiment evidence; run from the repository root."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

root = Path.cwd()
experiment = root / 'experiments/emnist_byclass/run_01'
evaluation = root / 'evidence/emnist/evaluation_run_01'
selection = json.loads((experiment / 'selection.json').read_text())
results = json.loads((evaluation / 'results.json').read_text())
assert results['selection_sha256'] == digest(experiment / 'selection.json')
assert selection['selected_before_test_evaluation']
assert selection['selected_model'] == results['selected_model']
checks = {}
for model in results['models']:
    name = model['model']
    with (evaluation / f'{name}_predictions.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == model['count'] == 116323
    assert [int(r['test_index']) for r in rows] == list(range(116323))
    correct = sum(r['true_id'] == r['predicted_id'] for r in rows)
    assert correct == model['correct']
    assert abs(correct / len(rows) - model['accuracy']) < 1e-12
    assert sum(map(sum, model['confusion_matrix'])) == len(rows)
    assert sum(model['confusion_matrix'][i][i] for i in range(62)) == correct
    assert sum(r['support'] for r in model['per_class']) == len(rows)
    assert digest(experiment / f'{name}.pt') == model['checkpoint_sha256']
    checks[name] = {'count': len(rows), 'correct': correct, 'accuracy': model['accuracy']}
for name, expected in selection['config']['source_sha256'].items():
    assert digest(root / 'hnrs_chars' / name) == expected, name
example = json.loads((evaluation / 'interface_example.json').read_text())
demo = json.loads((root / 'evidence/emnist/cli_demo.json').read_text(encoding='utf-8-sig'))
assert demo == example['predictions'][selection['selected_model']]
subprocess.run(['git', 'diff', '--exit-code', '73e11d0', '--',
                'app.py', 'gui_core.py', 'hnrs_ml', 'experiments/baseline'], check=True)
test_log = (root / 'evidence/emnist/unit_tests.txt').read_text(encoding='utf-8-sig')
assert 'Ran 24 tests' in test_log and test_log.rstrip().endswith('OK')
audit = {'verified_utc': datetime.now(timezone.utc).isoformat(), 'models': checks,
         'checks': ['Prediction CSV counts and accuracy match saved metrics',
                    'Confusion counts and class support match test set',
                    'Checkpoint and training source hashes match saved run',
                    'CLI output matches evaluated public interface',
                    'Original digit source and checkpoints unchanged from 73e11d0',
                    'Saved software test log records 24 passing tests'],
         'report_sha256': digest(root / 'docs/reports/emnist-results.md'),
         'current_source_sha256': {p.name: digest(p) for p in sorted((root / 'hnrs_chars').glob('*.py'))}}
(root / 'evidence/emnist/verification.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps(audit['models'], indent=2))
print('Saved evidence audit passed.')
