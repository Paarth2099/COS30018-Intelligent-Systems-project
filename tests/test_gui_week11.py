"""Mode-specific checkpoint routing and integrity guards for the four-model GUI."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from gui_core import resolve_checkpoint, DEFAULT_MODELS


class CheckpointRoutingTests(unittest.TestCase):
    def make_run(self,root,task='emnist',model='resnet18'):
        folder=root/'experiments/week11_gpu_run01'/task/model;folder.mkdir(parents=True)
        checkpoint=folder/f'{model}.pt';checkpoint.write_bytes(b'fixture checkpoint')
        selection=dict(model=model,config={'task':task},checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        (folder/'selection.json').write_text(json.dumps(selection))
        return checkpoint,folder/'selection.json'

    def test_task_routes_are_separate(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root=Path(tmp)
            for model in ('resnet18','mobilenet_v3_small'):
                for mode,task in [('Numbers','mnist'),('Alphanumeric','emnist')]:
                    checkpoint,_=self.make_run(root,task,model)
                    self.assertEqual(resolve_checkpoint(root,mode,model)[0],checkpoint)
            self.assertEqual(DEFAULT_MODELS,{'Numbers':'cnn','Alphanumeric':'resnet18'})

    def test_corruption_and_wrong_task_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root=Path(tmp);checkpoint,selection_path=self.make_run(root)
            checkpoint.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):resolve_checkpoint(root,'Alphanumeric','resnet18')
            selection=json.loads(selection_path.read_text());selection['config']['task']='mnist'
            selection_path.write_text(json.dumps(selection))
            with self.assertRaisesRegex(ValueError,'does not match'):resolve_checkpoint(root,'Alphanumeric','resnet18')

    def test_missing_candidate_does_not_fall_back(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            with self.assertRaises(FileNotFoundError):resolve_checkpoint(Path(tmp),'Alphanumeric','resnet18')
            with self.assertRaisesRegex(ValueError,'Unsupported'):resolve_checkpoint(Path(tmp),'Numbers','../cnn')
