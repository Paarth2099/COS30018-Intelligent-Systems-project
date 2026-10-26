"""Small synthetic checks of training, selection and CPU-portable checkpoints."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from research import train_models_week11 as training
from hnrs_ml.predict import DigitPredictor
from hnrs_chars.predict import CharacterPredictor


class TrainingDeviceTests(unittest.TestCase):
    def test_cuda_request_cannot_silently_fall_back(self):
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'CUDA was requested'):
                training.select_device('cuda')
            self.assertEqual(training.select_device('auto').type,'cpu')

    def test_cpu_batch_scaling(self):
        images=torch.full((2,28,28),255,dtype=torch.uint8)
        batch=training.batch(images,torch.tensor([1]),torch.device('cpu'))
        self.assertEqual(tuple(batch.shape),(1,1,28,28))
        self.assertEqual(batch.dtype,torch.float32)
        self.assertTrue(torch.all(batch==1))

    def exercise_training(self,device):
        torch.set_num_threads(4)
        for task,factory in [('mnist',DigitPredictor),('emnist',CharacterPredictor)]:
            for model in training.NAMES:
                with self.subTest(task=task,model=model,device=str(device)), tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
                    torch.manual_seed(7)
                    images=torch.randint(0,256,(7,28,28),dtype=torch.uint8)
                    labels=torch.tensor([0,1,2,3,4,0,1])
                    fixture=(images,labels,torch.arange(5),torch.arange(5,7))
                    with patch.object(training,'data',return_value=fixture):
                        training.train(task,model,Path(tmp),1,device,4)
                    folder=Path(tmp)/task/model
                    saved=torch.load(folder/f'{model}.pt',map_location='cpu',weights_only=True)
                    self.assertTrue(all(value.device.type=='cpu' for value in saved['state_dict'].values()))
                    predictor=factory(folder/f'{model}.pt')
                    with torch.inference_mode():
                        result=predictor.model(training.batch(images,torch.arange(2),torch.device('cpu')))
                    self.assertTrue(torch.isfinite(result).all())
                    selection=json.loads((folder/'selection.json').read_text())
                    self.assertEqual(selection['best_epoch'],1)
                    self.assertEqual(selection['config']['device'],str(device))
                    self.assertFalse(selection['config']['test_data_used'])
                    self.assertEqual(json.loads((folder/'progress.json').read_text())['status'],'completed')
                    # Completed runs are hash-checked, not retrained or overwritten.
                    with patch.object(training,'data',side_effect=AssertionError('Should skip')):
                        training.train(task,model,Path(tmp),1,device,4)

    def test_synthetic_cpu_training(self):
        self.exercise_training(torch.device('cpu'))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable on this computer')
    def test_synthetic_cuda_training(self):
        self.exercise_training(torch.device('cuda'))
