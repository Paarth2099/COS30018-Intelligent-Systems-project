"""Prediction contract, gradients and checkpoint round trips for new architectures."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from hnrs_ml.models import build_model as digits
from hnrs_chars.models import build_model as characters
from hnrs_ml.predict import DigitPredictor
from hnrs_chars.predict import CharacterPredictor
from hnrs_chars.data import CLASSES


class ExtendedModels(unittest.TestCase):
    def test_training_and_checkpoint_contract(self):
        torch.set_num_threads(4)
        for factory, count, predictor in ((digits,10,DigitPredictor),(characters,62,CharacterPredictor)):
            for name in ('resnet18','mobilenet_v3_small'):
                with self.subTest(classes=count, model=name), tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
                    torch.manual_seed(42)
                    model = factory(name)
                    x = torch.rand(2,1,28,28)
                    logits = model(x)
                    self.assertEqual(tuple(logits.shape),(2,count))
                    loss = torch.nn.functional.cross_entropy(logits,torch.tensor([0,count-1]))
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                    saved = dict(format_version=1,model_name=name,state_dict=model.state_dict())
                    if count == 62: saved.update(task='emnist_byclass_62',classes=CLASSES)
                    path=Path(tmp)/'model.pt';torch.save(saved,path)
                    restored=predictor(path)
                    model.eval()
                    with torch.inference_mode():
                        torch.testing.assert_close(model(x),restored.model(x),rtol=0,atol=0)
                        self.assertEqual(tuple(restored.model(x[:1]).shape),(1,count))
                    fn=restored.predict_number if count==10 else restored.predict_text
                    self.assertEqual(len(fn(x.numpy())['text']),2)
                    self.assertEqual(fn(np.zeros((0,28,28),np.float32))['text'],'')

    def test_unknown_architecture(self):
        with self.assertRaises(ValueError): digits('missing')
