"""Character interface and split regressions; no dataset download or training."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from torch import nn
from hnrs_chars.data import CLASSES, stratified_split, batch_images
from hnrs_chars.models import build_model
from hnrs_chars.predict import CharacterPredictor
from hnrs_chars.evaluate import metrics


class EncodedPixels(nn.Module):
    """Deterministic stub makes ordering and case checks independent of accuracy."""
    def forward(self,x):
        ids=(x[:,0,0,0]*61).round().long()
        return nn.functional.one_hot(ids,62).float()*10


class CharacterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_stratified_split_disjoint_repeatable_and_all_classes(self):
        targets=torch.cat([torch.full((20+i,),i) for i in range(62)])
        a,b=stratified_split(targets);c,d=stratified_split(targets)
        self.assertTrue(torch.equal(a,c) and torch.equal(b,d))
        self.assertFalse(set(a.tolist()) & set(b.tolist()))
        self.assertEqual(sorted(torch.cat([a,b]).tolist()),list(range(len(targets))))
        self.assertEqual(len(torch.unique(targets[b])),62)
        self.assertFalse(torch.equal(a,stratified_split(targets,seed=9)[0]))

    def test_model_save_load_roundtrip_both_architectures(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ('mlp','cnn'):
                model=build_model(name).eval();images=torch.rand(3,1,28,28)
                path=Path(directory)/f'{name}.pt'
                torch.save({'format_version':1,'task':'emnist_byclass_62','classes':CLASSES,
                            'model_name':name,'state_dict':model.state_dict()},path)
                loaded=CharacterPredictor(path)
                self.assertEqual(tuple(model(images).shape),(3,62))
                self.assertTrue(torch.equal(model(images),loaded.model(images)))

    def test_mapping_case_leading_zeros_and_order_across_batches(self):
        predictor=CharacterPredictor.__new__(CharacterPredictor);predictor.model=EncodedPixels();predictor.model_name='stub'
        expected='007AaZz';images=np.zeros((len(expected),28,28),np.float32)
        for i,char in enumerate(expected):images[i,0,0]=CLASSES.index(char)/61
        result=predictor.predict_characters(images,batch_size=2)
        self.assertEqual(''.join(r['character'] for r in result),expected)
        self.assertEqual(predictor.predict_text(images)['text'],expected)
        self.assertEqual(predictor.predict_text(np.empty((0,28,28),np.float32))['text'],'')

    def test_bad_checkpoint_and_bad_crops_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'digit.pt';torch.save({'format_version':1,'model_name':'cnn'},path)
            with self.assertRaises(ValueError):CharacterPredictor(path)
        predictor=CharacterPredictor.__new__(CharacterPredictor);predictor.model=EncodedPixels();predictor.model_name='stub'
        for images in [np.ones((1,28,28),np.uint8),np.zeros((1,27,28),np.float32),
                       np.full((1,28,28),np.nan,np.float32),np.full((1,28,28),2,np.float32)]:
            with self.assertRaises(ValueError):predictor.predict_characters(images)

    def test_metrics_keep_uppercase_lowercase_distinct(self):
        result=metrics(np.array([10,36,61,0]),np.array([10,10,61,1]))
        self.assertEqual(result['accuracy'],.5)
        self.assertEqual(result['confusion_matrix'][36][10],1)
        self.assertEqual(result['groups']['lowercase']['accuracy'],.5)
        self.assertEqual(result['per_class'][61]['f1'],1.)

    def test_batch_conversion_preserves_upright_pixels_and_source(self):
        images=torch.zeros(2,28,28,dtype=torch.uint8);images[1,3,7]=255
        batch=batch_images(images,torch.tensor([1,0]))
        self.assertEqual(batch[0,0,3,7].item(),1.)
        self.assertEqual(batch[0,0,7,3].item(),0.)
        self.assertEqual(images[1,3,7].item(),255)


if __name__=='__main__':unittest.main()
