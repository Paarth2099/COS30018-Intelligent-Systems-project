"""Guard the experiment against validation-label leakage and missing classes."""
import unittest
import torch
from research.week10_weighted import class_weights


class WeightTests(unittest.TestCase):
    def test_training_counts_only(self):
        training=torch.cat([torch.arange(62),torch.zeros(3,dtype=torch.long)])
        labels=torch.cat([training,torch.tensor([0,1,2])])
        ids=torch.arange(len(training))
        weights,counts=class_weights(labels,ids)
        labels[-3:]=61
        other,_=class_weights(labels,ids)
        self.assertTrue(torch.equal(weights,other))
        self.assertEqual(int(counts[0]),4)
        self.assertAlmostEqual(float(weights[1]/weights[0]),2.)
        self.assertAlmostEqual(float(weights.mean()),1.,places=6)

    def test_missing_training_class_rejected(self):
        with self.assertRaisesRegex(ValueError,'Every class'):
            class_weights(torch.arange(61),torch.arange(61))
