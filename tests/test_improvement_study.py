import unittest
import numpy as np
import torch
from PIL import Image,ImageDraw
from research.improvement_candidates import merge_fragments,segment_repaired_digits,padded_character,mask_characters
from segment_digits import segment_image
from hnrs_chars.segmentation import segment_characters
from research.train_improvements import augment


class ImprovementTests(unittest.TestCase):
    def test_disconnected_four_repaired_without_merging_next_digit(self):
        image=Image.new('L',(150,110),255);d=ImageDraw.Draw(image)
        d.line([(20,15),(10,65),(49,65)],fill=0,width=5)
        d.line([(54,10),(54,95)],fill=0,width=5)
        d.ellipse((85,15,130,95),outline=0,width=5)
        self.assertEqual(len(segment_image(image).boxes),3)
        self.assertEqual(len(segment_repaired_digits(image).boxes),2)

    def test_two_narrow_digits_stay_separate(self):
        self.assertEqual(merge_fragments([(0,0,4,80),(6,0,10,80)]),[[0],[1]])

    def test_distant_wide_narrow_stay_separate(self):
        self.assertEqual(merge_fragments([(0,0,45,80),(60,0,65,80)]),[[0],[1]])

    def test_dot_survives_candidate_preparation(self):
        image=Image.new('L',(70,110),255);d=ImageDraw.Draw(image)
        d.ellipse((29,10,40,21),fill=0);d.rectangle((31,40,37,95),fill=0)
        segmented=segment_characters(image)
        self.assertEqual(len(segmented.boxes),1)
        for a in (padded_character(image),mask_characters(segmented)[0]):
            self.assertEqual(a.shape,(28,28));self.assertTrue(np.isfinite(a).all())
            # Detached dot and stem remain two separate vertical runs.
            from segment_digits import occupied_intervals
            self.assertEqual(len(occupied_intervals((a>.2).any(axis=1))),2)

    def test_augmentation_reproducible_independent_and_bounded(self):
        x=torch.zeros(8,1,28,28);x[:,:,6:22,12:16]=1
        a=augment(x,torch.Generator().manual_seed(9))
        b=augment(x,torch.Generator().manual_seed(9))
        self.assertTrue(torch.equal(a,b));self.assertFalse(torch.equal(a[0],a[1]))
        self.assertEqual(a.shape,x.shape);self.assertTrue(torch.isfinite(a).all())
        self.assertGreaterEqual(float(a.min()),0);self.assertLessEqual(float(a.max()),1)
        self.assertTrue(torch.equal(x[:,:,6:22,12:16],torch.ones(8,1,16,4)))


if __name__=='__main__':unittest.main()
