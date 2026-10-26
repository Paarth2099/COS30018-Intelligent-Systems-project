import unittest
import numpy as np
from PIL import Image, ImageDraw
from hnrs_chars.preprocessing import prepare_character_variants as baseline
from hnrs_chars.preprocessing_candidates import prepare_character_variants as candidate
from research.experiment_preprocessing_week10 import normalise_digit_batch
from segment_digits import prepare_crops
from preprocess_digits import prepare_variants


class CandidateChecks(unittest.TestCase):
    def test_default_copy_and_detached_dot(self):
        image=Image.new('L',(160,200),240)
        d=ImageDraw.Draw(image);d.rectangle((76,80,84,160),fill=20);d.ellipse((75,50,85,60),fill=20)
        original=baseline(image)
        same=candidate(image)
        for method in original[0]:np.testing.assert_array_equal(original[0][method],same[0][method])
        filtered=candidate(image,filter_distant=True)
        self.assertEqual(filtered[2]['bbox'],original[2]['bbox'])
        np.testing.assert_array_equal(filtered[0]['otsu'],original[0]['otsu'])

    def test_peak_preserves_geometry_and_handles_empty(self):
        a=np.zeros((2,28,28),np.float32);a[0,5:20,13]=.2
        b=normalise_digit_batch(a)
        np.testing.assert_array_equal(a>0,b>0)
        self.assertEqual(b[0].max(),1)
        self.assertFalse(b[1].any())
        self.assertEqual(normalise_digit_batch(a[:0]).shape,(0,28,28))
        self.assertEqual(b.dtype,np.float32)

    def test_blank_rejected(self):
        for kwargs in ({'background_size':15},{'filter_distant':True}):
            with self.assertRaises(ValueError):candidate(Image.new('L',(80,80),255),**kwargs)

    def test_production_digit_integration_and_other_representations(self):
        image=Image.new('L',(400,400),255)
        ImageDraw.Draw(image).line((200,30,200,370),fill=0,width=2)
        variants=prepare_variants(image)[0]
        expected=normalise_digit_batch(variants['otsu'][None])
        np.testing.assert_array_equal(prepare_crops([image]),expected)
        for method in ('fixed','grayscale'):
            np.testing.assert_array_equal(prepare_crops([image],method)[0],variants[method])

    def test_production_preserves_far_detached_dot(self):
        image=Image.new('L',(160,240),240)
        d=ImageDraw.Draw(image);d.rectangle((76,120,84,200),fill=20);d.ellipse((76,30,84,38),fill=20)
        original=baseline(image)
        rejected=candidate(image,filter_distant=True)
        self.assertLess(original[2]['bbox'][1],rejected[2]['bbox'][1])
