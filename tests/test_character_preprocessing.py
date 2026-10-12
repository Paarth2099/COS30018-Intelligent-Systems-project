"""Behavioural checks for character-photo preparation and label integrity."""
from pathlib import Path
import unittest
import numpy as np
from PIL import Image, ImageDraw
from hnrs_chars.preprocessing import prepare_character, prepare_character_variants
from hnrs_chars.photo_experiment import label_from_path


class CharacterPreprocessingTests(unittest.TestCase):
    def test_detached_dot_and_stem_survive_every_variant(self):
        image=Image.new('L',(100,160),255);draw=ImageDraw.Draw(image)
        draw.ellipse((44,20,56,32),fill=0);draw.rectangle((46,60,54,140),fill=0)
        variants,_,details=prepare_character_variants(image)
        self.assertEqual(details['bbox'],[44,20,57,141])
        for array in variants.values():
            self.assertEqual(array.shape,(28,28));self.assertEqual(array.dtype,np.float32)
            self.assertTrue(np.isfinite(array).all())
            self.assertGreater(array[4:7].max(),.2)
            self.assertGreater(array[14:23].max(),.2)
            self.assertLess(array[8:10].max(),.05)
            self.assertTrue((array[:4]==0).all() and (array[24:]==0).all())
            self.assertGreaterEqual(array.min(),0);self.assertLessEqual(array.max(),1)

    def test_exif_orientation_matches_physically_rotated_image(self):
        image=Image.new('RGB',(120,80),'white');draw=ImageDraw.Draw(image)
        draw.rectangle((20,10,28,65),fill='black');draw.rectangle((20,10,90,18),fill='black')
        image.getexif()[274]=6
        expected=image.transpose(Image.Transpose.ROTATE_270);expected.getexif()[274]=1
        np.testing.assert_array_equal(prepare_character(image),prepare_character(expected))

    def test_blank_faint_and_unknown_method_rejected(self):
        for value in (0,128,255):
            with self.assertRaises(ValueError):prepare_character(Image.new('L',(80,80),value))
        image=Image.new('L',(80,80),250);ImageDraw.Draw(image).line((20,20,60,60),fill=245,width=3)
        with self.assertRaises(ValueError):prepare_character(image)
        with self.assertRaises(ValueError):prepare_character(image,method='magic')

    def test_clipped_foreground_warned_and_input_not_mutated(self):
        image=Image.new('L',(100,100),255);ImageDraw.Draw(image).rectangle((0,20,8,70),fill=0)
        before=image.tobytes();_,_,details=prepare_character_variants(image)
        self.assertTrue(any('edge' in w for w in details['warnings']))
        self.assertEqual(image.size,(100,100));self.assertEqual(image.tobytes(),before)

    def test_aspect_ratio_preserved_for_wide_character(self):
        image=Image.new('L',(200,100),255);ImageDraw.Draw(image).rectangle((40,40,159,59),fill=0)
        _,_,details=prepare_character_variants(image)
        self.assertEqual(details['resized_ink_size'],[20,3])

    def test_case_explicit_labels_and_writer_validation(self):
        self.assertEqual(label_from_path(Path('Paarth/Paarth_01_upper-A.jpg')),('Paarth','upper','A'))
        self.assertEqual(label_from_path(Path('Paarth/Paarth_02_lower-a.jpg')),('Paarth','lower','a'))
        for path in ('Paarth/Paarth_01_upper-a.jpg','Paarth/Paarth_01_digit-A.jpg',
                     'Paarth/Paarth_01_lower-aa.jpg','Haresh/Paarth_01_lower-a.jpg'):
            with self.assertRaises(ValueError):label_from_path(Path(path))

    def test_tiny_speck_removed_but_small_detached_dot_retained(self):
        image=Image.new('L',(120,160),255);draw=ImageDraw.Draw(image)
        draw.rectangle((55,55,59,140),fill=0)
        draw.rectangle((55,20,56,22),fill=0)
        draw.point((110,150),fill=0)
        _,_,details=prepare_character_variants(image)
        self.assertEqual(details['bbox'],[55,20,60,141])
        self.assertEqual(details['removed_components'],1)

    def test_downsampled_thin_ink_is_normalised_without_shape_change(self):
        image=Image.new('L',(300,300),255)
        ImageDraw.Draw(image).ellipse((60,30,240,270),outline=0,width=2)
        before=prepare_character(image,normalise_ink=False)
        after=prepare_character(image,normalise_ink=True)
        self.assertGreater(after.max(),before.max())
        self.assertAlmostEqual(float(after.max()),1.)
        np.testing.assert_array_equal(before>0,after>0)

    def test_smooth_lighting_gradient_does_not_expand_ink_crop(self):
        background=np.tile(np.linspace(140,245,200,dtype=np.uint8),(180,1))
        image=Image.fromarray(background);ImageDraw.Draw(image).rectangle((90,30,100,150),fill=20)
        _,_,details=prepare_character_variants(image)
        self.assertEqual(details['bbox'],[90,30,101,151])
        self.assertFalse(details['warnings'])


if __name__=='__main__':unittest.main()
