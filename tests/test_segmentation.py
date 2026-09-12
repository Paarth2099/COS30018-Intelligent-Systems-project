"""Synthetic geometry checks; these do not establish handwriting accuracy."""
import unittest
import numpy as np
from PIL import Image, ImageDraw
from segment_digits import connected_regions, segment_image, prepare_crops
from compare_segmentation import edit_distance


class SegmentationTests(unittest.TestCase):
    def test_diagonal_connectivity_and_separation(self):
        labels,regions=connected_regions(np.array([[1,0,0,0],[0,1,0,1],[0,0,0,1]],dtype=bool))
        self.assertEqual(sorted(r['area'] for r in regions),[2,2])
        self.assertEqual(labels[0,0],labels[1,1])
        self.assertNotEqual(labels[1,1],labels[1,3])

    def test_left_to_right_order_and_model_format(self):
        im=Image.new('L',(180,100),255); d=ImageDraw.Draw(im)
        d.rectangle((125,15,145,85),fill=0); d.rectangle((15,20,22,80),fill=0)
        d.ellipse((55,15,90,85),outline=0,width=5)
        for method in ['components','projection']:
            seg=segment_image(im,method)
            self.assertEqual(len(seg.boxes),3)
            self.assertEqual([b[0] for b in seg.boxes],[15,55,125])
            arrays=prepare_crops(seg.crops)
            self.assertEqual(arrays.shape,(3,28,28)); self.assertEqual(arrays.dtype,np.float32)
            self.assertTrue(0<=arrays.min()<=arrays.max()<=1)

    def test_empty_input(self):
        for method in ['components','projection']:
            seg=segment_image(Image.new('L',(100,100),255),method)
            self.assertEqual(seg.boxes,[]); self.assertEqual(prepare_crops(seg.crops).shape,(0,28,28))

    def test_noise_removed(self):
        im=Image.new('L',(100,100),255); d=ImageDraw.Draw(im)
        d.rectangle((40,20,50,80),fill=0); d.point((3,3),fill=0)
        for method in ['components','projection']:
            seg=segment_image(im,method); self.assertEqual(len(seg.boxes),1)
            self.assertEqual(seg.removed_components,1)

    def test_methods_differ_for_disconnected_vertical_parts(self):
        im=Image.new('L',(100,100),255); d=ImageDraw.Draw(im)
        d.rectangle((30,10,50,35),fill=0); d.rectangle((30,50,50,85),fill=0)
        self.assertEqual(len(segment_image(im,'components').boxes),2)
        self.assertEqual(len(segment_image(im,'projection').boxes),1)

    def test_touching_digits_are_not_magically_split(self):
        im=Image.new('L',(100,100),255); d=ImageDraw.Draw(im)
        d.rectangle((10,20,30,80),fill=0); d.rectangle((60,20,80,80),fill=0)
        d.line((30,50,60,50),fill=0,width=2)
        for method in ['components','projection']:
            self.assertEqual(len(segment_image(im,method).boxes),1)

    def test_edit_distance_including_leading_zero(self):
        self.assertEqual(edit_distance('007','07'),1)
        self.assertEqual(edit_distance('123','193'),1)
        self.assertEqual(edit_distance('123',''),3)
        self.assertEqual(edit_distance('123','1234'),1)

    def test_overlapping_component_boxes_do_not_leak_neighbour_ink(self):
        im=Image.new('L',(100,100),255); d=ImageDraw.Draw(im)
        d.line([(20,15),(20,70),(70,70)],fill=0,width=3)
        isolated=np.asarray(im).copy()
        d.line([(45,20),(65,20),(65,50)],fill=0,width=3)
        seg=segment_image(im,'components')
        self.assertEqual(len(seg.crops),2)
        self.assertEqual(int((np.asarray(seg.crops[0])<128).sum()),int((isolated<128).sum()))


if __name__=='__main__': unittest.main()
