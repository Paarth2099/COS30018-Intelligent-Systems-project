"""Character geometry/integration regressions, independent of handwriting scores."""
from pathlib import Path
import unittest
import numpy as np
from PIL import Image, ImageDraw
from hnrs_chars.segmentation import METHODS, segment_characters, prepare_character_crops, recognise_text, group_components
from hnrs_chars.string_experiment import parse_label, summarise


class RecordingPredictor:
    def predict_text(self,arrays):
        self.arrays=arrays
        return {'text':'0aI'[:len(arrays)],'characters':[],'model':'stub','task':'emnist_byclass_62'}


class CharacterSegmentationTests(unittest.TestCase):
    def test_dot_and_stem_grouped_without_merging_neighbour(self):
        image=Image.new('L',(180,150),255);d=ImageDraw.Draw(image)
        d.rectangle((35,50,40,125),fill=0);d.ellipse((34,20,41,27),fill=0)
        d.rectangle((110,30,120,125),fill=0)
        for method in METHODS:
            seg=segment_characters(image,method)
            self.assertEqual(len(seg.boxes),2)
            self.assertEqual(seg.boxes[0],(34,20,42,126))
            self.assertEqual(len(seg.groups[0]),2)

    def test_offset_dot_is_grouped_where_projection_splits_it(self):
        image=Image.new('L',(180,150),255);d=ImageDraw.Draw(image)
        d.rectangle((35,50,40,125),fill=0);d.rectangle((48,20,54,27),fill=0)
        d.rectangle((110,30,120,125),fill=0)
        self.assertEqual(len(segment_characters(image,'projection').boxes),3)
        seg=segment_characters(image,'grouped_components')
        self.assertEqual(len(seg.boxes),2);self.assertEqual(len(seg.groups[0]),2)

    def test_blank_returns_empty_batch_and_prediction(self):
        for method in METHODS:
            seg=segment_characters(Image.new('L',(90,90),255),method)
            self.assertEqual(seg.boxes,[])
            arrays=prepare_character_crops(seg.crops)
            self.assertEqual(arrays.shape,(0,28,28));self.assertEqual(arrays.dtype,np.float32)
            result=recognise_text(Image.new('L',(90,90),255),RecordingPredictor(),method)
            self.assertEqual(result['prediction']['text'],'')

    def test_public_pipeline_orders_and_passes_float_arrays(self):
        image=Image.new('L',(200,130),255);d=ImageDraw.Draw(image)
        for x in (130,20,75):d.rectangle((x,25,x+8,110),fill=0)
        predictor=RecordingPredictor();result=recognise_text(image,predictor)
        self.assertEqual([b[0] for b in result['segmentation'].boxes],[20,75,130])
        self.assertEqual(result['prediction']['text'],'0aI')
        self.assertEqual(predictor.arrays.shape,(3,28,28));self.assertEqual(predictor.arrays.dtype,np.float32)
        self.assertTrue(np.isfinite(predictor.arrays).all())
        self.assertTrue(0<=predictor.arrays.min()<=predictor.arrays.max()<=1)

    def test_overlap_boxes_do_not_contaminate_other_character(self):
        image=Image.new('L',(120,120),255);d=ImageDraw.Draw(image)
        d.line([(20,15),(20,90),(85,90)],fill=0,width=3)
        own=int((np.asarray(image)<128).sum())
        d.line([(50,20),(75,20),(75,65)],fill=0,width=3)
        seg=segment_characters(image,'grouped_components')
        self.assertEqual(len(seg.crops),2)
        self.assertEqual(int((np.asarray(seg.crops[0])<128).sum()),own)

    def test_touching_characters_are_not_split_using_expected_count(self):
        image=Image.new('L',(140,120),255);d=ImageDraw.Draw(image)
        d.rectangle((15,25,25,95),fill=0);d.rectangle((85,25,95,95),fill=0)
        d.line((25,60,85,60),fill=0,width=3)
        for method in METHODS:self.assertEqual(len(segment_characters(image,method).boxes),1)

    def test_noise_removed_but_small_dot_retained(self):
        image=Image.new('L',(100,150),255);d=ImageDraw.Draw(image)
        d.rectangle((40,50,45,130),fill=0);d.rectangle((40,20,42,22),fill=0);d.point((90,90),fill=0)
        seg=segment_characters(image)
        self.assertEqual(len(seg.boxes),1);self.assertEqual(seg.boxes[0][1],20)
        self.assertEqual(seg.removed_components,1)

    def test_exif_orientation_matches_physical_rotation(self):
        image=Image.new('L',(150,100),255);d=ImageDraw.Draw(image)
        d.rectangle((20,20,30,80),fill=0);d.rectangle((80,20,95,80),fill=0)
        rotated=image.transpose(Image.Transpose.ROTATE_90);rotated.getexif()[274]=6
        self.assertEqual(segment_characters(image).boxes,segment_characters(rotated).boxes)

    def test_invalid_method_and_small_resolution_rejected(self):
        image=Image.new('L',(60,60),255)
        with self.assertRaises(ValueError):segment_characters(image,'unknown')
        with self.assertRaises(ValueError):segment_characters(image,max_side=10)

    def test_labels_preserve_case_zeros_and_metrics_count_insertions(self):
        self.assertEqual(parse_label(Path('Manula/Manula_07_text-007Ab.jpg')),('Manula','007Ab'))
        with self.assertRaises(ValueError):parse_label(Path('Paarth/Manula_01_text-Ab12.jpg'))
        row={'reference':'i','count_correct':0,'exact_correct':0,'edit_distance':2,'error':'',
             'segmentation_ms':1,'preparation_ms':2,'inference_ms':3}
        self.assertEqual(summarise([row])['character_error_rate'],2.)

    def test_short_i_beside_tall_character_retains_dot(self):
        image=Image.new('L',(180,180),255);d=ImageDraw.Draw(image)
        d.rectangle((20,20,30,160),fill=0)
        d.rectangle((100,105,106,155),fill=0);d.ellipse((99,80,107,88),fill=0)
        seg=segment_characters(image,'grouped_components')
        self.assertEqual(len(seg.boxes),2);self.assertEqual(len(seg.groups[1]),2)

    def test_tiny_pen_gap_closed_without_filling_grayscale_crop(self):
        image=Image.new('L',(120,120),255);d=ImageDraw.Draw(image)
        d.rectangle((20,20,25,95),fill=0);d.rectangle((55,20,60,95),fill=0)
        d.rectangle((25,54,53,58),fill=0)
        ink=int((np.asarray(image)<128).sum())
        seg=segment_characters(image,'grouped_components')
        self.assertEqual(len(seg.boxes),1)
        self.assertEqual(int((np.asarray(seg.crops[0])<128).sum()),ink)

    def test_fragmented_dot_attaches_to_short_thin_stem(self):
        image=Image.new('L',(220,260),255);d=ImageDraw.Draw(image)
        d.rectangle((20,30,35,230),fill=0)
        d.rectangle((125,120,130,180),fill=0)
        d.rectangle((120,70,136,85),fill=0);d.rectangle((147,73,150,80),fill=0)
        seg=segment_characters(image,'grouped_components')
        self.assertEqual(len(seg.boxes),2)
        self.assertEqual(seg.boxes[1],(120,70,151,181))

    def test_dot_prefers_aligned_stem_over_neighbour_box(self):
        # Compact dot lies inside tall neighbour's box but directly above i.
        regions=[{'id':1,'box':[10,10,90,150],'area':2000},
                 {'id':2,'box':[80,80,88,150],'area':560},
                 {'id':3,'box':[79,40,89,50],'area':60}]
        groups,_=group_components(regions)
        self.assertEqual([[r['id'] for r in g] for g in groups],[[1],[2,3]])


if __name__=='__main__':unittest.main()
