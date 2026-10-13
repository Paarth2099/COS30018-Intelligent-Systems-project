"""Character acquisition/export contracts without a desktop window."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from PIL import Image, ImageDraw
from gui_core import character_catalog, generate_text, export_result, RecognitionService
from hnrs_chars.segmentation import segment_characters, prepare_character_crops


def samples(folder):
    for label in ('digit-0','upper-A','lower-a','lower-i'):
        image=Image.new('L',(100,120),255);draw=ImageDraw.Draw(image)
        if label=='lower-i':
            draw.rectangle((44,45,52,100),fill=0);draw.ellipse((44,20,52,28),fill=0)
        else:draw.ellipse((25,15,75,105),outline=0,width=7)
        image.save(folder/f'Writer_01_{label}.png')


class CharacterGuiTests(unittest.TestCase):
    def test_catalog_case_and_string_exclusion(self):
        with TemporaryDirectory() as tmp:
            folder=Path(tmp);samples(folder)
            for name in ('Writer_02_text-Ab12','Writer_03_upper-a','Writer_04_lower-A'):
                Image.new('L',(40,40),255).save(folder/f'{name}.png')
            self.assertEqual(set(character_catalog(folder)),{'0','A','a','i'})

    def test_generation_preserves_case_zeros_provenance_and_sources(self):
        with TemporaryDirectory() as tmp:
            folder=Path(tmp);samples(folder)
            before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
            image,origin=generate_text(folder,'00Aa')
            self.assertEqual(origin['requested_text'],'00Aa')
            self.assertEqual([s['character'] for s in origin['sources']],list('00Aa'))
            self.assertEqual(image.tobytes(),generate_text(folder,'00Aa')[0].tobytes())
            self.assertEqual(before,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()})
            self.assertEqual(len(segment_characters(image).crops),4)

    def test_invalid_missing_case_and_blank_sources(self):
        with TemporaryDirectory() as tmp:
            folder=Path(tmp);samples(folder)
            for value in ('','hello world','a-b','é','A'*13):
                with self.assertRaisesRegex(ValueError,'1 to 12'):generate_text(folder,value)
            with self.assertRaisesRegex(ValueError,'I'):generate_text(folder,'I')
            Image.new('L',(80,80),255).save(folder/'Writer_02_upper-B.png')
            with self.assertRaisesRegex(ValueError,'Cannot prepare'):generate_text(folder,'B')

    def test_export_retains_actual_prediction_separate_from_request(self):
        with TemporaryDirectory() as tmp:
            folder=Path(tmp);samples(folder);image,origin=generate_text(folder,'00Aa')
            seg=segment_characters(image);arrays=prepare_character_crops(seg.crops)
            result={'segmentation':seg,'prepared':arrays,'prediction':{'text':'OOaa','characters':[],'model':'test'}}
            first=export_result(folder,image,origin,result,{'mode':'Alphanumeric','model':'test'})
            second=export_result(folder,image,origin,result,{'mode':'Alphanumeric','model':'test'})
            self.assertNotEqual(first,second)
            record=json.loads((first/'result.json').read_text())
            self.assertEqual(record['input']['requested_text'],'00Aa')
            self.assertEqual(record['prediction']['text'],'OOaa')
            self.assertTrue((first/'detected_characters.png').is_file())
            self.assertEqual(record['input_png_sha256'],hashlib.sha256((first/'input.png').read_bytes()).hexdigest())
            self.assertTrue(np.array_equal(np.load(first/'prepared_crops.npy'),arrays))

    def test_service_rejects_corrupt_character_checkpoint(self):
        with TemporaryDirectory() as tmp:
            folder=Path(tmp);model=folder/'experiments/emnist_byclass/run_01';model.mkdir(parents=True)
            (model/'cnn.pt').write_bytes(b'corrupt')
            (model/'selection.json').write_text(json.dumps({'models':[{'model':'cnn','checkpoint_sha256':'wrong'}]}))
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                RecognitionService(folder).recognise(Image.new('RGB',(20,20)),{'mode':'Alphanumeric','model':'cnn','method':'grouped_components'})


if __name__=='__main__':unittest.main()
