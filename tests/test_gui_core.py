"""Tests for GUI acquisition/export without opening a desktop window."""
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import unittest
import numpy as np
from PIL import Image,ImageDraw
from gui_core import digit_catalog,generate_number,open_photo,export_result
from segment_digits import segment_image,prepare_crops


def samples(folder):
    for digit in ['0','7']:
        image=Image.new('L',(100,120),255);d=ImageDraw.Draw(image)
        if digit=='0':d.ellipse((25,15,75,105),outline=0,width=7)
        else:d.line([(20,20),(80,20),(35,105)],fill=0,width=7)
        image.save(folder/f'Test_01_label-{digit}.png')
    Image.new('L',(100,100),255).save(folder/'Test_label-123.png')


class GuiCoreTests(unittest.TestCase):
    def test_generation_leading_zeros_and_no_source_changes(self):
        with TemporaryDirectory() as tmp:
            folder=Path(tmp);samples(folder)
            before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
            image,origin=generate_number(folder,'007')
            self.assertEqual(origin['requested_text'],'007')
            self.assertEqual([s['digit'] for s in origin['sources']],['0','0','7'])
            self.assertEqual(image.tobytes(),generate_number(folder,'007')[0].tobytes())
            self.assertEqual(len(segment_image(image).boxes),3)
            self.assertEqual(before,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()})

    def test_catalog_excludes_multi_digit_and_supports_nested_folders(self):
        with TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'member').mkdir();samples(folder/'member')
            index=digit_catalog(folder)
            self.assertEqual(sum(map(len,index.values())),2)

    def test_invalid_or_missing_requested_digits(self):
        with TemporaryDirectory() as tmp:
            folder=Path(tmp);samples(folder)
            for value in ['', 'abc','12.3','-1','0'*13]:
                with self.assertRaises(ValueError):generate_number(folder,value)
            with self.assertRaisesRegex(ValueError,'No single-digit image for: 9'):
                generate_number(folder,'09')

    def test_bad_source_is_reported(self):
        with TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'7.png').write_text('not an image')
            with self.assertRaisesRegex(ValueError,'Cannot prepare'):
                generate_number(folder,'7')

    def test_export_records_provenance_and_keeps_previous_exports(self):
        with TemporaryDirectory() as tmp:
            folder=Path(tmp);samples(folder)
            image,origin=generate_number(folder,'007');seg=segment_image(image)
            result={'segmentation':seg,'prepared':prepare_crops(seg.crops),
                    'prediction':{'text':'007','model':'test','digits':[]}}
            first=export_result(folder,image,origin,result,{'model':'test'})
            second=export_result(folder,image,origin,result,{'model':'test'})
            self.assertNotEqual(first,second)
            data=json.loads((first/'result.json').read_text())
            self.assertEqual(data['prediction']['text'],'007')
            self.assertEqual(data['input']['requested_text'],'007')
            self.assertEqual(np.load(first/'prepared_crops.npy').shape,(3,28,28))
            self.assertEqual(open_photo(first/'input.png').mode,'RGB')


if __name__=='__main__':unittest.main()
