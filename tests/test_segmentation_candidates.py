import unittest
from hnrs_chars.segmentation_candidates import repair_broken_bodies,group_components


def region(i,box,area=100):return dict(id=i,box=box,area=area)


class SegmentationCandidateTests(unittest.TestCase):
    def test_complementary_broken_body_keeps_component_ids(self):
        groups=[[region(1,(10,10,70,110))],[region(2,(72,10,87,110))]]
        result,n=repair_broken_bodies(groups)
        self.assertEqual(n,1);self.assertEqual([[r['id'] for r in g] for g in result],[[1,2]])

    def test_close_narrow_letters_stay_separate(self):
        groups=[[region(1,(10,10,25,110))],[region(2,(27,10,42,110))]]
        self.assertEqual(repair_broken_bodies(groups),(groups,0))

    def test_wide_pair_or_distant_pair_not_joined(self):
        for box in [(72,10,130,110),(80,10,95,110)]:
            groups=[[region(1,(10,10,70,110))],[region(2,box)]]
            self.assertEqual(repair_broken_bodies(groups),(groups,0))

    def test_short_stem_eligibility_retains_dot(self):
        regions=[region(1,(10,10,50,210),5000),region(2,(100,120,128,179),600),region(3,(108,95,118,105),80),region(4,(160,10,170,210),1800)]
        groups,_=group_components(regions,stem_ratio=.55)
        self.assertTrue(any({r['id'] for r in g}=={2,3} for g in groups))
        self.assertEqual(sorted(r['id'] for g in groups for r in g),[1,2,3,4])
