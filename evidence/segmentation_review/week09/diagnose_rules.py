"""Reproduce grouping measurements from the repository root; print JSON only."""
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from hnrs_chars.segmentation import foreground,group_components

with Path(__file__).with_name('diagnostics.csv').open(newline='') as stream:
    rows=list(csv.DictReader(stream))
records=[]
for row in rows:
    if int(row['index']) not in (122,125):continue
    with Image.open(row['file']) as image:
        _,_,regions,_,_=foreground(image)
    typical=float(np.percentile([r['box'][3]-r['box'][1] for r in regions],80))
    details=[]
    for region in regions:
        a,b,c,d=region['box'];width=c-a;height=d-b
        details.append(dict(id=region['id'],box=region['box'],width=width,height=height,area=region['area'],
            classified_body=height>=.4*typical or (height>=.25*typical and width<=.45*height)))
    groups,warnings=group_components(regions)
    records.append(dict(index=int(row['index']),file=row['file'],typical_height=typical,
        body_height_threshold=.4*typical,short_body_height_threshold=.25*typical,narrow_width_to_height_limit=.45,
        regions=details,groups=[[r['id'] for r in g] for g in groups],warnings=warnings))
expected=json.loads(Path(__file__).with_name('failure_rule_diagnostics.json').read_text())
assert records==expected,'Grouping diagnostics differ from the saved review'
print(json.dumps(records,indent=2))
