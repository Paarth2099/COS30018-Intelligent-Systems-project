"""Experimental preparation/segmentation; never imported by the production GUI.

No labels, expected counts or predictions enter these candidate algorithms.
"""
import numpy as np
from PIL import Image, ImageOps
from hnrs_chars.preprocessing import prepare_character
from segment_digits import segment_image, connected_regions, SegmentationResult


def padded_character(image):
    """Test whether a 15% pale margin stabilises local background estimation."""
    gray = ImageOps.exif_transpose(image).convert('L')
    paper = int(np.percentile(np.asarray(gray), 90))
    margin = max(3, round(.15 * max(gray.size)))
    return prepare_character(ImageOps.expand(gray, border=margin, fill=paper))


def mask_characters(segmented):
    """Reuse the selected foreground instead of re-estimating crop illumination.

    Excludes other character groups when their bounding rectangles overlap.
    Group membership is recovered by identical component boxes from the clean
    mask, since its new connected-component ids need not equal original ids.
    """
    labels,regions=connected_regions(segmented.mask)
    prepared=[]
    for a,b,c,d in segmented.boxes:
        # Components fully inside a group union may include a neighbour in
        # overlapping boxes. Use the existing crop's foreground would repeat
        # the contested threshold; therefore this candidate conservatively
        # refuses overlap and falls back to production preparation there.
        own=(a,b,c,d)
        overlaps=any(other!=own and max(a,other[0])<min(c,other[2]) and
            max(b,other[1])<min(d,other[3]) for other in segmented.boxes)
        if overlaps:
            prepared.append(prepare_character(segmented.crops[len(prepared)]));continue
        tile=segmented.mask[b:d,a:c].astype('uint8')*255
        scale=20/max(c-a,d-b);size=(max(1,round((c-a)*scale)),max(1,round((d-b)*scale)))
        ink=Image.fromarray(tile).resize(size,Image.Resampling.LANCZOS)
        canvas=Image.new('L',(28,28),0);canvas.paste(ink,((28-size[0])//2,(28-size[1])//2))
        array=np.asarray(canvas,dtype=np.float32)/255
        if array.max()>0:array/=array.max()
        prepared.append(array)
    return np.stack(prepared) if prepared else np.empty((0,28,28),np.float32)


def merge_fragments(boxes):
    """Conservative adjacent horizontal-overlap rule, excluding narrow pairs.

    Overlapping x ranges alone are insufficient: require a narrow/wide pair,
    small gap, complementary vertical support and a bounded union aspect ratio.
    Returns index groups so no surviving foreground is discarded.
    """
    groups = []; i = 0
    while i < len(boxes):
        if i + 1 < len(boxes):
            a,b,c,d = boxes[i]; x,y,z,w = boxes[i+1]
            height = max(d,w)-min(b,y)
            widths = sorted((c-a,z-x))
            gap = max(0,x-c)
            voverlap = max(0,min(d,w)-max(b,y))
            if (height > 0 and gap <= .04*height and
                widths[0] <= .25*height and widths[1] >= .35*height and
                max(c,z)-min(a,x) <= .95*height and
                voverlap >= .4*min(d-b,w-y) and
                min(d-b,w-y) >= .35*height):
                groups.append([i,i+1]); i += 2; continue
        groups.append([i]); i += 1
    return groups


def segment_repaired_digits(image):
    base = segment_image(image, 'components')
    groups = merge_fragments(base.boxes)
    if all(len(g)==1 for g in groups):
        return base
    labels, regions = connected_regions(base.mask)
    boxes=[]; crops=[]
    pixels=np.asarray(base.gray)
    background=pixels[~base.mask]
    paper=int(np.percentile(background,90)) if background.size else 255
    for group in groups:
        members=[base.boxes[i] for i in group]
        a=min(b[0] for b in members); b=min(b[1] for b in members)
        c=max(bx[2] for bx in members); d=max(bx[3] for bx in members)
        boxes.append((a,b,c,d))
        ids=[r['id'] for r in regions if tuple(r['box']) in members]
        tile=pixels[b:d,a:c].copy()
        region=labels[b:d,a:c]
        tile[(region>0)&~np.isin(region,ids)]=paper
        pad=max(2,round(.03*(d-b)))
        crops.append(ImageOps.expand(Image.fromarray(tile),pad,fill=paper))
    return SegmentationResult(base.gray,base.mask,boxes,crops,
        base.warnings+['Experimental fragment merge: inspect boundaries'],base.removed_components)
