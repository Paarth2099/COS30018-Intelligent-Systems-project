"""One-line separated-character segmentation; never uses reference text.

Reuses Manula's existing connected-region utility and Paarth's character
preparation. Projection and grouped components share the same foreground mask.
"""
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from segment_digits import connected_regions, occupied_intervals
from preprocess_digits import otsu_threshold
from .preprocessing import prepare_character

METHODS = ('projection', 'grouped_components')
DEFAULT_METHOD = 'grouped_components'


@dataclass
class CharacterSegmentation:
    gray: Image.Image
    mask: np.ndarray
    boxes: list
    crops: list
    warnings: list
    groups: list
    removed_components: int
    method: str


def foreground(image, max_side=1200):
    """Shared paper-lighting correction and conservative noise filtering.

    Lighting formula follows Paarth's preparation. No component-height filter:
    tiny vertical extent alone is not grounds for discarding a letter's dot.
    Coordinates belong to returned upright, resized grayscale image.
    """
    gray=ImageOps.exif_transpose(image).convert('L')
    gray.thumbnail((max_side,max_side),Image.Resampling.LANCZOS)
    pixels=np.asarray(gray)
    if int(pixels.max())-int(pixels.min())<10:
        return gray,np.zeros(pixels.shape,np.int32),[],0,['Blank or very low contrast image']
    small=gray.copy();small.thumbnail((256,256),Image.Resampling.LANCZOS)
    background=small.filter(ImageFilter.MaxFilter(31)).filter(ImageFilter.GaussianBlur(8))
    background=background.resize(gray.size,Image.Resampling.BILINEAR)
    corrected=np.clip(pixels.astype(np.float32)*255/np.maximum(np.asarray(background),1),0,255)
    adjusted=np.asarray(ImageOps.autocontrast(Image.fromarray(np.rint(corrected).astype('uint8'))))
    mask=adjusted<=otsu_threshold(adjusted)
    labels,regions=connected_regions(mask)
    # Close tiny sampling/pen gaps in the connectivity mask, not in returned
    # grayscale crops. Scale modestly with writing height and cap at 7 pixels.
    substantial=[r for r in regions if r['area']>=max((s['area'] for s in regions),default=0)*.05]
    typical=float(np.percentile([r['box'][3]-r['box'][1] for r in substantial],80)) if substantial else 0
    kernel=min(7,max(3,int(round(.04*typical))|1))
    joined=Image.fromarray(mask.astype('uint8')*255).filter(ImageFilter.MaxFilter(kernel)).filter(ImageFilter.MinFilter(kernel))
    labels,regions=connected_regions(np.asarray(joined)>0)
    cutoff=max(3,max((r['area'] for r in regions),default=0)*.005)
    kept=[r for r in regions if r['area']>=cutoff]
    clean=np.isin(labels,[r['id'] for r in kept]);warnings=[]
    if clean[0].any() or clean[-1].any() or clean[:,0].any() or clean[:,-1].any():
        warnings.append('Foreground touches image edge; inspect clipping or paper artifacts')
    if clean.mean()>.45:warnings.append('Large foreground fraction; inspect shadow or wrong polarity')
    return gray,labels,kept,len(regions)-len(kept),warnings


def union_box(regions):
    return (min(r['box'][0] for r in regions),min(r['box'][1] for r in regions),
            max(r['box'][2] for r in regions),max(r['box'][3] for r in regions))


def group_components(regions):
    """Attach small components above nearby bodies using geometry only.

    Bound the horizontal/vertical gaps relative to observed body height. Do not
    merge full-height neighbours or use the number of expected characters.
    Unassigned small regions remain visible as separate groups with a warning.
    """
    if not regions:return [],[]
    height=float(np.percentile([r['box'][3]-r['box'][1] for r in regions],80))
    bodies=[r for r in regions if r['box'][3]-r['box'][1]>=.4*height or
            (r['box'][3]-r['box'][1]>=.25*height and
             r['box'][2]-r['box'][0]<=.45*(r['box'][3]-r['box'][1]))]
    small=[r for r in regions if r not in bodies]
    groups=[[r] for r in bodies];warnings=[]
    # A rough pen dot can itself be fragmented. Join only close small pieces
    # whose combined height remains dot-sized before assigning them to a stem.
    clusters=[[r] for r in small]
    changed=True
    while changed:
        changed=False
        for i in range(len(clusters)):
            if changed:break
            for j in range(i+1,len(clusters)):
                a,b,c,d=union_box(clusters[i]);x0,y0,x1,y1=union_box(clusters[j])
                distance=np.hypot(max(0,a-x1,x0-c),max(0,b-y1,y0-d))
                if distance<=.12*height and max(d,y1)-min(b,y0)<.4*height:
                    clusters[i].extend(clusters.pop(j));changed=True;break
    for parts in clusters:
        x0,y0,x1,y1=union_box(parts);area=sum(r['area'] for r in parts);candidates=[]
        for i,body in enumerate(bodies):
            a,b,c,d=body['box'];gap=max(0,a-x1,x0-c)
            vertical=b-y1
            distance=float(np.hypot(gap,max(0,b-y1,y0-d)))
            near_fragment=distance<=.06*height and area<=.3*body['area']
            upper_dot=(y0<b and -.08*height<=vertical<=.65*height and
                       gap<=.25*height and area<=body['area'])
            if near_fragment or upper_dot:
                score=distance+.15*abs((x0+x1-a-c)/2)
                candidates.append((score,i))
        if candidates:
            candidates.sort();groups[candidates[0][1]].extend(parts)
            if len(candidates)>1 and candidates[1][0]-candidates[0][0]<.05*height:
                warnings.append('Ambiguous detached component assignment; inspect dot grouping')
        else:
            groups.append(parts);warnings.append('Unassigned small component retained; inspect noise or broken character')
    return groups,warnings


def segment_characters(image, method=DEFAULT_METHOD, max_side=1200):
    if method not in METHODS:raise ValueError('Unknown character segmentation method')
    if max_side<28:raise ValueError('max_side must be at least 28')
    gray,labels,regions,removed,warnings=foreground(image,max_side)
    clean=np.isin(labels,[r['id'] for r in regions])
    if not regions:
        return CharacterSegmentation(gray,clean,[],[],warnings or ['No ink found'],[],removed,method)
    if method=='projection':
        groups=[]
        for left,right in occupied_intervals(clean.any(axis=0)):
            ids=set(np.unique(labels[:,left:right][clean[:,left:right]]).tolist())
            groups.append([r for r in regions if r['id'] in ids])
    else:
        groups,extra=group_components(regions);warnings.extend(extra)
    groups.sort(key=lambda group:(union_box(group)[0],union_box(group)[1]))
    boxes=[union_box(group) for group in groups];crops=[]
    typical=float(np.median([b[3]-b[1] for b in boxes]))
    if any(b[2]-b[0]>1.6*typical for b in boxes):
        warnings.append('Wide region: touching/overlapping characters may be merged')
    for box,group in zip(boxes,groups):
        left,top,right,bottom=box;pad=max(2,round(.04*(bottom-top)))
        # Synthetic pale margin also protects ink when original photo is clipped.
        tile=np.asarray(gray.crop(box)).copy()
        region_labels=labels[top:bottom,left:right]
        ids=[r['id'] for r in group]
        background=np.asarray(gray)[labels==0]
        paper=int(np.percentile(background,90)) if len(background) else 255
        tile[(region_labels>0)&~np.isin(region_labels,ids)]=paper
        crop=Image.new('L',(right-left+2*pad,bottom-top+2*pad),paper)
        crop.paste(Image.fromarray(tile),(pad,pad));crops.append(crop)
    return CharacterSegmentation(gray,clean,boxes,crops,warnings,
                                 [[r['id'] for r in g] for g in groups],removed,method)


def prepare_character_crops(crops):
    """Paarth's selected preparation, preserving reading order and empty batches."""
    return np.stack([prepare_character(crop) for crop in crops]) if crops else np.empty((0,28,28),np.float32)


def recognise_text(image,predictor,method=DEFAULT_METHOD):
    """Integration API: caller loads a character predictor once; no true label."""
    segmented=segment_characters(image,method)
    prepared=prepare_character_crops(segmented.crops)
    return {'segmentation':segmented,'prepared':prepared,'prediction':predictor.predict_text(prepared)}
