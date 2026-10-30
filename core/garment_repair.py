"""Bounded repair of invented arm/hand skin for explicit long-sleeve swaps."""
import cv2
import numpy as np
from PIL import Image

from core.inpaint_regions import composite_alpha


def unexpected_arm_skin(source_labels, generated_labels, allowed):
    spill = (np.isin(generated_labels, (14,15)) &
             ~np.isin(source_labels, (14,15)) & allowed)
    count, components, stats, _ = cv2.connectedComponentsWithStats(spill.astype('uint8'))
    minimum = max(20, round(spill.size * 0.00009))
    return np.isin(components, [i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA] >= minimum])


def repair_arm_spill(image, source, edit_mask, model_cfg, seed, request_id,
                     progress_cb=None, lora_name=None, lora_weight=0.8, garment_color=None):
    """Attempt once; accept only reduced spill with unchanged protected pixels.

    The parser is an estimate. This check does not certify anatomy or general
    image quality, and it is only appropriate when bare arms were not requested.
    """
    from core import config, pipeline
    from core.clothing_engine import parse_clothing_labels
    source_labels = parse_clothing_labels(source)
    labels = parse_clothing_labels(image)
    allowed = np.asarray(edit_mask) > 0
    spill = unexpected_arm_skin(source_labels, labels, allowed)
    before = int(spill.sum())
    info = {'attempted':False, 'accepted':False, 'unexpected_arm_skin_before':before,
            'unexpected_arm_skin_after':before}
    if before == 0:
        return image, info
    if before > spill.size*0.05:
        info['reason'] = 'Large anatomy disagreement requires review'
        return image, info
    protected = ~allowed | np.isin(source_labels, (14,15))
    radius = max(3, round(max(image.size)*7/1785))
    alpha = composite_alpha(spill, protected, radius=radius)
    ys, xs = np.where(alpha > 0)
    padding = max(48, round(max(image.size)*90/1785))
    box = (max(0,int(xs.min())-padding), max(0,int(ys.min())-padding),
           min(image.width,int(xs.max())+padding+1), min(image.height,int(ys.max())+padding+1))
    crop = image.crop(box)
    crop_alpha = Image.fromarray(alpha).crop(box)
    source_path = config.INPUTS_DIR/f'repair_{request_id}.png'
    mask_path = config.INPUTS_DIR/f'repair_mask_{request_id}.png'
    crop.save(source_path)
    Image.fromarray(np.where(np.asarray(crop_alpha)>0,255,0).astype('uint8')).save(mask_path)
    width,height = pipeline.calculate_working_dimensions(*crop.size, is_in_place=True)
    graph,_ = pipeline.build_sdxl_inpaint_graph(
        pipeline.upload_image(source_path), pipeline.upload_image(mask_path),
        f'{garment_color or "matching"} dress fabric, continuous opaque fabric, same texture as surrounding garment, natural fabric folds, photograph',
        'skin, hands, fingers, limbs, hair, jewelry, holes, blurry', model_cfg,
        width, height, seed, lora_name, lora_weight)
    info['attempted'] = True
    if progress_cb:
        progress_cb('Refining garment beside preserved hands...')
    raw = pipeline.execute_workflow(graph, progress_cb=progress_cb)
    patch = Image.composite(raw.resize(crop.size,Image.Resampling.LANCZOS),crop,crop_alpha)
    candidate = image.copy()
    candidate.paste(patch,box[:2])
    after = int(unexpected_arm_skin(source_labels,parse_clothing_labels(candidate),allowed).sum())
    if after < before*0.5:
        info.update(accepted=True, unexpected_arm_skin_after=after)
        return candidate, info
    info['reason'] = 'Repair did not sufficiently reduce the detected anatomy disagreement'
    return image, info
