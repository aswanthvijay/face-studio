"""Conservative automatic checks; independent fixture cores remain separate."""
import cv2
import numpy as np
from PIL import Image
from core.outfit_regions import GARMENT, SKIN, core, dilate


def preservation_checks(source, candidate, regions, reference_cores=None):
    a, b = np.asarray(source), np.asarray(candidate)
    changed = np.any(a != b, axis=2)
    masks = {**regions.regions, 'outside_approved': ~regions.approved}
    if reference_cores:
        masks.update({'reference_' + name: mask for name, mask in reference_cores.items()})
    return {name: {'pixels': int(mask.sum()), 'changed': int((changed & mask).sum()),
                   'passed': not bool((changed & mask).any())} for name, mask in masks.items()}


def revealed_background(regions, labels):
    return (labels == 0) & regions.garment & regions.approved


def fabric_remnants(source, candidate, regions):
    """Source-colored pixels beside protected edges, not a target color score."""
    a = cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2LAB).astype(float)
    b = cv2.cvtColor(np.asarray(candidate), cv2.COLOR_RGB2LAB).astype(float)
    cloth = core(regions.garment & ~regions.protected, 4)
    if cloth.sum() < 20:
        return np.zeros(regions.labels.shape, bool)
    old = np.median(a[cloth], axis=0)
    new = np.median(b[cloth], axis=0)
    if np.linalg.norm(old-new) < 24:
        return np.zeros(regions.labels.shape, bool)
    near = dilate(regions.regions['hair'] | regions.regions['objects'] | regions.regions['bag'], 10)
    old_colored = np.linalg.norm(b-old, axis=2) < 22
    # Include protected uncertainty: unchanged old fabric is still a failure.
    return near & dilate(regions.garment, 2) & old_colored & (np.linalg.norm(b-new,axis=2) > 28)


def background_line_check(source, candidate, regions, revealed):
    """Check continuation of strong source scene lines across revealed clothing.

    This does not certify all floor geometry, but catches interrupted shutter
    lines independently of the generation mask.
    """
    gray = cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    edges[~core(regions.labels == 0, 3)] = 0
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 45, minLineLength=65, maxLineGap=8)
    expected = np.zeros_like(edges)
    if lines is not None:
        # OpenCV 4 returns (N,1,4); OpenCV 5 may return (N,4).
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1,4):
            dx, dy = x2-x1, y2-y1
            norm = max(1, np.hypot(dx,dy))
            extension = min(200, max(source.size)*.20) / norm
            p1 = (round(x1-dx*extension), round(y1-dy*extension))
            p2 = (round(x2+dx*extension), round(y2+dy*extension))
            cv2.line(expected, p1, p2, 255, 1)
    expected = (expected > 0) & core(revealed, 2)
    actual = cv2.Canny(cv2.cvtColor(np.asarray(candidate), cv2.COLOR_RGB2GRAY), 40, 110)
    matched = expected & dilate(actual > 0, 4)
    fraction = float(matched.sum()/max(1, expected.sum()))
    return {'tested_pixels': int(expected.sum()), 'matched_fraction': fraction,
            'passed': bool(expected.sum() < 60 or fraction >= .50)}


def skin_discontinuity(source, candidate, regions, labels):
    new = np.isin(labels, SKIN) & regions.approved & ~np.isin(regions.labels, SKIN)
    original = np.isin(regions.labels, SKIN) & regions.protected
    seam = new & dilate(original, 12)
    if seam.sum() < 15:
        return {'tested_pixels': int(seam.sum()), 'passed': True, 'delta_lab': 0.0}
    a = cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2LAB).astype('float32')
    b = cv2.cvtColor(np.asarray(candidate), cv2.COLOR_RGB2LAB).astype('float32')
    reference = original & dilate(seam, 18)
    delta = np.abs(np.median(b[seam],axis=0)-np.median(a[reference],axis=0))
    return {'tested_pixels': int(seam.sum()), 'passed': bool(delta[0] <= 18 and max(delta[1:]) <= 10),
            'delta_lab': [float(x) for x in delta]}


def correct_new_skin(source, candidate, regions, labels):
    """One bounded spatial correction; original visible skin is never written."""
    new = np.isin(labels, SKIN) & regions.approved & ~np.isin(regions.labels, SKIN)
    original = np.isin(regions.labels, SKIN) & regions.protected
    info = {'attempted': False, 'max_lab_correction': [8, 4, 4]}
    seam = new & dilate(original, 12)
    if seam.sum() < 15:
        info['reason'] = 'No adjacent source skin reference'
        return candidate, info
    a = cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2LAB).astype('float32')
    b = cv2.cvtColor(np.asarray(candidate), cv2.COLOR_RGB2LAB).astype('float32')
    reference = original & dilate(seam, 18)
    delta = np.clip(np.median(a[reference],axis=0)-np.median(b[seam],axis=0), [-8,-4,-4], [8,4,4])
    inside = cv2.distanceTransform(new.astype('uint8'), cv2.DIST_L2, 5)
    near_reference = cv2.distanceTransform((~original).astype('uint8'), cv2.DIST_L2, 5)
    weight = np.minimum(inside/8,1) * np.maximum(0,1-near_reference/96)
    weight = weight*weight*(3-2*weight)
    corrected = cv2.cvtColor(np.clip(b+weight[:,:,None]*delta,0,255).astype('uint8'), cv2.COLOR_LAB2RGB)
    result = np.asarray(candidate).copy()
    corrected_region = new & (weight > 0)
    result[corrected_region] = corrected[corrected_region]
    info.update(attempted=True, delta=[float(x) for x in delta])
    return Image.fromarray(result), info


def validate_candidate(source, candidate, regions, labels, reference_cores=None):
    reasons = list(regions.reasons)
    preserve = preservation_checks(source, candidate, regions, reference_cores)
    for name, check in preserve.items():
        if not check['passed']:
            reasons.append('Protected pixels changed: ' + name)
    new_skin = np.isin(labels, SKIN) & regions.approved & ~np.isin(regions.labels, SKIN)
    unwanted = np.zeros(labels.shape, bool)
    if not regions.intent['bare_arms']:
        unwanted |= new_skin & np.isin(labels, (14,15))
    if not regions.intent['neckline_change']:
        unwanted |= new_skin & (labels == 11)
    if not regions.intent['short_hem']:
        unwanted |= new_skin & np.isin(labels, (12,13))
    if unwanted.sum() > max(30, regions.approved.sum()*.001):
        reasons.append('The candidate exposes skin beyond the requested sleeve, neckline or hem change.')
    remnants = fabric_remnants(source, candidate, regions)
    if remnants.sum() > 35:
        reasons.append('Visible source-colored fabric remains beside protected hair or objects.')
    bg = background_line_check(source, candidate, regions, revealed_background(regions,labels))
    if not bg['passed']:
        reasons.append('Newly revealed background breaks strong scene lines (shutter or floor).')
    skin = skin_discontinuity(source, candidate, regions, labels)
    if not skin['passed']:
        reasons.append('New skin has an unresolved seam against original skin.')
    changed = np.any(np.asarray(source) != np.asarray(candidate),axis=2) & regions.approved
    if changed.sum() < max(20, regions.approved.sum()*.20):
        reasons.append('The candidate did not replace enough of the garment.')
    return {'validation_status': 'rejected' if reasons else 'passed',
            'failure_reasons': list(dict.fromkeys(reasons)), 'preservation_checks': preserve,
            'background_lines': bg, 'skin_seam': skin,
            'source_fabric_remnant_pixels': int(remnants.sum()),
            'unexpected_skin_pixels': int(unwanted.sum())}, (unwanted | remnants) & regions.approved
