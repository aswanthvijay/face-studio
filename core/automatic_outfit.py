"""Bounded automatic outfit repair, isolated from the portrait pipeline.

Two total candidates (also for variation requests), one background repair and
one localized defect repair per candidate. Diagnostics are never served as
downloadable successful results.
"""
import json
import secrets
import uuid
from PIL import Image
import numpy as np
from core import config
from core.outfit_regions import build_source_regions
from core.outfit_validation import (validate_candidate, revealed_background,
                                   background_line_check, correct_new_skin)


class OutfitRejected(Exception):
    def __init__(self, quality):
        self.quality = quality
        super().__init__('; '.join(quality['failure_reasons']))


def rejection_response(error, source_url):
    return {'success': False, 'status': 'rejected', 'source_url': source_url,
            'error': str(error), 'failure_reasons': error.quality['failure_reasons'],
            'quality': error.quality}


def render_stage(image, mask, prompt, negative, model_cfg, seed, directory,
                 name, progress_cb=None, lora_name=None, lora_weight=.8):
    from core import pipeline
    pipeline.ensure_backend()
    image_path, mask_path = directory / (name+'_input.png'), directory / (name+'_mask.png')
    image.save(image_path)
    Image.fromarray(mask.astype('uint8')*255).save(mask_path)
    w,h = pipeline.calculate_working_dimensions(*image.size,
        target_size=model_cfg.get('target_size',1024), is_in_place=True)
    graph,_ = pipeline.build_sdxl_inpaint_graph(
        pipeline.upload_image(image_path), pipeline.upload_image(mask_path),
        prompt, negative, model_cfg, w,h,seed,lora_name,lora_weight)
    (directory / (name+'_graph.json')).write_text(json.dumps(graph,indent=2))
    stage_progress = (lambda message: progress_cb(f'{name.replace("_", " ").title()}: {message}')) if progress_cb else None
    raw = pipeline.execute_workflow(graph,progress_cb=stage_progress)
    raw.save(directory / (name+'_raw.png'))
    return raw.resize(image.size,Image.Resampling.LANCZOS)


def repair_crop(image, mask, prompt, negative, cfg, seed, directory, name, **kwargs):
    ys,xs = np.where(mask)
    if not len(xs):
        return image
    pad = max(48, round(max(image.size)*.08))
    box = (max(0,int(xs.min())-pad),max(0,int(ys.min())-pad),
           min(image.width,int(xs.max())+pad+1),min(image.height,int(ys.max())+pad+1))
    x,y,x2,y2 = box
    local = mask[y:y2,x:x2]
    crop = image.crop(box)
    rendered = render_stage(crop,local,prompt,negative,cfg,seed,directory,name,**kwargs)
    result = image.copy()
    result.paste(Image.composite(rendered,crop,Image.fromarray(local.astype('uint8')*255)),(x,y))
    result.save(directory/(name+'_composite.png'))
    return result


def run_automatic_outfit(source, prompt, model_cfg, seed=None, variations=1,
                         progress_cb=None, lora_name=None, lora_weight=.8,
                         reference_cores=None, target='dress', diagnostic_only=False,
                         negative_prompt=""):
    from core.clothing_engine import parse_clothing_labels
    request_id = uuid.uuid4().hex[:12]
    directory = config.WORKSPACE_ROOT/'diagnostics'/'outfit'/request_id
    directory.mkdir(parents=True)
    source.save(directory/'source.png')
    seed = int(seed) if seed is not None else secrets.randbelow(2147483646)
    if seed < 0 or seed >= 2**64-1:
        raise ValueError('Seed must allow a second candidate at seed + 1.')
    requested = max(1,min(int(variations),4))
    journal = {'request_id':request_id,'requested_seed':seed,'attempted_seeds':[],
               'validation_status':'pending','failure_reasons':[], 'candidates':[],
               'requested_variations':requested, 'generated_variations':0,
               'diagnostic_only':bool(diagnostic_only)}

    def save():
        (directory/'quality.json').write_text(json.dumps(journal,indent=2),encoding='utf-8')

    def progress(message):
        journal['stage'] = message
        save()
        if progress_cb:
            progress_cb(message)

    try:
        regions = build_source_regions(source,prompt,progress,target)
    except Exception as exc:
        journal.update(validation_status='rejected', failure_reasons=[
            'Source protection could not be established: ' + str(exc)])
        save()
        raise OutfitRejected(journal) from exc
    regions.save(directory/'regions')
    journal['source_decisions'] = regions.decisions
    if reference_cores:
        # Qualification annotations are an independent oracle, never inputs to
        # source map construction. A missed core cannot be hidden by compositing.
        journal['reference_protection_checks'] = {
            name: {'pixels':int(mask.sum()), 'editable_core_pixels':int((mask & regions.approved).sum())}
            for name,mask in reference_cores.items()}
        for name,check in journal['reference_protection_checks'].items():
            if check['editable_core_pixels']:
                regions.reasons.append('Independent reference protection failed: ' + name)
    if model_cfg.get('type','sdxl') != 'sdxl':
        regions.reasons.append('Automatic outfit repair currently requires CyberRealistic XL (SDXL).')
    if regions.reasons and not diagnostic_only:
        journal.update(validation_status='rejected',failure_reasons=regions.reasons)
        progress('Edit rejected: source protection could not be established safely.')
        raise OutfitRejected(journal)
    positive = prompt + ', photograph, same person, same pose, preserve all original accessories'
    negative = config.DEFAULT_NEGATIVE_PROMPT
    if negative_prompt and str(negative_prompt).strip():
        negative = f"{str(negative_prompt).strip()}, {negative}"
    if not regions.intent['neckline_change']:
        positive += ', preserve the exact source neckline and collar coverage'
        negative += ', one shoulder, asymmetric neckline, halter, strapless, plunging neckline'
    if not regions.intent['bare_arms'] and not regions.intent['long_sleeves']:
        positive += ', preserve the source sleeve length and shoulder coverage'
        negative += ', bare shoulder, sleeveless'
    kwargs = dict(progress_cb=progress_cb,lora_name=lora_name,lora_weight=lora_weight)
    accepted = []
    accepted_quality = []
    for candidate_seed in (seed,seed+1):
        if len(accepted) >= requested:
            break
        journal['attempted_seeds'].append(candidate_seed)
        stage_dir = directory/str(candidate_seed)
        stage_dir.mkdir()
        item = {'seed':candidate_seed,'repairs':{},'stage_outputs':[]}
        journal['candidates'].append(item)
        try:
            progress(f'Generating candidate {len(journal["attempted_seeds"])}/2...')
            raw = render_stage(source,regions.synthesis,positive,negative,model_cfg,
                               candidate_seed,stage_dir,'generation',**kwargs)
            candidate = Image.composite(raw,source,Image.fromarray(regions.alpha))
            candidate.save(stage_dir/'initial_composite.png')
            labels = parse_clothing_labels(candidate)
            revealed = revealed_background(regions,labels)
            repair_info = {'attempted':False,'accepted':False,'pixels':int(revealed.sum())}
            item['repairs']['background'] = repair_info
            if revealed.sum() > 20:
                progress('Reconstructing newly revealed background in a locked crop...')
                repair_info['attempted'] = True
                before = background_line_check(source,candidate,regions,revealed)
                patched = repair_crop(candidate,revealed,
                    'Continue the surrounding background exactly, continuous straight scene lines, matching floor perspective, same lighting, photograph',
                    'person, clothing, skin, hands, bag, objects, text, distorted lines',
                    model_cfg,candidate_seed,stage_dir,'background',**kwargs)
                after_labels = parse_clothing_labels(patched)
                after = background_line_check(source,patched,regions,revealed)
                keeps_background = float((after_labels[revealed] == 0).mean()) >= .95
                repair_info.update(before=before,after=after)
                if keeps_background and after['passed'] and (before['passed'] or after['matched_fraction'] > before['matched_fraction']):
                    candidate,labels = patched,after_labels
                    repair_info['accepted'] = True
                else:
                    repair_info['reason'] = 'Repair failed background continuity or introduced a subject/object.'
            candidate,skin_info = correct_new_skin(source,candidate,regions,labels)
            item['repairs']['skin'] = skin_info
            candidate.save(stage_dir/'skin_composite.png')
            labels = parse_clothing_labels(candidate)
            quality,defects = validate_candidate(source,candidate,regions,labels,reference_cores)
            local_info = {'attempted':False,'accepted':False,'pixels':int(defects.sum())}
            item['repairs']['localized'] = local_info
            if 0 < defects.sum() <= regions.approved.size*.05:
                progress('Repairing a detected garment defect within the approved region...')
                local_info['attempted'] = True
                patched = repair_crop(candidate,defects,
                    positive+', continuous matching garment fabric and natural seams',
                    negative,model_cfg,candidate_seed,stage_dir,'localized',**kwargs)
                check,new_defects = validate_candidate(source,patched,regions,parse_clothing_labels(patched),reference_cores)
                if new_defects.sum() < defects.sum()*.5 and len(check['failure_reasons']) <= len(quality['failure_reasons']):
                    candidate,quality = patched,check
                    local_info['accepted'] = True
                else:
                    local_info['reason'] = 'Localized repair did not sufficiently reduce defects.'
            progress('Checking preservation, coverage, skin seams and scene continuity...')
            quality,_ = validate_candidate(source,candidate,regions,parse_clothing_labels(candidate),reference_cores)
            # Informational measurements never override the preservation gates.
            from core.prompt_engine import garment_requirements
            from core.quality_checks import garment_color_check
            color = garment_requirements(prompt).get('color')
            if color:
                try:
                    quality['garment_color'] = garment_color_check(candidate,parse_clothing_labels(candidate),color)
                except Exception as exc:
                    quality['garment_color'] = {'requested':color,'matches':None,'reason':str(exc)}
            quality['identity_similarity'] = None
            quality['identity_measurement'] = 'not_computed; original face pixels checked exactly'
            candidate.save(stage_dir/'final_diagnostic.png')
            item['quality'] = quality
            if quality['validation_status'] == 'passed' and not diagnostic_only:
                filename = f'out_{request_id}_{candidate_seed}_automatic.png'
                candidate.save(config.OUTPUTS_DIR/filename)
                accepted.append(filename)
                accepted_quality.append(quality)
        except Exception as exc:
            item['quality'] = {'validation_status':'rejected','failure_reasons':[f'Candidate stage failed: {exc}']}
        finally:
            item['stage_outputs'] = sorted(p.name for p in stage_dir.iterdir() if p.is_file())
            save()
    journal['generated_variations'] = len(accepted)
    if not accepted:
        reasons = [r for item in journal['candidates'] for r in item['quality']['failure_reasons']]
        if diagnostic_only:
            reasons.append('Development diagnostic run: no candidates are published, regardless of checks.')
        journal.update(validation_status='rejected',failure_reasons=list(dict.fromkeys(reasons)))
        progress('Both candidates rejected. Original source retained.')
        raise OutfitRejected(journal)
    journal.update(validation_status='passed',preservation_checks=accepted_quality[0]['preservation_checks'])
    progress('Accepted outfit passed automatic checks.')
    warnings = []
    if len(accepted) < requested:
        warnings.append(f'Only {len(accepted)} of {requested} variations passed within the two-candidate limit.')
    return {'filename':accepted[0], 'output_path':str(config.OUTPUTS_DIR/accepted[0]),
            'model_name':model_cfg.get('name','CyberRealistic XL'),
            'results':['/outputs/'+n for n in accepted], 'requested_variations':requested,
            'generated_variations':len(accepted), 'warnings':warnings,
            'quality':journal,'quality_results':accepted_quality,
            'coverage':float(regions.approved.mean()),'face_swapped':False}
