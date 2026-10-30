"""Offline contracts for automatic outfit repair; no model download or GPU."""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from core import config, automatic_outfit as engine, outfit_regions as mapping
from core import outfit_validation as validation


class Fixture(unittest.TestCase):
    def setUp(self):
        self.labels = np.zeros((128,96),np.uint8)
        self.labels[40:110,30:70] = 7
        self.labels[10:35,40:60] = 11
        self.labels[38:100,20:29] = 14
        self.labels[20:75,30:35] = 2
        self.labels[70:95,68:78] = 16
        self.rgb = np.full((128,96,3),150,np.uint8)
        self.rgb[self.labels == 7] = [60,80,190]
        self.source = Image.fromarray(self.rgb)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.enterContext(patch.object(config,'OUTPUTS_DIR',self.root))
        self.enterContext(patch.object(config,'WORKSPACE_ROOT',self.root))
        hand_dir = self.root/'models/hands'
        hand_dir.mkdir(parents=True)
        for n in ('palm_detection_mediapipe_2023feb.onnx','handpose_estimation_mediapipe_2023feb.onnx'):
            (hand_dir/n).touch()
        self.parse = self.enterContext(patch('core.clothing_engine.parse_clothing_labels',return_value=self.labels))
        self.enterContext(patch('core.clothing_engine.detect_face_landmarks',return_value=[{'box':(40,10,20,25)}]))
        self.enterContext(patch.object(mapping,'refine_hair_boundary',return_value=(self.labels==2,np.zeros_like(self.labels,bool))))
        hand = np.zeros_like(self.labels,bool)
        hand[89:100,20:29] = True
        self.hand = self.enterContext(patch('core.hand_regions.hand_protection',return_value=(hand,[{'points':np.array([[22,90],[28,96],[20,98]])}])))
        self.proposals = self.enterContext(patch.object(mapping,'clip_proposals',side_effect=lambda image,q:{k:np.zeros_like(self.labels,float) for k in q}))
        self.text = self.enterContext(patch('core.text_regions.detect_text',return_value=[]))

    def regions(self,prompt='mini black dress'):
        return mapping.build_source_regions(self.source,prompt)


class RegionTests(Fixture):
    def test_every_swap_detects_hands_and_bag_is_protected(self):
        for prompt in ('mini black dress','sleeveless black dress','long sleeve dress'):
            r = self.regions(prompt)
            self.assertFalse(r.approved[self.labels==16].any())
            self.assertFalse(r.synthesis[r.protected].any())
            self.assertFalse(r.alpha[r.protected].any())
        self.assertEqual(self.hand.call_count,3)

    def test_no_upper_arm_exemption(self):
        r = self.regions()
        self.assertTrue(r.protected[self.labels==14].all())
        self.assertFalse(r.approved[self.labels==14].any())

    def test_explicit_long_sleeves_allow_coverage_but_lock_hands(self):
        r = self.regions('long sleeve black dress')
        self.assertTrue(r.coverage[50,23])
        self.assertFalse(r.approved[93,23])

    def test_crossed_hand_over_garment_wins(self):
        mask = np.zeros_like(self.labels,bool)
        mask[60:70,40:60] = True
        self.hand.return_value = mask,[{'points':np.array([[40,60],[60,60],[50,70]])}]
        r = self.regions()
        self.assertFalse(r.approved[mask].any())

    def test_phone_proposal_requires_hand_context(self):
        def proposals(image, queries):
            values = {k:np.zeros_like(self.labels,float) for k in queries}
            values['phone'][90:100,28:36] = 1
            values['phone'][0:5,85:90] = 1
            return values
        self.proposals.side_effect = proposals
        r = self.regions()
        self.assertTrue(r.regions['objects'][95,30])
        self.assertFalse(r.regions['objects'][1,87])

    def test_overlapping_text_rejects_instead_of_inventing_letters(self):
        self.text.return_value = [{'points':np.array([[40,90],[60,90],[60,100],[40,100]])}]
        r = self.regions()
        self.assertTrue(any('text overlaps' in reason for reason in r.reasons))
        self.assertFalse(r.approved[r.regions['graphics']].any())

    def test_missing_text_or_object_detector_is_not_silent(self):
        self.text.side_effect = RuntimeError('missing text model')
        self.proposals.side_effect = RuntimeError('missing CLIPSeg')
        r = self.regions()
        self.assertEqual(len(r.reasons),2)

    def test_missing_hand_detector_is_rejected(self):
        (self.root/'models/hands/palm_detection_mediapipe_2023feb.onnx').unlink()
        self.assertTrue(any('Hand protection models' in x for x in self.regions().reasons))

    def test_unlocalized_hands_protect_entire_arms_and_reject(self):
        self.hand.return_value = np.zeros_like(self.labels,bool),[]
        r = self.regions('long sleeve dress')
        self.assertTrue(r.protected[self.labels==14].all())
        self.assertTrue(any('Hands could not' in x for x in r.reasons))

    def test_mini_does_not_request_bare_shoulders(self):
        intent = mapping.coverage_intent('change to mini black dress')
        self.assertTrue(intent['short_hem'])
        self.assertFalse(intent['bare_arms'])
        self.assertFalse(intent['neckline_change'])

    def test_sleeveless_does_not_implicitly_change_neckline(self):
        intent = mapping.coverage_intent('sleeveless mini black dress')
        self.assertTrue(intent['bare_arms'])
        self.assertFalse(intent['neckline_change'])

    def test_landmarks_protect_crossed_hand_mislabeled_as_clothing(self):
        self.hand.return_value = np.zeros_like(self.labels,bool),[{'points':np.array([[40,60],[60,60],[50,75]])}]
        r = self.regions()
        self.assertFalse(r.approved[65,50])


class ValidationTests(Fixture):
    def test_real_hough_lines_handle_opencv_output_and_broken_background(self):
        import cv2
        from types import SimpleNamespace
        rgb = np.full((200,300,3),190,np.uint8)
        for y in (35,70,105,140,175):
            cv2.line(rgb,(0,y),(299,y),(30,30,30),2)
        labels = np.zeros((200,300),np.uint8)
        labels[:,120:180] = 7
        revealed = labels == 7
        r = SimpleNamespace(labels=labels)
        source = Image.fromarray(rgb)
        good = validation.background_line_check(source,source,r,revealed)
        damaged = rgb.copy()
        damaged[revealed] = 190
        bad = validation.background_line_check(source,Image.fromarray(damaged),r,revealed)
        self.assertGreater(good['tested_pixels'],60)
        self.assertTrue(good['passed'])
        self.assertFalse(bad['passed'])

    def test_independent_core_catches_bad_pipeline_mask(self):
        r = self.regions()
        candidate = self.rgb.copy()
        candidate[80:85,50:55] = 0
        reference = np.zeros_like(self.labels,bool)
        reference[80:85,50:55] = True
        checks = validation.preservation_checks(self.source,Image.fromarray(candidate),r,{'held_object':reference})
        self.assertFalse(checks['reference_held_object']['passed'])
        self.assertTrue(checks['outside_approved']['passed'])

    def test_short_hem_and_bare_arms_do_not_trigger_long_sleeve_repair(self):
        r = self.regions('sleeveless mini dress')
        labels = self.labels.copy()
        labels[45:55,45:55] = 14
        labels[90:105,45:55] = 12
        candidate = self.rgb.copy()
        candidate[r.approved] = 30
        check,_ = validation.validate_candidate(self.source,Image.fromarray(candidate),r,labels)
        self.assertEqual(check['unexpected_skin_pixels'],0)

    def test_unrequested_bare_arms_are_rejected(self):
        r = self.regions()
        labels = self.labels.copy()
        labels[45:65,45:60] = 14
        check,_ = validation.validate_candidate(self.source,self.source,r,labels)
        self.assertGreater(check['unexpected_skin_pixels'],30)

    def test_skin_correction_never_changes_original_pixels(self):
        r = self.regions('sleeveless mini dress')
        labels = self.labels.copy()
        labels[40:65,30:45] = 14
        image = self.rgb.copy()
        image[40:65,30:45] = [220,170,130]
        output,info = validation.correct_new_skin(self.source,Image.fromarray(image),r,labels)
        new = np.isin(labels,mapping.SKIN) & r.approved & ~np.isin(r.labels,mapping.SKIN)
        np.testing.assert_array_equal(np.asarray(output)[~new],image[~new])


class RunnerTests(Fixture):
    def setUp(self):
        super().setUp()
        self.r = self.regions()
        self.enterContext(patch.object(engine,'build_source_regions',return_value=self.r))
        self.render = self.enterContext(patch.object(engine,'render_stage',return_value=Image.new('RGB',self.source.size,(20,20,20))))
        self.repair = self.enterContext(patch.object(engine,'repair_crop',side_effect=lambda image,*a,**k:image))
        self.good = {'validation_status':'passed','failure_reasons':[], 'preservation_checks':{}}
        self.bad = {'validation_status':'rejected','failure_reasons':['Unresolved scene seam'], 'preservation_checks':{}}
        self.empty = np.zeros_like(self.labels,bool)
        self.validate = self.enterContext(patch.object(engine,'validate_candidate',return_value=(self.good,self.empty)))

    def run_it(self,variations=1):
        return engine.run_automatic_outfit(self.source,'mini black dress',{'type':'sdxl'},596663721,variations)

    def test_success_stops_after_one_seed(self):
        result = self.run_it()
        self.assertEqual(result['quality']['attempted_seeds'],[596663721])
        self.assertEqual(self.render.call_count,1)

    def test_synthesis_context_cannot_expand_compositing(self):
        self.assertTrue((self.r.synthesis & ~self.r.approved).any())
        result = self.run_it()
        output = np.asarray(Image.open(result['output_path']))
        np.testing.assert_array_equal(output[~self.r.approved],self.rgb[~self.r.approved])

    def test_diagnostic_mode_cannot_publish_even_a_passing_candidate(self):
        with self.assertRaises(engine.OutfitRejected):
            engine.run_automatic_outfit(self.source,'mini black dress',{'type':'sdxl'},5,diagnostic_only=True)
        self.assertFalse(list(self.root.glob('out_*.png')))
        self.assertEqual(self.render.call_count,2)

    def test_informational_color_failure_does_not_override_checks(self):
        with patch('core.quality_checks.garment_color_check',side_effect=RuntimeError('metric unavailable')):
            result = self.run_it()
        self.assertEqual(result['quality']['validation_status'],'passed')
        self.assertIsNone(result['quality_results'][0]['garment_color']['matches'])

    def test_both_fail_only_two_deterministic_seeds_no_download(self):
        self.validate.return_value = self.bad,self.empty
        with self.assertRaises(engine.OutfitRejected) as error:
            self.run_it()
        self.assertEqual(error.exception.quality['attempted_seeds'],[596663721,596663722])
        response = engine.rejection_response(error.exception,'/inputs/source.png')
        self.assertNotIn('output_url',response)
        self.assertNotIn('result_url',response)
        self.assertFalse(list(self.root.glob('out_*.png')))
        self.assertEqual(len(list((self.root/'diagnostics/outfit').glob('*/quality.json'))),1)
        self.assertEqual([call.args[5] for call in self.render.call_args_list],[596663721,596663722])

    def test_preflight_failure_never_renders(self):
        self.r.reasons.append('Text overlaps old clothing')
        with self.assertRaises(engine.OutfitRejected) as error:
            self.run_it()
        self.render.assert_not_called()
        self.assertEqual(error.exception.quality['attempted_seeds'],[])

    def test_partial_batch_has_two_candidate_total_budget(self):
        calls = [self.good,self.good,self.bad,self.bad]
        self.validate.side_effect = lambda *args: (calls.pop(0),self.empty)
        result = self.run_it(4)
        self.assertEqual(result['generated_variations'],1)
        self.assertEqual(result['requested_variations'],4)
        self.assertEqual(self.render.call_count,2)
        self.assertTrue(result['warnings'])

    def test_local_repair_not_accepted_and_not_repeated(self):
        defect = self.empty.copy()
        defect[60:65,45:50] = True
        self.validate.return_value = self.bad,defect
        with self.assertRaises(engine.OutfitRejected) as error:
            self.run_it()
        self.assertEqual(self.repair.call_count,2)
        for candidate in error.exception.quality['candidates']:
            self.assertTrue(candidate['repairs']['localized']['attempted'])
            self.assertFalse(candidate['repairs']['localized']['accepted'])

    def test_stage_exception_consumes_candidate_budget(self):
        self.render.side_effect = RuntimeError('local inference failed')
        with self.assertRaises(engine.OutfitRejected) as error:
            self.run_it()
        self.assertEqual(self.render.call_count,2)
        self.assertEqual(len(error.exception.quality['candidates']),2)

    def test_background_repair_runs_at_most_once_per_candidate(self):
        generated_labels = self.labels.copy()
        generated_labels[80:90,40:55] = 0
        self.parse.return_value = generated_labels
        self.validate.return_value = self.bad,self.empty
        with patch.object(engine,'background_line_check',return_value={'passed':False,'matched_fraction':0,'tested_pixels':100}):
            with self.assertRaises(engine.OutfitRejected) as error:
                self.run_it()
        self.assertEqual(self.repair.call_count,2)
        for item in error.exception.quality['candidates']:
            self.assertTrue(item['repairs']['background']['attempted'])
            self.assertFalse(item['repairs']['background']['accepted'])

    def test_reference_core_miss_rejects_before_generation(self):
        oracle = np.zeros_like(self.labels,bool)
        oracle[60:65,40:45] = True
        with self.assertRaises(engine.OutfitRejected) as error:
            engine.run_automatic_outfit(self.source,'mini black dress',{'type':'sdxl'},5,reference_cores={'phone':oracle})
        self.render.assert_not_called()
        self.assertIn('Independent reference protection failed: phone',error.exception.quality['failure_reasons'])


class PinnedEvidenceTests(unittest.TestCase):
    def test_failed_result_cannot_pass_independent_protection(self):
        import hashlib
        from scripts.verify_automatic_outfit import reference_masks
        folder = Path(__file__).parent/'tests/fixtures/outfit_596663721'
        manifest = json.loads((folder/'manifest.json').read_text())
        for name,digest in manifest['sha256'].items():
            self.assertEqual(hashlib.sha256((folder/name).read_bytes()).hexdigest(),digest)
        source = Image.open(folder/'source.png').convert('RGB')
        failed = Image.open(folder/'failed.png').convert('RGB')
        changed = np.any(np.asarray(source)!=np.asarray(failed),axis=2)
        masks = reference_masks(source)
        self.assertFalse(changed[masks['face']].any())
        for name in ('phone','bag_strap','watermark','exposed_arms','foreground_obstruction'):
            self.assertTrue(changed[masks[name]].any(),name)


class APITests(unittest.TestCase):
    def test_rejected_request_is_422_and_does_not_enter_history(self):
        import server
        handler = object.__new__(server.FaceStudioHandler)
        payload = json.dumps({'image_id':'fixture','prompt':'mini black dress'}).encode()
        handler.path = '/api/change-dress'
        handler.headers = {'Content-Length':str(len(payload))}
        handler.rfile = io.BytesIO(payload)
        rejected = engine.OutfitRejected({'validation_status':'rejected','failure_reasons':['Text overlaps clothing'],'attempted_seeds':[5,6]})
        with patch.object(server,'extract_request_image',return_value=(Image.new('RGB',(10,10)),'/inputs/source.png')), \
             patch.object(server,'run_face_studio_pipeline',side_effect=rejected), \
             patch.object(handler,'send_json') as send:
            before = list(server.HISTORY)
            handler.do_POST()
            status,payload = send.call_args.args
            self.assertEqual(status,422)
            self.assertFalse(payload['success'])
            self.assertEqual(payload['source_url'],'/inputs/source.png')
            self.assertNotIn('output_url',payload)
            self.assertEqual(list(server.HISTORY),before)


if __name__ == '__main__':
    unittest.main()
