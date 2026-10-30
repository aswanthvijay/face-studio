"""Regression checks for CyberRealistic's shared processing paths; no GPU renders."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from core import config, pipeline, face_engine, skin_engine, clothing_engine
from core.prompt_engine import build_zero_pollution_prompt, merge_negative_prompt, garment_requirements
from core.quality_checks import garment_color_check


class PromptTests(unittest.TestCase):
    def test_explicit_lighting_and_expression_are_not_overridden(self):
        request = "black velvet gown, moody backlight, eyes closed, sultry expression"
        positive, _ = build_zero_pollution_prompt(request, mode="full_scene")
        self.assertTrue(positive.startswith("(" + request))
        self.assertNotIn("soft studio lighting", positive)
        self.assertNotIn("relaxed natural expression", positive)
        self.assertNotIn("looking towards camera", positive)

    def test_garment_color_does_not_come_from_accessories(self):
        attributes = garment_requirements("gold necklace with black velvet gown")
        self.assertEqual(attributes["color"], "black")
        self.assertEqual(garment_requirements("peach colored high collar turtleneck dress")["color"], "peach")
    def test_requested_details_are_not_negated(self):
        for request, term in (("sari with bindi and sindoor", "bindi"),
                              ("sari with bindi and sindoor", "sindoor"),
                              ("shirt with text saying HELLO", "text"),
                              ("shirt with logo", "logo")):
            with self.subTest(request=request, term=term):
                positive, negative = build_zero_pollution_prompt(request, mode="full_scene")
                self.assertIn(term, positive)
                self.assertNotIn(term, negative)

    def test_unrequested_details_still_suppressed(self):
        _, negative = build_zero_pollution_prompt("red dress")
        self.assertIn("bindi", negative)
        self.assertIn("sindoor", negative)
        _, negative = build_zero_pollution_prompt("red dress without bindi")
        self.assertIn("bindi", negative)

    def test_covered_neckline_is_not_forced_bare(self):
        for model_type in ("sdxl", "flux", "moss"):
            positive, _ = build_zero_pollution_prompt("black turtleneck", model_type=model_type)
            self.assertNotIn("bare chest", positive)
            self.assertNotIn("bare collarbones", positive)

    def test_duplicates_retain_strongest_weight(self):
        negative = merge_negative_prompt(["(oily skin, floating head:1.3)",
                                          "oily skin, (floating head:1.4)"], "red dress")
        self.assertEqual(negative, "(oily skin:1.3), (floating head:1.4)")


class LoraTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        folder = Path(self.directory.name)
        for name in ("fs_couture", "moss_image_color_depth_control", "ImpurePores",
                     "skin_tone_style_zib_v1.1", "unknown_custom"):
            with (folder / f"{name}.safetensors").open("wb") as file:
                file.truncate(100_001)
        for key in ("LORA_DIR", "LOCAL_LORA_DIR"):
            self.enterContext(patch.object(config, key, folder))

    def test_registry_keeps_architecture(self):
        registry = config.get_available_loras()
        self.assertEqual(registry["moss_image_color_depth_control"]["compat"], "moss")
        self.assertEqual(registry["ImpurePores"]["compat"], "flux")

    def test_ids_and_filenames_have_identical_guards(self):
        for name in ("moss_image_color_depth_control", "ImpurePores", "skin_tone_style_zib_v1.1"):
            for identifier in (name, name + ".safetensors"):
                with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                    config.resolve_lora(identifier, "sdxl")
        self.assertEqual(config.resolve_lora("fs_couture.safetensors", "sdxl")["id"], "fs_couture")

    def test_unknown_or_uninstalled_architecture_is_not_guessed(self):
        for identifier in ("unknown_custom", "missing.safetensors", "../fs_couture.safetensors"):
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                config.resolve_lora(identifier, "sdxl")


class ImageFixture:
    def setUp(self):
        self.rgb = np.full((256, 160, 3), [40, 60, 90], dtype=np.uint8)
        self.rgb[20:70, 60:100] = [200, 150, 120]
        self.rgb[105:225, 15:40] = [135, 85, 65]
        self.labels = np.zeros((256, 160), dtype=np.uint8)
        self.labels[20:70, 60:100] = 11
        self.labels[105:225, 15:40] = 14
        self.labels[100:235, 55:125] = 4
        self.source = Image.fromarray(self.rgb)
        self.face = SimpleNamespace(bbox=np.array([60, 20, 100, 70], dtype=np.float32))
        self.enterContext(patch.object(face_engine, "get_face_modules", return_value=(object(), None, None, None, None)))
        self.enterContext(patch.object(face_engine, "detect_faces_robust", return_value=[self.face]))


class ImageTests(ImageFixture, unittest.TestCase):
    def test_body_correction_preserves_directional_lightness(self):
        import cv2
        rgb = self.rgb.copy()
        for y in range(105,225):
            rgb[y,15:40] = [70+y//2, 40+y//3, 30+y//4]
        output, info = skin_engine.harmonize_unified_canvas_skin(Image.fromarray(rgb),
            face_bbox=self.face.bbox, clothing_labels=self.labels, return_info=True)
        before_l = cv2.cvtColor(rgb.astype(np.float32)/255.0, cv2.COLOR_RGB2LAB)[:,:,0]
        after_l = cv2.cvtColor(np.array(output,dtype=np.float32)/255.0, cv2.COLOR_RGB2LAB)[:,:,0]
        self.assertLess(float(np.max(np.abs(after_l-before_l))), 0.6)
        delta = np.array(output,dtype=np.int16) - rgb.astype(np.int16)
        self.assertLessEqual(int(np.max(np.abs(np.diff(delta[:,15:40],axis=0)))), 3)
        self.assertEqual(float(info["alpha_mask"][105,25]), 0.0)

    def test_hair_recovery_does_not_paste_dark_skin(self):
        labels = self.labels.copy()
        labels[5:20,40:120] = 2
        scene = self.rgb.copy()
        restored = np.full_like(scene,190)
        with patch.object(clothing_engine,"parse_clothing_labels",return_value=labels):
            output = face_engine.restore_scene_hair(restored,scene,self.face.bbox)
        np.testing.assert_array_equal(output[labels!=2],restored[labels!=2])
        self.assertTrue(np.any(output[10:15,60:100]!=restored[10:15,60:100]))
    def test_face_calibration_preserves_distant_pixels_exactly(self):
        noisy = np.random.default_rng(7).integers(0, 256, (256, 160, 3), dtype=np.uint8)
        output = face_engine.calibrate_swapped_face_tone(noisy, self.rgb, self.face.bbox, self.face.bbox)
        np.testing.assert_array_equal(output[150:], noisy[150:])
        self.assertTrue(np.any(output[20:70, 60:100] != noisy[20:70, 60:100]))

    def test_skin_colored_fabric_stays_protected(self):
        rgb = self.rgb.copy()
        rgb[100:150, 65:95] = [170, 120, 90]
        output = skin_engine.harmonize_unified_canvas_skin(Image.fromarray(rgb),
            face_bbox=self.face.bbox, clothing_labels=self.labels)
        np.testing.assert_array_equal(np.array(output)[100:150, 65:95], rgb[100:150, 65:95])

    def test_semantic_bare_chest_still_gets_corrected(self):
        rgb = self.rgb.copy()
        rgb[100:150, 65:95] = [135, 85, 65]
        labels = self.labels.copy()
        labels[100:150, 65:95] = 11
        output = skin_engine.harmonize_unified_canvas_skin(Image.fromarray(rgb),
            face_bbox=self.face.bbox, clothing_labels=labels)
        self.assertTrue(np.any(np.array(output)[110:140, 70:90] != rgb[110:140, 70:90]))

    def test_skin_colored_hair_beside_neck_stays_protected(self):
        rgb = self.rgb.copy()
        labels = self.labels.copy()
        rgb[72:105, 52:64] = [170, 120, 90]
        labels[72:105, 52:64] = 2
        output = skin_engine.harmonize_unified_canvas_skin(Image.fromarray(rgb),
            face_bbox=self.face.bbox, clothing_labels=labels)
        hair = labels == 2
        np.testing.assert_array_equal(np.array(output)[hair], rgb[hair])

    def test_skin_edit_mask_preserves_unedited_skin(self):
        mask = Image.new("L", self.source.size)
        mask.paste(255, (15, 105, 40, 155))
        output = skin_engine.harmonize_unified_canvas_skin(self.source,
            face_bbox=self.face.bbox, clothing_labels=self.labels, edit_mask=mask)
        outside = np.array(mask) == 0
        np.testing.assert_array_equal(np.array(output)[outside], self.rgb[outside])
        self.assertTrue(np.any(np.array(output)[115:145, 20:35] != self.rgb[115:145, 20:35]))


class PipelineTests(ImageFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        for key in ("INPUTS_DIR", "OUTPUTS_DIR"):
            self.enterContext(patch.object(config, key, Path(self.directory.name)))
        self.enterContext(patch.object(pipeline, "ensure_backend", return_value=True))
        self.enterContext(patch.object(pipeline, "upload_image", side_effect=lambda path: path.name))
        self.enterContext(patch.object(pipeline, "comfy_free_memory"))
        self.enterContext(patch.object(pipeline.torch.cuda, "is_available", return_value=False))
        self.enterContext(patch.object(pipeline, "build_zero_pollution_prompt", return_value=("dress", "blur")))
        self.render = self.enterContext(patch.object(pipeline, "execute_workflow", side_effect=lambda *a, **k: self.source.copy()))
        self.face_lock = self.enterContext(patch.object(pipeline, "apply_face_lock", side_effect=lambda **k: (k["target_image"], True)))
        self.enterContext(patch.object(skin_engine, "extract_universal_skin_profile", return_value={"lab_mean": [150, 145, 145]}))
        self.parse = self.enterContext(patch.object(clothing_engine, "parse_clothing_labels", return_value=self.labels))
        self.enterContext(patch.object(clothing_engine,"detect_face_landmarks",return_value=[{
            "box":(60,20,40,50),"chin_y":70,"eye_y":35,"mouth_y":55}]))
        self.mask = Image.new("L", self.source.size)
        self.mask.paste(255, (55, 100, 125, 235))
        self.enterContext(patch.object(pipeline, "generate_dress_mask", return_value=(self.mask, self.source, [], 0.25)))

    def run_pipeline(self, **kwargs):
        return pipeline.run_face_studio_pipeline(self.source, "red dress", **kwargs)

    def test_requested_variations_are_generated(self):
        result = self.run_pipeline(mode="full_scene", variations=3)
        self.assertEqual(len(result["results"]), 3)
        self.assertEqual(result["warnings"], [])
        self.assertEqual(self.render.call_count, 3)

    def test_failed_face_transfer_retries_and_uses_retry_seed(self):
        self.face_lock.side_effect = [(self.source, False), (self.source, True)]
        result = self.run_pipeline(mode="full_scene", seed=11)
        self.assertEqual(self.render.call_count, 2)
        self.assertTrue(result["face_swapped"])
        self.assertNotEqual(result["seed"], 11)

    def test_two_failed_face_transfers_do_not_report_success(self):
        self.face_lock.side_effect = lambda **k: (k["target_image"], False)
        with self.assertRaisesRegex(RuntimeError, "after two attempts"):
            self.run_pipeline(mode="full_scene")
        self.assertEqual(self.render.call_count, 2)

    def test_partial_batch_reports_failure(self):
        self.render.side_effect = [self.source, RuntimeError("render failed"), self.source]
        result = self.run_pipeline(mode="full_scene", variations=3)
        self.assertEqual(result["generated_variations"], 2)
        self.assertEqual(result["requested_variations"], 3)
        self.assertIn("Variation 2 failed", result["warnings"][0])

    def test_in_place_preserves_pixels_outside_mask(self):
        generated = self.rgb.copy()
        generated[100:235, 55:125] = [140, 20, 180]
        self.render.side_effect = lambda *a, **k: Image.fromarray(generated)
        result = self.run_pipeline(mode="in_place_swap")
        with Image.open(result["output_path"]) as output:
            actual = np.array(output)
        outside = np.array(self.mask) == 0
        np.testing.assert_array_equal(actual[outside], self.rgb[outside])
        self.assertTrue(np.any(actual[120:210, 65:115] != self.rgb[120:210, 65:115]))
        self.assertGreaterEqual(self.parse.call_count, 1)
        self.assertIsNot(self.parse.call_args.args[0], self.source)

    def test_inpaint_conditioning_masks_old_pattern_before_sampling(self):
        graph, _ = pipeline.build_sdxl_inpaint_graph("source.png","mask.png","peach dress","blur",config.MODELS["cyberrealistic_xl"])
        self.assertEqual(graph["neutral_source"]["inputs"]["image"],["resize_source",0])
        self.assertEqual(graph["inpaint_conditioning"]["inputs"]["pixels"],["neutral_source",0])

    def test_invalid_model_rejected_before_backend_call(self):
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            self.run_pipeline(model_id="missing")
        pipeline.ensure_backend.assert_not_called()


class ClothingMaskTests(unittest.TestCase):
    def test_clipseg_clothing_confidence_cannot_override_hand_protection(self):
        labels=np.zeros((280,160),dtype=np.uint8)
        labels[90:270,10:35]=14
        labels[100:270,40:120]=4
        processor=lambda **kwargs: {"count":len(kwargs["text"])}
        model=lambda **kwargs: SimpleNamespace(logits=pipeline.torch.full((kwargs["count"],40,40),10.0))
        hand_mask = np.zeros_like(labels, dtype=bool)
        hand_mask[235:270,10:35] = True
        with patch.object(clothing_engine,"parse_clothing_labels",return_value=labels), \
             patch.object(clothing_engine,"get_clipseg",return_value=(processor,model)), \
             patch("core.hand_regions.hand_protection", return_value=(hand_mask, [{}])):
            mask,_,_,_=clothing_engine.generate_dress_mask(self.image,prompt="long sleeve dress")
        self.assertEqual(np.array(mask)[255,20],0)
        self.assertGreater(np.array(mask)[150,20],200)

    def test_missing_hand_landmarks_preserve_unknown_arm_and_report_it(self):
        labels=np.zeros((280,160),dtype=np.uint8)
        labels[90:270,10:35]=14
        labels[100:270,40:120]=4
        with patch.object(clothing_engine,"parse_clothing_labels",return_value=labels), \
             patch("core.hand_regions.hand_protection", return_value=(np.zeros_like(labels,dtype=bool), [])):
            mask,_,_,_=clothing_engine.generate_dress_mask(self.image,prompt="long sleeve dress")
        self.assertEqual(np.array(mask)[150,20],0)
        self.assertIn("hand boundary", mask.quality_warnings[0])

    def setUp(self):
        self.image = Image.new("RGB", (160, 280), (180, 130, 100))
        self.enterContext(patch.object(clothing_engine, "face_contour", return_value=None))
        labels = np.zeros((280, 160), dtype=np.uint8)
        labels[15:80, 55:105] = 11
        labels[80:120, 65:95] = 11
        labels[100:120, 30:60] = 11
        labels[120:155, 35:125] = 4
        labels[155:210, 55:105] = 11
        labels[210:270, 35:125] = 5
        self.enterContext(patch.object(clothing_engine, "parse_clothing_labels", return_value=labels))
        self.enterContext(patch.object(clothing_engine, "get_clipseg", side_effect=FileNotFoundError("optional")))
        self.enterContext(patch.object(clothing_engine, "detect_face_landmarks", return_value=[
            {"box": (55, 15, 50, 65), "chin_y": 80, "eye_y": 40, "mouth_y": 65}]))

    def test_high_collar_and_midriff_are_editable_but_face_is_protected(self):
        mask, _, _, _ = clothing_engine.generate_dress_mask(self.image, prompt="high collar turtleneck dress")
        array = np.array(mask)
        self.assertGreater(array[100, 80], 200)
        self.assertGreater(array[110, 45], 200)
        self.assertGreater(array[180, 80], 200)
        self.assertEqual(array[45, 80], 0)

    def test_requested_bare_midriff_is_preserved(self):
        mask, _, _, _ = clothing_engine.generate_dress_mask(self.image, prompt="crop top and skirt, bare midriff")
        self.assertEqual(np.array(mask)[180, 80], 0)

    def test_jaw_contour_cannot_unprotect_forehead(self):
        labels = np.full((280,160),4,dtype=np.uint8)
        contour = np.zeros_like(labels,dtype=bool)
        contour[35:81,55:105] = True
        with patch.object(clothing_engine,"parse_clothing_labels",return_value=labels), \
             patch.object(clothing_engine,"face_contour",return_value=contour):
            mask,_,_,_=clothing_engine.generate_dress_mask(self.image,prompt="high collar dress")
        self.assertEqual(np.array(mask)[20,80],0)


class QualityCheckTests(unittest.TestCase):
    def test_peach_is_distinguished_from_magenta(self):
        labels=np.full((50,50),7,dtype=np.uint8)
        peach=garment_color_check(Image.new("RGB",(50,50),(255,203,164)),labels,"peach")
        pink=garment_color_check(Image.new("RGB",(50,50),(240,40,150)),labels,"peach")
        self.assertTrue(peach["matches"])
        self.assertFalse(pink["matches"])

    def test_wrong_white_garment_does_not_pass_black_request(self):
        labels=np.full((50,50),7,dtype=np.uint8)
        self.assertFalse(garment_color_check(Image.new("RGB",(50,50),"ivory"),labels,"black")["matches"])


class InpaintRegionTests(unittest.TestCase):
    def test_spill_check_excludes_original_arms_and_protected_pixels(self):
        from core.garment_repair import unexpected_arm_skin
        source = np.full((100,100),4,dtype=np.uint8)
        source[10:60,10:30]=14
        result = source.copy()
        result[65:90,40:60]=14
        result[65:90,70:90]=15
        allowed = np.ones_like(source,dtype=bool)
        allowed[:,70:]=False
        spill = unexpected_arm_skin(source,result,allowed)
        self.assertEqual(int(spill.sum()),500)
        self.assertFalse(spill[20,20])
        self.assertFalse(spill[70,75])

    def test_blending_does_not_reintroduce_clothing_at_image_border(self):
        from core.inpaint_regions import composite_alpha
        garment = np.zeros((70, 80), dtype=bool)
        garment[20:, 20:60] = True
        protected = np.zeros_like(garment)
        protected[35:50, 30:40] = True
        alpha = composite_alpha(garment, protected, 10)
        self.assertTrue(np.all(alpha[garment & ~protected] == 255))
        self.assertTrue(np.all(alpha[protected] == 0))
        self.assertEqual(alpha[-1, 50], 255)
        self.assertGreater(alpha[40, 18], 0)
        self.assertEqual(alpha[40, 0], 0)



if __name__ == "__main__":
    unittest.main(verbosity=2)
