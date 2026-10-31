import io
import unittest

from PIL import Image

from server import decode_uploaded_image


class UploadedImageTests(unittest.TestCase):
    def encode(self, image, format, exif=None):
        stream = io.BytesIO()
        options = {"exif": exif} if exif is not None else {}
        image.save(stream, format=format, **options)
        return stream.getvalue()

    def test_decodes_jpeg_png_and_webp_as_rgb(self):
        source = Image.new("RGBA", (12, 8), (20, 80, 140, 255))
        for image_format in ("JPEG", "PNG", "WEBP"):
            with self.subTest(image_format=image_format):
                payload = self.encode(source.convert("RGB"), image_format)
                decoded = decode_uploaded_image(payload)
                self.assertEqual(decoded.mode, "RGB")
                self.assertEqual(decoded.size, source.size)

    def test_applies_exif_orientation_before_returning_image(self):
        exif = Image.Exif()
        exif[274] = 6
        payload = self.encode(Image.new("RGB", (20, 10), "white"), "JPEG", exif)
        decoded = decode_uploaded_image(payload)
        self.assertEqual(decoded.size, (10, 20))

    def test_rejects_camera_raw_with_export_guidance(self):
        stream = io.BytesIO()
        Image.new("RGB", (8, 8)).save(stream, "TIFF")
        with self.assertRaisesRegex(ValueError, "camera RAW files must be exported"):
            decode_uploaded_image(stream.getvalue())

    def test_rejects_corrupt_image_with_supported_formats(self):
        with self.assertRaisesRegex(ValueError, "not a readable JPEG"):
            decode_uploaded_image(b"not an image")


if __name__ == "__main__":
    unittest.main()
