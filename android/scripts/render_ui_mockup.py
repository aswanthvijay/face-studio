"""
Generates a photorealistic, pixel-perfect UI preview mockup for Photovance AI Studio.
"""
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter

brain_dir = r"C:\Users\Aswanth Vijay\.gemini\antigravity\brain\a85f4511-27eb-48cb-9eea-6faf8206cea6"
drawable_dir = r"d:\image mod\android\app\src\main\res\drawable"
sample_img_p = r"d:\image mod\samples\sample_model.png"
result_img_p = os.path.join(brain_dir, "facelock_monaco_yacht_gold_dress_img1.png")

W, H = 1080, 2400
screen = Image.new("RGBA", (W, H), (7, 9, 14, 255))
draw = ImageDraw.Draw(screen)

# Load fonts
try:
    font_bold_lg = ImageFont.truetype("arialbd.ttf", 46)
    font_bold_md = ImageFont.truetype("arialbd.ttf", 34)
    font_bold_sm = ImageFont.truetype("arialbd.ttf", 26)
    font_reg_md = ImageFont.truetype("arial.ttf", 30)
    font_reg_sm = ImageFont.truetype("arial.ttf", 24)
    font_chip = ImageFont.truetype("arialbd.ttf", 22)
except Exception:
    font_bold_lg = font_bold_md = font_bold_sm = font_reg_md = font_reg_sm = font_chip = ImageFont.load_default()

# 1. Subtle Ambient Radial Backlight Glows
glow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
glow_draw = ImageDraw.Draw(glow_layer)
# Top-left violet glow
glow_draw.ellipse([(-100, 150), (600, 850)], fill=(139, 92, 246, 45))
# Center-right cyan glow
glow_draw.ellipse([(550, 400), (1200, 1100)], fill=(6, 182, 212, 35))
# Bottom magenta glow
glow_draw.ellipse([(200, 1500), (900, 2200)], fill=(236, 72, 153, 40))
glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(140))
screen = Image.alpha_composite(screen, glow_layer)
draw = ImageDraw.Draw(screen)

# 2. Status Bar
draw.text((60, 40), "15:42", fill=(240, 245, 255, 255), font=font_bold_sm)
draw.text((W - 180, 40), "5G  ●  98%", fill=(200, 210, 230, 255), font=font_bold_sm)

# 3. Top Header Bar (Y: 100 - 180)
draw.text((60, 115), "PHOTOVANCE", fill=(255, 255, 255, 255), font=font_bold_lg)
draw.text((410, 115), "AI", fill=(236, 72, 153, 255), font=font_bold_lg)

# PRO Badge
badge_x, badge_y, badge_w, badge_h = 475, 122, 90, 38
draw.rounded_rectangle([badge_x, badge_y, badge_x + badge_w, badge_y + badge_h], radius=10, fill=(245, 158, 11, 230))
draw.text((badge_x + 16, badge_y + 6), "PRO", fill=(0, 0, 0, 255), font=font_bold_sm)

# Status Pill (GPU Online)
pill_x, pill_y, pill_w, pill_h = 600, 118, 260, 48
draw.rounded_rectangle([pill_x, pill_y, pill_x + pill_w, pill_y + pill_h], radius=24, fill=(19, 27, 46, 220), outline=(255, 255, 255, 40), width=1)
draw.ellipse([pill_x + 18, pill_y + 18, pill_x + 30, pill_y + 30], fill=(16, 185, 129, 255))
draw.text((pill_x + 42, pill_y + 12), "GPU Online · 0.0s", fill=(220, 230, 245, 255), font=font_chip)

# Settings Gear Icon button
gear_x = W - 110
draw.rounded_rectangle([gear_x, 118, gear_x + 50, 166], radius=25, fill=(19, 27, 46, 220), outline=(255, 255, 255, 40), width=1)
draw.text((gear_x + 13, 124), "⚙", fill=(255, 255, 255, 230), font=font_bold_md)

# 4. Mode Switcher (Y: 200 - 275)
mode_box = [60, 205, W - 60, 275]
draw.rounded_rectangle(mode_box, radius=18, fill=(15, 22, 38, 230), outline=(255, 255, 255, 30), width=1)
# Active Tab: Face Studio
draw.rounded_rectangle([66, 211, 530, 269], radius=14, fill=(99, 102, 241, 255))
draw.text((170, 224), "🌟 Face Studio (Any Pose/Scene)", fill=(255, 255, 255, 255), font=font_bold_sm)
# Inactive Tab: In-Place Swap
draw.text((580, 224), "👗 In-Place Outfit Swap", fill=(148, 163, 184, 255), font=font_bold_sm)

# 5. Hero Studio Canvas (Portrait Frame) (Y: 305 - 840)
canvas_box = [60, 305, W - 60, 840]
# Neon border glow
neon_border = Image.new("RGBA", (W, H), (0, 0, 0, 0))
nb_draw = ImageDraw.Draw(neon_border)
nb_draw.rounded_rectangle([58, 303, W - 58, 842], radius=28, outline=(139, 92, 246, 120), width=3)
screen = Image.alpha_composite(screen, neon_border)
draw = ImageDraw.Draw(screen)

# Load portrait image
if os.path.exists(sample_img_p):
    with Image.open(sample_img_p) as s_img:
        s_img = s_img.convert("RGBA")
        cw = W - 120
        ch = 840 - 305
        # Resize to cover
        scale = max(cw / s_img.width, ch / s_img.height)
        new_size = (int(s_img.width * scale), int(s_img.height * scale))
        s_resized = s_img.resize(new_size, Image.Resampling.LANCZOS)
        # Center crop
        crop_x = (s_resized.width - cw) // 2
        crop_y = (s_resized.height - ch) // 2
        s_cropped = s_resized.crop((crop_x, crop_y, crop_x + cw, crop_y + ch))

        # Rounded mask
        mask = Image.new("L", (cw, ch), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rounded_rectangle([0, 0, cw, ch], radius=26, fill=255)
        screen.paste(s_cropped, (60, 305), mask)

# Gradient overlay at bottom of canvas
grad_overlay = Image.new("RGBA", (W - 120, 220), (0, 0, 0, 0))
go_draw = ImageDraw.Draw(grad_overlay)
for y in range(220):
    alpha = int((y / 220.0) ** 1.5 * 220)
    go_draw.line([(0, y), (W - 120, y)], fill=(7, 9, 14, alpha))
screen.paste(grad_overlay, (60, 840 - 220), grad_overlay)
draw = ImageDraw.Draw(screen)

# Floating Badges on Portrait
# Face Locked Badge
draw.rounded_rectangle([85, 330, 360, 380], radius=20, fill=(10, 15, 26, 210), outline=(16, 185, 129, 180), width=2)
draw.ellipse([105, 347, 121, 363], fill=(16, 185, 129, 255))
draw.text((135, 342), "Face Identity Locked 100%", fill=(240, 250, 245, 255), font=font_chip)

# Retake Button
draw.rounded_rectangle([W - 250, 330, W - 85, 380], radius=20, fill=(10, 15, 26, 210), outline=(255, 255, 255, 50), width=1)
draw.text((W - 225, 342), "🔄 Retake Photo", fill=(240, 245, 255, 255), font=font_chip)

# Bottom Studio Engine Chip on Canvas
draw.rounded_rectangle([85, 775, 620, 820], radius=18, fill=(99, 102, 241, 40), outline=(99, 102, 241, 140), width=1)
draw.text((105, 785), "⚡ Engine: Flux.2 Klein 4B Turbo · Heretic DiT", fill=(225, 235, 255, 255), font=font_chip)

# 6. Visual Style Inspiration Reel Section (Y: 875 - 1320)
draw.text((60, 875), "Style Inspiration Reel", fill=(255, 255, 255, 255), font=font_bold_md)
draw.text((W - 320, 882), "Explore 24+ Styles  ›", fill=(139, 92, 246, 255), font=font_bold_sm)

# Category Filter Tabs (Y: 930 - 975)
cat_tabs = [("All", True), ("Red Carpet", False), ("Vacation", False), ("Street Casual", False), ("Cyberpunk", False)]
cx = 60
for label, active in cat_tabs:
    tw = int(font_chip.getlength(label)) + 36
    if active:
        draw.rounded_rectangle([cx, 930, cx + tw, 975], radius=14, fill=(99, 102, 241, 240))
        draw.text((cx + 18, 942), label, fill=(255, 255, 255, 255), font=font_chip)
    else:
        draw.rounded_rectangle([cx, 930, cx + tw, 975], radius=14, fill=(19, 27, 46, 160), outline=(255, 255, 255, 30), width=1)
        draw.text((cx + 18, 942), label, fill=(148, 163, 184, 255), font=font_chip)
    cx += tw + 14

# Visual Style Cards (Y: 1000 - 1320)
cards = [
    ("style_monaco_yacht.webp", "Monaco Yacht", "Gold Chainmail", "🔥 Trending", True),
    ("style_cannes_gala.webp", "Cannes Gala", "Emerald Velvet", "👑 VIP", False),
    ("style_maldives_beach.webp", "Maldives Beach", "Tropical Red Bikini", "✨ Popular", False),
    ("style_tokyo_street.webp", "Tokyo Moto", "Black Leather Biker", "⚡ Edgy", False),
]

card_w, card_h = 280, 330
card_x = 60
card_y = 1000

for img_name, title, subtitle, badge, is_sel in cards:
    img_path = os.path.join(drawable_dir, img_name)
    if os.path.exists(img_path):
        with Image.open(img_path) as c_img:
            c_img = c_img.convert("RGBA").resize((card_w, card_h), Image.Resampling.LANCZOS)
            c_mask = Image.new("L", (card_w, card_h), 0)
            ImageDraw.Draw(c_mask).rounded_rectangle([0, 0, card_w, card_h], radius=22, fill=255)
            screen.paste(c_img, (card_x, card_y), c_mask)

    # Card gradient bottom overlay
    c_grad = Image.new("RGBA", (card_w, 140), (0, 0, 0, 0))
    cg_draw = ImageDraw.Draw(c_grad)
    for y in range(140):
        cg_draw.line([(0, y), (card_w, y)], fill=(7, 9, 14, int((y / 140.0) ** 1.4 * 235)))
    screen.paste(c_grad, (card_x, card_y + card_h - 140), c_grad)
    draw = ImageDraw.Draw(screen)

    # Card border
    if is_sel:
        draw.rounded_rectangle([card_x - 3, card_y - 3, card_x + card_w + 3, card_y + card_h + 3], radius=24, outline=(236, 72, 153, 255), width=3)
        # Selection checkmark badge
        draw.rounded_rectangle([card_x + card_w - 46, card_y + 12, card_x + card_w - 12, card_y + 46], radius=12, fill=(236, 72, 153, 255))
        draw.text((card_x + card_w - 38, card_y + 16), "✓", fill=(255, 255, 255, 255), font=font_bold_sm)
    else:
        draw.rounded_rectangle([card_x, card_y, card_x + card_w, card_y + card_h], radius=22, outline=(255, 255, 255, 30), width=1)

    # Tag Badge (top-left)
    draw.rounded_rectangle([card_x + 12, card_y + 12, card_x + 12 + int(font_chip.getlength(badge)) + 20, card_y + 42], radius=10, fill=(10, 15, 26, 210))
    draw.text((card_x + 22, card_y + 16), badge, fill=(255, 255, 255, 255), font=font_chip)

    # Titles
    draw.text((card_x + 16, card_y + card_h - 68), title, fill=(255, 255, 255, 255), font=font_bold_sm)
    draw.text((card_x + 16, card_y + card_h - 38), subtitle, fill=(148, 163, 184, 255), font=font_chip)

    card_x += card_w + 20

# 7. Prompt & Magic Modifiers Bar (Y: 1360 - 1570)
draw.text((60, 1360), "Outfit & Scene Prompt", fill=(255, 255, 255, 255), font=font_bold_sm)

prompt_box = [60, 1400, W - 60, 1500]
draw.rounded_rectangle(prompt_box, radius=18, fill=(15, 22, 38, 220), outline=(99, 102, 241, 100), width=1)
draw.text((85, 1420), "wearing a glamorous backless gold chainmail dress, standing on the teak", fill=(240, 245, 255, 255), font=font_reg_sm)
draw.text((85, 1455), "deck of a luxury superyacht in Monaco during golden hour sunset...", fill=(148, 163, 184, 255), font=font_reg_sm)

# Quick modifier chips (Y: 1515 - 1565)
chips = ["🌅 Golden Hour", "📸 8K Studio Flash", "🧵 Silk Drape", "✨ Glitter Sequin", "🎬 Cinematic"]
cx = 60
for chip in chips:
    cw = int(font_chip.getlength(chip)) + 28
    draw.rounded_rectangle([cx, 1518, cx + cw, 1560], radius=12, fill=(255, 255, 255, 15), outline=(255, 255, 255, 30), width=1)
    draw.text((cx + 14, 1528), chip, fill=(210, 220, 235, 255), font=font_chip)
    cx += cw + 12

# 8. Hero Generate Button (Y: 1595 - 1695)
btn_box = [60, 1595, W - 60, 1695]
# Button gradient simulation
btn_grad = Image.new("RGBA", (W - 120, 100), (0, 0, 0, 0))
bg_draw = ImageDraw.Draw(btn_grad)
for x in range(W - 120):
    t = x / float(W - 120)
    r = int(99 * (1 - t) + 236 * t)
    g = int(102 * (1 - t) + 72 * t)
    b = int(241 * (1 - t) + 153 * t)
    bg_draw.line([(x, 0), (x, 100)], fill=(r, g, b, 255))
btn_mask = Image.new("L", (W - 120, 100), 0)
ImageDraw.Draw(btn_mask).rounded_rectangle([0, 0, W - 120, 100], radius=24, fill=255)
screen.paste(btn_grad, (60, 1595), btn_mask)
draw = ImageDraw.Draw(screen)

# Button Text
draw.text((250, 1618), "✨ GENERATE WITH PHOTOVANCE AI", fill=(255, 255, 255, 255), font=font_bold_md)
draw.text((365, 1662), "⚡ 4-step DiT render · ~12s with 100% Face Lock", fill=(255, 235, 245, 220), font=font_chip)

# 9. Result Showcase with Before/After Split Slider (Y: 1730 - 2280)
draw.text((60, 1730), "Result: Interactive Before & After", fill=(255, 255, 255, 255), font=font_bold_md)

res_box = [60, 1775, W - 60, 2220]
rw = W - 120
rh = 2220 - 1775

# Create split image
split_img = Image.new("RGBA", (rw, rh), (0, 0, 0, 255))

# Load original & result
if os.path.exists(sample_img_p) and os.path.exists(result_img_p):
    with Image.open(sample_img_p) as orig_img, Image.open(result_img_p) as res_img:
        orig_img = orig_img.convert("RGBA")
        res_img = res_img.convert("RGBA")

        # Scale both to fill rw, rh
        scale_o = max(rw / orig_img.width, rh / orig_img.height)
        o_scaled = orig_img.resize((int(orig_img.width * scale_o), int(orig_img.height * scale_o)), Image.Resampling.LANCZOS)
        o_crop = o_scaled.crop(((o_scaled.width - rw) // 2, (o_scaled.height - rh) // 2, (o_scaled.width - rw) // 2 + rw, (o_scaled.height - rh) // 2 + rh))

        scale_r = max(rw / res_img.width, rh / res_img.height)
        r_scaled = res_img.resize((int(res_img.width * scale_r), int(res_img.height * scale_r)), Image.Resampling.LANCZOS)
        r_crop = r_scaled.crop(((r_scaled.width - rw) // 2, (r_scaled.height - rh) // 2, (r_scaled.width - rw) // 2 + rw, (r_scaled.height - rh) // 2 + rh))

        # Split at 45% (slider position)
        split_x = int(rw * 0.45)
        split_img.paste(o_crop.crop((0, 0, split_x, rh)), (0, 0))
        split_img.paste(r_crop.crop((split_x, 0, rw, rh)), (split_x, 0))

# Apply rounded corners to split result
res_mask = Image.new("L", (rw, rh), 0)
ImageDraw.Draw(res_mask).rounded_rectangle([0, 0, rw, rh], radius=24, fill=255)
screen.paste(split_img, (60, 1775), res_mask)
draw = ImageDraw.Draw(screen)

# Draw slider line & handle
slider_pos_x = 60 + int(rw * 0.45)
draw.line([(slider_pos_x, 1775), (slider_pos_x, 2220)], fill=(255, 255, 255, 240), width=4)
# Glowing handle knob
handle_y = 1775 + rh // 2
draw.ellipse([slider_pos_x - 28, handle_y - 28, slider_pos_x + 28, handle_y + 28], fill=(255, 255, 255, 255), outline=(99, 102, 241, 255), width=4)
draw.text((slider_pos_x - 14, handle_y - 12), "◄ ►", fill=(99, 102, 241, 255), font=font_chip)

# Floating Pills: "Original" and "Photovance AI"
draw.rounded_rectangle([85, 1800, 220, 1845], radius=14, fill=(10, 15, 26, 210), outline=(255, 255, 255, 60), width=1)
draw.text((105, 1812), "Original", fill=(220, 230, 245, 255), font=font_chip)

draw.rounded_rectangle([W - 275, 1800, W - 85, 1845], radius=14, fill=(236, 72, 153, 210))
draw.text((W - 255, 1812), "Photovance 8K", fill=(255, 255, 255, 255), font=font_chip)

# Result Actions (Y: 2240 - 2310)
btn_actions = [
    ("💾 Save Gallery", False),
    ("📤 Share HD", False),
    ("🔄 Hold to Compare", True)
]
bx = 60
bw = (rw - 30) // 3
for text, is_hl in btn_actions:
    if is_hl:
        draw.rounded_rectangle([bx, 2240, bx + bw, 2300], radius=14, fill=(99, 102, 241, 240))
        draw.text((bx + 18, 2256), text, fill=(255, 255, 255, 255), font=font_chip)
    else:
        draw.rounded_rectangle([bx, 2240, bx + bw, 2300], radius=14, fill=(19, 27, 46, 220), outline=(255, 255, 255, 30), width=1)
        draw.text((bx + 26, 2256), text, fill=(240, 245, 255, 255), font=font_chip)
    bx += bw + 15

# 10. Android Navigation Pill (Y: 2360)
draw.rounded_rectangle([W // 2 - 140, 2360, W // 2 + 140, 2370], radius=5, fill=(255, 255, 255, 120))

# Save output
out_dir = r"d:\image mod\android\play_store_assets"
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "photovance_ui_mockup_sample.png")
screen.convert("RGB").save(out_file, "PNG", quality=95)
print(f"Mockup saved to: {out_file}")

# Copy to brain artifact directory so it can be viewed and linked in artifact
brain_out_file = os.path.join(brain_dir, "photovance_ui_mockup_sample.png")
screen.convert("RGB").save(brain_out_file, "PNG", quality=95)
print(f"Copied to brain artifact dir: {brain_out_file}")
