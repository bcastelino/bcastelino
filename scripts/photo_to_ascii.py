"""
Convert a photo into ASCII art and inject it into the stats SVGs.

Usage:
    pip install pillow lxml
    python scripts/photo_to_ascii.py assets/me.jpg
    python scripts/photo_to_ascii.py assets/me.jpg --width 36 --dy 17

It replaces the contents of the <text id="ascii_art"> element in both
assets/profile/dark_mode.svg and assets/profile/light_mode.svg. The dark card uses a normal brightness
ramp (bright pixels -> dense chars) and the light card uses the inverted
ramp so the portrait reads correctly on each background.
"""
import argparse
import os
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from lxml import etree

RAMP = " .:-=+*#%@"
SVG_NS = 'http://www.w3.org/2000/svg'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILE_DIR = os.path.join(ROOT, 'assets', 'profile')


def prep_rgba(path, crop=None, pad=0.06, blur=0.6, max_aspect=1.15):
    img = Image.open(path)
    has_alpha = img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info)
    img = img.convert('RGBA')
    w, h = img.size
    if crop:
        l, t, r, b = crop
        img = img.crop((int(l * w), int(t * h), int(r * w), int(b * h)))
    elif has_alpha:
        bbox = img.split()[-1].getbbox()
        if bbox:
            px = int((bbox[2] - bbox[0]) * pad)
            py = int((bbox[3] - bbox[1]) * pad)
            img = img.crop((max(0, bbox[0] - px), max(0, bbox[1] - py), min(w, bbox[2] + px), min(h, bbox[3] + py)))
    if max_aspect:
        cw, ch = img.size
        limit = int(cw * max_aspect)
        if ch > limit:
            img = img.crop((0, 0, cw, limit))
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    return img, has_alpha


def composite_L(rgba, bg_fill, alpha_thresh=128):
    r, g, b, a = rgba.split()
    a = a.point(lambda v: 255 if v >= alpha_thresh else 0)
    subject = Image.merge('RGBA', (r, g, b, a))
    bg = Image.new('RGBA', rgba.size, (bg_fill, bg_fill, bg_fill, 255))
    return Image.alpha_composite(bg, subject).convert('L')


def stylize_dark(rgba, alpha_thresh=128, floor=95):
    """Render the full subject silhouette on black: every masked pixel gets at
    least `floor` ink so dark hair/suit/shoulders stay visible, brights peak."""
    r, g, b, a = rgba.split()
    mask = np.asarray(a) >= alpha_thresh
    lum = np.asarray(Image.merge('RGB', (r, g, b)).convert('L'), dtype=np.float32)
    out = np.zeros_like(lum)
    if mask.any():
        vals = lum[mask]
        lo, hi = float(vals.min()), float(vals.max())
        norm = np.clip((lum - lo) / (hi - lo + 1e-6), 0.0, 1.0)
        v = floor + norm * (255 - floor)
        out[mask] = v[mask]
    return Image.fromarray(out.astype('uint8'), 'L')


def bg_level(img):
    """Estimate backdrop luminance from the two top corners."""
    px = img.load()
    w, h = img.size
    p = max(4, w // 18)
    vals = [px[x, y] for x0 in (0, w - p) for x in range(x0, x0 + p) for y in range(0, p)]
    return sum(vals) / len(vals)


def apply_levels(img, black, white):
    black = max(0, min(254, black))
    white = max(black + 1, min(255, white))
    scale = 255.0 / (white - black)
    return img.point(lambda p: 0 if p <= black else (255 if p >= white else int((p - black) * scale)))


def to_rows(img, width, invert, levels=0, char_aspect=0.52):
    w, h = img.size
    rows = max(1, int(round(width * (h / w) * char_aspect)))
    img = img.resize((width, rows), Image.LANCZOS)
    px = img.load()
    ramp = RAMP[::-1] if invert else RAMP
    n = len(ramp) - 1

    def ch(p):
        if levels > 1:
            p = round(p / 255 * (levels - 1)) * 255 / (levels - 1)
        return ramp[int(p / 255 * n)]

    return [(''.join(ch(px[x, y]) for x in range(width)).rstrip() or ' ') for y in range(rows)]


def inject(svg_name, lines, x=15, y0=28, dy=17):
    svg_path = os.path.join(PROFILE_DIR, svg_name)
    tree = etree.parse(svg_path)
    root = tree.getroot()
    el = root.find(".//*[@id='ascii_art']")
    if el is None:
        raise SystemExit(f"No element with id='ascii_art' found in {svg_name}")
    el.tag = f'{{{SVG_NS}}}text'
    el.set('x', '25')
    el.set('y', '40')
    el.set('class', 'ascii')
    el.attrib.pop('fill', None)
    for child in list(el):
        el.remove(child)
    el.text = None
    for i, line in enumerate(lines):
        tspan = etree.SubElement(el, f'{{{SVG_NS}}}tspan')
        tspan.set('x', str(x))
        tspan.set('y', str(y0 + i * dy))
        tspan.text = line
    style = root.find(".//*[@id='frame_style']")
    if style is not None:
        style.getparent().remove(style)
    tree.write(svg_path, encoding='utf-8', xml_declaration=True)


def main():
    ap = argparse.ArgumentParser(description='Convert a photo to ASCII art and inject it into the stats SVGs.')
    ap.add_argument('image', help='path to source image (jpg/png)')
    ap.add_argument('--width', type=int, default=36, help='ASCII width in characters (default 36)')
    ap.add_argument('--dy', type=int, default=17, help='line height in px (default 17)')
    ap.add_argument('--y0', type=int, default=28, help='y of the first row (default 28)')
    ap.add_argument('--crop', default=None, help='crop as fractions "left,top,right,bottom" (0-1), e.g. 0.20,0.06,0.84,0.60')
    ap.add_argument('--blur', type=float, default=0.9, help='gaussian blur radius to denoise before sampling (default 0.9)')
    ap.add_argument('--bg-margin', type=int, default=14, dest='bg_margin', help='luminance margin for backdrop suppression (default 14)')
    ap.add_argument('--max-aspect', type=float, default=1.15, dest='max_aspect', help='max height/width; trims lower torso so it fits the card (default 1.15, 0 disables)')
    ap.add_argument('--alpha-thresh', type=int, default=128, dest='alpha_thresh', help='alpha cutoff to harden the cutout edge and remove speckle (default 128)')
    ap.add_argument('--floor', type=int, default=95, help='minimum ink for the dark-theme silhouette so dark areas stay visible (default 95)')
    ap.add_argument('--levels', type=int, default=6, help='posterize to N tonal levels to reduce facial noise/crowding (default 6, 0 disables)')
    args = ap.parse_args()

    crop = tuple(float(v) for v in args.crop.split(',')) if args.crop else None
    base, has_alpha = prep_rgba(args.image, crop=crop, blur=args.blur, max_aspect=args.max_aspect)
    if has_alpha:
        dark_lines = to_rows(stylize_dark(base, args.alpha_thresh, args.floor), args.width, invert=False, levels=args.levels)
        light_lines = to_rows(ImageOps.autocontrast(composite_L(base, 255, args.alpha_thresh), cutoff=1), args.width, invert=True, levels=args.levels)
    else:
        gray = base.convert('L')
        bg = bg_level(gray)
        dark_lines = to_rows(apply_levels(gray, bg + args.bg_margin, 245), args.width, invert=False, levels=args.levels)
        light_lines = to_rows(apply_levels(gray, 25, bg - args.bg_margin), args.width, invert=True, levels=args.levels)
    inject('dark_mode.svg', dark_lines, y0=args.y0, dy=args.dy)
    inject('light_mode.svg', light_lines, y0=args.y0, dy=args.dy)
    print(f"Injected {len(dark_lines)} rows (width {args.width}, dy {args.dy}) into dark_mode.svg and light_mode.svg")
    if len(dark_lines) * args.dy + args.y0 > 470:
        print("Note: the portrait is tall; consider a smaller --width so it fits the 480px card height.")


if __name__ == '__main__':
    main()
