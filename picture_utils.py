import numpy as np
from PIL import Image

from logging_utils import dlog

def center_crop_resize(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Match JS: scale by max ratio then center-crop."""
    img = img.convert("RGB")
    w, h = img.size
    scale = max(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img2 = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    dlog(f"Resize: {w}x{h} -> {new_w}x{new_h}, crop ({left},{top})")
    return img2.crop((left, top, left + target_w, top + target_h))


def adjust_contrast(arr: np.ndarray, factor: float) -> np.ndarray:
    """JS: (v-128)*factor+128 per channel."""
    out = (arr - 128.0) * factor + 128.0
    return np.clip(out, 0, 255)


def center_crop_cover(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """
    Always keep image center, cut off excess borders (CSS cover behavior).
    """
    img = img.convert("RGB")
    w, h = img.size

    scale = max(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h

    return img_resized.crop((left, top, right, bottom))


def rotate_image(img: Image.Image, degrees: int) -> Image.Image:
    """
    Rotate image clockwise by degrees (0,90,180,270).
    Pillow rotates counter-clockwise, so we negate.
    """
    if degrees == 0:
        return img
    dlog(f"Rotate image: {degrees}° clockwise")
    return img.rotate(-degrees, expand=True)

def adjust_saturation(arr: np.ndarray, factor: float) -> np.ndarray:
    """
    Increase/decrease saturation in RGB space via luma interpolation.
    factor=1.0 keeps original, >1 increases saturation, <1 decreases.
    """
    # arr: float32 RGB 0..255
    gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(arr.dtype)
    gray3 = np.stack([gray, gray, gray], axis=-1)
    out = gray3 + (arr - gray3) * factor
    return np.clip(out, 0, 255)

def boost_green_for_spectra6(arr: np.ndarray,
                            sat_factor: float = 1.25,
                            g_gain: float = 1.12,
                            r_reduce: float = 0.92,
                            hue_shift_deg: float = -6.0) -> np.ndarray:
    """
    arr: float32 RGB 0..255
    Boost greens for 6-color e-ink by:
      - selective saturation in green hues
      - slight hue shift (yellow-green -> greener)
      - increase G, reduce R in green region
    """

    x = np.clip(arr, 0, 255).astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]

    # RGB -> HSV (vectorized)
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Hue
    h = np.zeros_like(cmax)
    mask = delta > 1e-6
    # where cmax == r
    idx = mask & (cmax == r)
    h[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6.0
    # where cmax == g
    idx = mask & (cmax == g)
    h[idx] = ((b[idx] - r[idx]) / delta[idx]) + 2.0
    # where cmax == b
    idx = mask & (cmax == b)
    h[idx] = ((r[idx] - g[idx]) / delta[idx]) + 4.0
    h = (h / 6.0)  # 0..1

    # Saturation
    s = np.zeros_like(cmax)
    s[cmax > 1e-6] = delta[cmax > 1e-6] / cmax[cmax > 1e-6]

    v = cmax

    # --- Green mask: Hue roughly 80..170 degrees ---
    # degrees = h*360
    green = (h >= (80/360)) & (h <= (170/360)) & (s > 0.08) & (v > 0.08)

    # shift hue a bit toward greener (negative shift moves yellow-green -> green)
    h2 = h.copy()
    h2[green] = (h2[green] + (hue_shift_deg / 360.0)) % 1.0

    # selective saturation boost
    s2 = s.copy()
    s2[green] = np.clip(s2[green] * sat_factor, 0, 1)

    # HSV -> RGB
    c = v * s2
    hh = h2 * 6.0
    x2 = c * (1.0 - np.abs((hh % 2.0) - 1.0))
    m = v - c

    rp = np.zeros_like(r)
    gp = np.zeros_like(g)
    bp = np.zeros_like(b)

    i = np.floor(hh).astype(np.int32)
    # 0
    sel = (i == 0)
    rp[sel], gp[sel], bp[sel] = c[sel], x2[sel], 0
    # 1
    sel = (i == 1)
    rp[sel], gp[sel], bp[sel] = x2[sel], c[sel], 0
    # 2
    sel = (i == 2)
    rp[sel], gp[sel], bp[sel] = 0, c[sel], x2[sel]
    # 3
    sel = (i == 3)
    rp[sel], gp[sel], bp[sel] = 0, x2[sel], c[sel]
    # 4
    sel = (i == 4)
    rp[sel], gp[sel], bp[sel] = x2[sel], 0, c[sel]
    # 5
    sel = (i >= 5)
    rp[sel], gp[sel], bp[sel] = c[sel], 0, x2[sel]

    out = np.stack([rp + m, gp + m, bp + m], axis=-1)

    # final channel tweak only in green region (push away from yellow)
    out_g = out[..., 1]
    out_r = out[..., 0]
    out[..., 1] = np.where(green, np.clip(out_g * g_gain, 0, 1), out_g)
    out[..., 0] = np.where(green, np.clip(out_r * r_reduce, 0, 1), out_r)

    return np.clip(out * 255.0, 0, 255)