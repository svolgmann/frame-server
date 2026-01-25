import time
import numpy as np

from logging_utils import dlog, progress_row
from color_utils import quantize_to_six_rgb_fast


def floyd_steinberg_dither(img: np.ndarray, strength: float) -> np.ndarray:
    h, w, _ = img.shape
    temp = img.copy()
    for y in range(h):
        progress_row(y, h)
        for x in range(w):
            r, g, b = temp[y, x]
            new = quantize_to_six_rgb_fast(float(r), float(g), float(b))
            err = (np.array([r, g, b], dtype=np.float32) - new) * strength
            temp[y, x] = new

            if x + 1 < w:
                temp[y, x + 1] = np.clip(temp[y, x + 1] + err * (7/16), 0, 255)
            if y + 1 < h:
                if x > 0:
                    temp[y + 1, x - 1] = np.clip(temp[y + 1, x - 1] + err * (3/16), 0, 255)
                temp[y + 1, x] = np.clip(temp[y + 1, x] + err * (5/16), 0, 255)
                if x + 1 < w:
                    temp[y + 1, x + 1] = np.clip(temp[y + 1, x + 1] + err * (1/16), 0, 255)
    return temp


def atkinson_dither(img: np.ndarray, strength: float) -> np.ndarray:
    h, w, _ = img.shape
    temp = img.copy()
    out = np.zeros_like(img)
    frac = 1/8

    for y in range(h):
        progress_row(y, h)
        for x in range(w):
            r, g, b = temp[y, x]
            new = quantize_to_six_rgb_fast(float(r), float(g), float(b))
            out[y, x] = new

            err = (np.array([r, g, b], dtype=np.float32) - new) * strength

            if x + 1 < w:
                temp[y, x + 1] = np.clip(temp[y, x + 1] + err * frac, 0, 255)
            if x + 2 < w:
                temp[y, x + 2] = np.clip(temp[y, x + 2] + err * frac, 0, 255)

            if y + 1 < h:
                if x > 0:
                    temp[y + 1, x - 1] = np.clip(temp[y + 1, x - 1] + err * frac, 0, 255)
                temp[y + 1, x] = np.clip(temp[y + 1, x] + err * frac, 0, 255)
                if x + 1 < w:
                    temp[y + 1, x + 1] = np.clip(temp[y + 1, x + 1] + err * frac, 0, 255)

            if y + 2 < h:
                temp[y + 2, x] = np.clip(temp[y + 2, x] + err * frac, 0, 255)

    return out


def stucki_dither(img: np.ndarray, strength: float) -> np.ndarray:
    h, w, _ = img.shape
    temp = img.copy()
    div = 42.0

    for y in range(h):
        progress_row(y, h)
        for x in range(w):
            r, g, b = temp[y, x]
            new = quantize_to_six_rgb_fast(float(r), float(g), float(b))
            err = (np.array([r, g, b], dtype=np.float32) - new) * strength
            temp[y, x] = new

            # same row
            if x + 1 < w:
                temp[y, x + 1] = np.clip(temp[y, x + 1] + err * (8/div), 0, 255)
            if x + 2 < w:
                temp[y, x + 2] = np.clip(temp[y, x + 2] + err * (4/div), 0, 255)

            # y+1
            if y + 1 < h:
                if x > 1:
                    temp[y + 1, x - 2] = np.clip(temp[y + 1, x - 2] + err * (2/div), 0, 255)
                if x > 0:
                    temp[y + 1, x - 1] = np.clip(temp[y + 1, x - 1] + err * (4/div), 0, 255)
                temp[y + 1, x] = np.clip(temp[y + 1, x] + err * (8/div), 0, 255)
                if x + 1 < w:
                    temp[y + 1, x + 1] = np.clip(temp[y + 1, x + 1] + err * (4/div), 0, 255)
                if x + 2 < w:
                    temp[y + 1, x + 2] = np.clip(temp[y + 1, x + 2] + err * (2/div), 0, 255)

            # y+2
            if y + 2 < h:
                if x > 1:
                    temp[y + 2, x - 2] = np.clip(temp[y + 2, x - 2] + err * (1/div), 0, 255)
                if x > 0:
                    temp[y + 2, x - 1] = np.clip(temp[y + 2, x - 1] + err * (2/div), 0, 255)
                temp[y + 2, x] = np.clip(temp[y + 2, x] + err * (4/div), 0, 255)
                if x + 1 < w:
                    temp[y + 2, x + 1] = np.clip(temp[y + 2, x + 1] + err * (2/div), 0, 255)
                if x + 2 < w:
                    temp[y + 2, x + 2] = np.clip(temp[y + 2, x + 2] + err * (1/div), 0, 255)

    return temp


def jarvis_dither(img: np.ndarray, strength: float) -> np.ndarray:
    h, w, _ = img.shape
    temp = img.copy()
    div = 48.0

    for y in range(h):
        progress_row(y, h)
        for x in range(w):
            r, g, b = temp[y, x]
            new = quantize_to_six_rgb_fast(float(r), float(g), float(b))
            err = (np.array([r, g, b], dtype=np.float32) - new) * strength
            temp[y, x] = new

            # y row
            if x + 1 < w:
                temp[y, x + 1] = np.clip(temp[y, x + 1] + err * (7/div), 0, 255)
            if x + 2 < w:
                temp[y, x + 2] = np.clip(temp[y, x + 2] + err * (5/div), 0, 255)

            # y+1 row
            if y + 1 < h:
                if x > 1:
                    temp[y + 1, x - 2] = np.clip(temp[y + 1, x - 2] + err * (3/div), 0, 255)
                if x > 0:
                    temp[y + 1, x - 1] = np.clip(temp[y + 1, x - 1] + err * (5/div), 0, 255)
                temp[y + 1, x] = np.clip(temp[y + 1, x] + err * (7/div), 0, 255)
                if x + 1 < w:
                    temp[y + 1, x + 1] = np.clip(temp[y + 1, x + 1] + err * (5/div), 0, 255)
                if x + 2 < w:
                    temp[y + 1, x + 2] = np.clip(temp[y + 1, x + 2] + err * (3/div), 0, 255)

            # y+2 row
            if y + 2 < h:
                if x > 1:
                    temp[y + 2, x - 2] = np.clip(temp[y + 2, x - 2] + err * (1/div), 0, 255)
                if x > 0:
                    temp[y + 2, x - 1] = np.clip(temp[y + 2, x - 1] + err * (3/div), 0, 255)
                temp[y + 2, x] = np.clip(temp[y + 2, x] + err * (5/div), 0, 255)
                if x + 1 < w:
                    temp[y + 2, x + 1] = np.clip(temp[y + 2, x + 1] + err * (3/div), 0, 255)
                if x + 2 < w:
                    temp[y + 2, x + 2] = np.clip(temp[y + 2, x + 2] + err * (1/div), 0, 255)

    return temp


def dither_image(rgb: np.ndarray, method: str, strength: float) -> np.ndarray:
    t0 = time.time()
    if method == "floyd":
        out = floyd_steinberg_dither(rgb, strength)
    elif method == "atkinson":
        out = atkinson_dither(rgb, strength)
    elif method == "stucki":
        out = stucki_dither(rgb, strength)
    elif method == "jarvis":
        out = jarvis_dither(rgb, strength)
    else:
        raise ValueError(method)
    dlog(f"Dither '{method}' done in {time.time() - t0:.2f}s")
    return out