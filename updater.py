#!/usr/bin/env python3
import argparse
import json
import os
import time
import threading
import subprocess
import platform
import io
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

from PIL import Image
from PIL import ImageEnhance
import numpy as np
try:
    from smb.SMBConnection import SMBConnection
    from smb.smb_structs import ProtocolError
    from smb.base import SMBTimeout, NotConnectedError
except Exception:
    SMBConnection = None
    ProtocolError = None
    SMBTimeout = None
    NotConnectedError = None

W, H = 1200, 1600

# Palette wie in eink.html (Index 0..5)
PALETTE = [
    ("Yellow", 255, 255,   0),
    ("Green",   41, 204,  20),
    ("Blue",     0,   0, 255),
    ("Red",    255,   0,   0),
    ("Black",    0,   0,   0),
    ("White",  255, 255, 255),
]

# Nibble‑Mapping wie in eink.html
INDEX_TO_NIBBLE = {
    0: 2,  # Yellow
    1: 5,  # Green
    2: 4,  # Blue
    3: 3,  # Red
    4: 0,  # Black
    5: 1,  # White
}
NIBBLE_TO_INDEX = {v: k for k, v in INDEX_TO_NIBBLE.items()}

def rgb_to_lab(r, g, b):
    r /= 255.0; g /= 255.0; b /= 255.0
    r = ((r + 0.055)/1.055) ** 2.4 if r > 0.04045 else r / 12.92
    g = ((g + 0.055)/1.055) ** 2.4 if g > 0.04045 else g / 12.92
    b = ((b + 0.055)/1.055) ** 2.4 if b > 0.04045 else b / 12.92
    r *= 100.0; g *= 100.0; b *= 100.0
    x = r*0.4124 + g*0.3576 + b*0.1805
    y = r*0.2126 + g*0.7152 + b*0.0722
    z = r*0.0193 + g*0.1192 + b*0.9505
    x /= 95.047; y /= 100.0; z /= 108.883
    x = x ** (1/3) if x > 0.008856 else (7.787*x) + (16/116)
    y = y ** (1/3) if y > 0.008856 else (7.787*y) + (16/116)
    z = z ** (1/3) if z > 0.008856 else (7.787*z) + (16/116)
    return (116*y - 16, 500*(x - y), 200*(y - z))

def lab_dist(l1, l2):
    dl = l1[0]-l2[0]; da = l1[1]-l2[1]; db = l1[2]-l2[2]
    return (0.2*dl*dl + 3*da*da + 3*db*db) ** 0.5

PALETTE_LAB = [rgb_to_lab(r, g, b) for _, r, g, b in PALETTE]

def nearest_palette_index(r, g, b):
    # Blue‑Bias wie im JS
    if r < 50 and g < 150 and b > 100:
        return 2
    lab = rgb_to_lab(r, g, b)
    best_i = 0
    best_d = 1e9
    for i, plab in enumerate(PALETTE_LAB):
        d = lab_dist(lab, plab)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i

def floyd_steinberg_dither_indices(arr: np.ndarray, strength: float) -> np.ndarray:
    h, w, _ = arr.shape
    temp = arr.astype(np.float32, copy=True)
    idx8 = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            r, g, b = temp[y, x]
            idx = nearest_palette_index(float(r), float(g), float(b))
            pr, pg, pb = PALETTE[idx][1], PALETTE[idx][2], PALETTE[idx][3]
            idx8[y, x] = idx

            err = (np.array([r, g, b], dtype=np.float32) - np.array([pr, pg, pb], dtype=np.float32)) * strength
            temp[y, x] = (pr, pg, pb)

            if x + 1 < w:
                temp[y, x + 1] = np.clip(temp[y, x + 1] + err * (7 / 16), 0, 255)
            if y + 1 < h:
                if x > 0:
                    temp[y + 1, x - 1] = np.clip(temp[y + 1, x - 1] + err * (3 / 16), 0, 255)
                temp[y + 1, x] = np.clip(temp[y + 1, x] + err * (5 / 16), 0, 255)
                if x + 1 < w:
                    temp[y + 1, x + 1] = np.clip(temp[y + 1, x + 1] + err * (1 / 16), 0, 255)

    return idx8

def image_to_palette_indices(arr: np.ndarray) -> np.ndarray:
    h, w, _ = arr.shape
    idx8 = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            r, g, b = arr[y, x]
            idx8[y, x] = nearest_palette_index(float(r), float(g), float(b))
    return idx8

def pack_indices(idx8: np.ndarray) -> bytes:
    out = bytearray((W * H) // 2)
    oi = 0
    for y in range(H):
        row = idx8[y]
        for x in range(W - 1, -1, -2):  # rowMajorXFlip
            hi = INDEX_TO_NIBBLE[int(row[x])] & 0x0F
            lo = INDEX_TO_NIBBLE[int(row[x - 1])] & 0x0F if x - 1 >= 0 else hi
            out[oi] = (hi << 4) | lo
            oi += 1
    return bytes(out)

def unpack_packed_indices(packed: bytes) -> np.ndarray:
    idx8 = np.zeros((H, W), dtype=np.uint8)
    bi = 0
    for y in range(H):
        row = idx8[y]
        for x in range(W - 1, -1, -2):  # rowMajorXFlip
            b = packed[bi]
            bi += 1
            hi = (b >> 4) & 0x0F
            lo = b & 0x0F
            row[x] = NIBBLE_TO_INDEX.get(hi, 0)
            if x - 1 >= 0:
                row[x - 1] = NIBBLE_TO_INDEX.get(lo, 0)
    return idx8

def indices_to_image(idx8: np.ndarray) -> Image.Image:
    palette = np.array([[r, g, b] for _, r, g, b in PALETTE], dtype=np.uint8)
    rgb = palette[idx8]
    return Image.fromarray(rgb, "RGB")

def _fit_center_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    src_w, src_h = img.size
    if src_w == 0 or src_h == 0:
        return img.resize((target_w, target_h), Image.LANCZOS)
    scale = max(target_w / src_w, target_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    left = max(0, (new_w - target_w) // 2)
    top = max(0, (new_h - target_h) // 2)
    right = left + target_w
    bottom = top + target_h
    return resized.crop((left, top, right, bottom))

def _enhance_for_spectra6(img: Image.Image, contrast: float, color: float, brightness: float, sharpness: float, gamma: float) -> Image.Image:
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if color != 1.0:
        img = ImageEnhance.Color(img).enhance(color)
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if sharpness != 1.0:
        img = ImageEnhance.Sharpness(img).enhance(sharpness)
    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        lut = [int(pow(i / 255.0, inv_gamma) * 255.0 + 0.5) for i in range(256)]
        img = img.point(lut * 3)
    return img

def jpeg_to_packed(path, dither_method: str, dither_strength: float, rotate_deg: int,
                   enhance: bool, contrast: float, color: float, brightness: float, sharpness: float, gamma: float):
    img = Image.open(path).convert("RGB")
    if rotate_deg:
        img = img.rotate(rotate_deg, expand=True)
    img = _fit_center_crop(img, W, H)
    if enhance:
        img = _enhance_for_spectra6(img, contrast, color, brightness, sharpness, gamma)
    arr = np.asarray(img, dtype=np.float32)

    if dither_method == "none":
        idx8 = image_to_palette_indices(arr)
    elif dither_method == "floyd":
        idx8 = floyd_steinberg_dither_indices(arr, dither_strength)
    else:
        raise ValueError(dither_method)

    return pack_indices(idx8)

def list_images(images_dir):
    out = []
    for name in sorted(os.listdir(images_dir)):
        if name.lower().endswith((".jpg", ".jpeg")):
            out.append(name)
    return out[:8]  # Firmware kann max. 8 URLs
    # (siehe version_info_t.image_urls[8])

def _parse_smb_share(share):
    if share.startswith("smb://"):
        share = share[len("smb://"):]
    if share.startswith("//"):
        share = share[2:]
    parts = share.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("invalid SMB share, expected //server/share or smb://server/share")
    return parts[0], parts[1]

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        msg = "%s - - [%s] %s" % (
            self.client_address[0],
            self.log_date_time_string(),
            format % args,
        )
        print(msg)

    def do_GET(self):
        start_time = time.time()
        parsed = urlparse(self.path)
        print(f"[debug] incoming GET {parsed.path} from {self.client_address[0]}")
        if parsed.path == "/version.json":
            print("[debug] building version.json response")
            images = self.server.list_images()
            self.server._image_list = images
            reset_index = bool(self.server.reset_index_once)
            if reset_index:
                self.server.reset_index_once = False
            payload = {
                "settings": {
                    "current_version": self.server.version,
                    "image_change_interval_sec": self.server.interval_sec,
                    "reset_index": reset_index,
                    "heartbeat_interval_sec": self.server.heartbeat_interval_sec,
                },
                "images": [{"url": f"{self.server.base_url}/images/{n}"} for n in images],
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            elapsed_ms = (time.time() - start_time) * 1000.0
            print(f"[debug] sent version.json ({len(body)} bytes, {len(images)} images) in {elapsed_ms:.1f} ms")
            return

        if parsed.path.startswith("/images/"):
            name = parsed.path[len("/images/"):]
            if self.server.smb_direct:
                is_found = name in self.server.list_images()
            else:
                fs_path = os.path.join(self.server.images_dir, name)
                is_found = os.path.isfile(fs_path)
            if is_found:
                print(f"[debug] preparing image payload for {name}")
                data = self.server.get_packed(name if self.server.smb_direct else fs_path)
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                    elapsed_ms = (time.time() - start_time) * 1000.0
                    print(f"[debug] sent image bytes for {name} ({len(data)} bytes) in {elapsed_ms:.1f} ms")
                    self.server._last_requested = name
                    self.server.precompute_next_image(name)
                except (BrokenPipeError, ConnectionResetError):
                    # Client disconnected mid-response; ignore to avoid noisy traceback.
                    elapsed_ms = (time.time() - start_time) * 1000.0
                    print(f"[debug] client disconnected while sending {name} after {elapsed_ms:.1f} ms")
                    return
                return

        self.send_response(404)
        self.end_headers()
        elapsed_ms = (time.time() - start_time) * 1000.0
        print(f"[debug] 404 for {parsed.path} after {elapsed_ms:.1f} ms")

    def do_POST(self):
        start_time = time.time()
        parsed = urlparse(self.path)
        print(f"[debug] incoming POST {parsed.path} from {self.client_address[0]}")
        if parsed.path == "/heartbeat":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                payload = json.loads(raw.decode("utf-8")) if raw else {}
            except (ValueError, UnicodeDecodeError):
                self.send_response(400)
                self.end_headers()
                elapsed_ms = (time.time() - start_time) * 1000.0
                print(f"[debug] heartbeat invalid JSON after {elapsed_ms:.1f} ms")
                return
            self.server.last_heartbeat = payload
            self.server.last_heartbeat_at = time.time()
            if payload:
                fields = [
                    "version",
                    "interval_sec",
                    "heartbeat_interval_sec",
                    "image_count",
                    "current_index",
                    "next_index",
                    "current_url",
                    "next_url",
                ]
                parts = [f"{k}={payload.get(k)}" for k in fields]
                print("[debug] heartbeat payload: " + ", ".join(parts))
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"OK")
            elapsed_ms = (time.time() - start_time) * 1000.0
            print(f"[debug] heartbeat OK in {elapsed_ms:.1f} ms")
            return

        self.send_response(404)
        self.end_headers()
        elapsed_ms = (time.time() - start_time) * 1000.0
        print(f"[debug] 404 for {parsed.path} after {elapsed_ms:.1f} ms")

class Server(HTTPServer):
    def __init__(self, addr, handler, images_dir, base_url, version, interval_sec, heartbeat_interval_sec,
                 dither_method, dither_strength, rotate_deg, enhance, contrast, color, brightness, sharpness, gamma,
                 debug_preview_dir, scan_interval_sec, smb_direct, smb_share, smb_user, smb_pass, smb_domain,
                 smb_guest, smb_dir, reset_index):
        super().__init__(addr, handler)
        self.images_dir = images_dir
        self.base_url = base_url.rstrip("/")
        self.version = version
        self.interval_sec = interval_sec
        self.heartbeat_interval_sec = heartbeat_interval_sec
        self.dither_method = dither_method
        self.dither_strength = dither_strength
        self.rotate_deg = rotate_deg
        self.enhance = enhance
        self.contrast = contrast
        self.color = color
        self.brightness = brightness
        self.sharpness = sharpness
        self.gamma = gamma
        self.debug_preview_dir = debug_preview_dir
        self.scan_interval_sec = scan_interval_sec
        self.reset_index_once = reset_index
        self.smb_direct = smb_direct
        self.smb_share = smb_share
        self.smb_user = smb_user
        self.smb_pass = smb_pass
        self.smb_domain = smb_domain
        self.smb_guest = smb_guest
        smb_dir = smb_dir or ""
        if smb_dir in (".", "./"):
            smb_dir = ""
        self.smb_dir = smb_dir.strip("/")
        self._cache = {}
        self._last_requested = None
        self._image_list = []
        self.last_heartbeat = None
        self.last_heartbeat_at = None
        self._smb_conn = None
        self._smb_server = None
        self._smb_share_name = None
        self._smb_mtime = {}
        self._smb_use_netbios = False
        if self.smb_direct and self.smb_share:
            self._smb_server, self._smb_share_name = _parse_smb_share(self.smb_share)

    def _reset_smb_conn(self, use_netbios: bool):
        self._smb_conn = None
        self._smb_use_netbios = use_netbios

    def _get_smb_conn(self):
        if self._smb_conn:
            return self._smb_conn
        if not SMBConnection:
            raise RuntimeError("pysmb is not installed; cannot use SMB direct mode")
        user = "guest" if self.smb_guest else (self.smb_user or "")
        password = "" if self.smb_guest else (self.smb_pass or "")
        if self._smb_use_netbios:
            self._smb_conn = SMBConnection(
                user,
                password,
                socket.gethostname(),
                self._smb_server,
                domain=self.smb_domain or "",
                use_ntlm_v2=True,
                is_direct_tcp=False,
            )
            if not self._smb_conn.connect(self._smb_server, 139):
                self._smb_conn = None
                raise RuntimeError("failed to connect to SMB server (netbios)")
        else:
            self._smb_conn = SMBConnection(
                user,
                password,
                socket.gethostname(),
                self._smb_server,
                domain=self.smb_domain or "",
                use_ntlm_v2=True,
                is_direct_tcp=True,
            )
            if not self._smb_conn.connect(self._smb_server, 445):
                self._smb_conn = None
                self._smb_use_netbios = True
                return self._get_smb_conn()
        return self._smb_conn

    def _smb_list_images(self):
        conn = self._get_smb_conn()
        path = f"/{self.smb_dir}" if self.smb_dir else "/"
        try:
            entries = conn.listPath(self._smb_share_name, path)
        except Exception as exc:
            if ProtocolError and isinstance(exc, ProtocolError) and not self._smb_use_netbios:
                print("[debug] SMB2 protocol error, retrying with NetBIOS (SMB1)")
                self._reset_smb_conn(use_netbios=True)
                conn = self._get_smb_conn()
                entries = conn.listPath(self._smb_share_name, path)
            elif SMBTimeout and isinstance(exc, SMBTimeout):
                print("[debug] SMB timeout, reconnecting and retrying listPath")
                self._reset_smb_conn(use_netbios=self._smb_use_netbios)
                conn = self._get_smb_conn()
                entries = conn.listPath(self._smb_share_name, path)
            elif NotConnectedError and isinstance(exc, NotConnectedError):
                print("[debug] SMB not connected, reconnecting and retrying listPath")
                self._reset_smb_conn(use_netbios=self._smb_use_netbios)
                conn = self._get_smb_conn()
                entries = conn.listPath(self._smb_share_name, path)
            elif isinstance(exc, BrokenPipeError):
                print("[debug] SMB broken pipe, reconnecting and retrying listPath")
                self._reset_smb_conn(use_netbios=self._smb_use_netbios)
                conn = self._get_smb_conn()
                entries = conn.listPath(self._smb_share_name, path)
            else:
                raise
        out = []
        mtimes = {}
        for entry in entries:
            if entry.isDirectory:
                continue
            name = entry.filename
            if name.lower().endswith((".jpg", ".jpeg")):
                out.append(name)
                mtimes[name] = entry.last_write_time
        out = sorted(out)[:8]
        self._smb_mtime = {k: mtimes[k] for k in out}
        return out

    def list_images(self):
        if self.smb_direct:
            return self._smb_list_images()
        return list_images(self.images_dir)

    def _refresh_image_list(self):
        self._image_list = self.list_images()
        return self._image_list

    def _next_image_name(self, current_name):
        images = self._image_list or self._refresh_image_list()
        if not images:
            return None
        if not current_name or current_name not in images:
            return images[0]
        idx = images.index(current_name)
        return images[(idx + 1) % len(images)]

    def _prune_cache_keep(self, keep_names):
        if self.smb_direct:
            keep_paths = set(keep_names)
        else:
            keep_paths = {os.path.join(self.images_dir, name) for name in keep_names}
        newest = {}
        for key in list(self._cache.keys()):
            path = key[0]
            mtime = key[1]
            if path in keep_paths:
                if path not in newest or mtime > newest[path][1]:
                    newest[path] = (key, mtime)
        removed = 0
        for key in list(self._cache.keys()):
            path = key[0]
            if path not in keep_paths:
                del self._cache[key]
                removed += 1
                continue
            if newest.get(path, (None, None))[0] != key:
                del self._cache[key]
                removed += 1
        if removed:
            print(f"[debug] pruned {removed} cached entries")

    def precompute_next_image(self, current_name=None):
        images = self._refresh_image_list()
        if not images:
            print("[debug] no images to precompute")
            self._prune_cache_keep([])
            return
        next_name = self._next_image_name(current_name)
        if not next_name:
            return
        print(f"[debug] precomputing next image: {next_name}")
        try:
            self.get_packed(next_name if self.smb_direct else os.path.join(self.images_dir, next_name))
        except Exception as exc:
            print(f"[debug] failed to precompute {next_name}: {exc}")
        keep = [n for n in [current_name, next_name] if n]
        self._prune_cache_keep(keep)

    def start_image_watcher(self):
        if self.scan_interval_sec <= 0:
            return

        def _loop():
            while True:
                try:
                    self.precompute_next_image(self._last_requested)
                except Exception as exc:
                    print(f"[debug] image scan failed: {exc}")
                time.sleep(self.scan_interval_sec)

        thread = threading.Thread(target=_loop, name="image-watcher", daemon=True)
        thread.start()

    def _write_debug_preview(self, path, packed):
        if not self.debug_preview_dir:
            return
        os.makedirs(self.debug_preview_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(path))[0]
        strength = f"{self.dither_strength:.2f}".replace(".", "_")
        out_name = f"{base}.dither-{self.dither_method}-{strength}.preview.png"
        out_path = os.path.join(self.debug_preview_dir, out_name)
        idx8 = unpack_packed_indices(packed)
        preview = indices_to_image(idx8)
        preview.save(out_path)

    def get_packed(self, path):
        if self.smb_direct:
            name = path
            conn = self._get_smb_conn()
            mtime = self._smb_mtime.get(name)
            if mtime is None:
                smb_path = f"/{self.smb_dir}/{name}" if self.smb_dir else f"/{name}"
                try:
                    attr = conn.getAttributes(self._smb_share_name, smb_path)
                except Exception as exc:
                    if ProtocolError and isinstance(exc, ProtocolError) and not self._smb_use_netbios:
                        print("[debug] SMB2 protocol error, retrying with NetBIOS (SMB1)")
                        self._reset_smb_conn(use_netbios=True)
                        conn = self._get_smb_conn()
                        attr = conn.getAttributes(self._smb_share_name, smb_path)
                    elif SMBTimeout and isinstance(exc, SMBTimeout):
                        print("[debug] SMB timeout, reconnecting and retrying getAttributes")
                        self._reset_smb_conn(use_netbios=self._smb_use_netbios)
                        conn = self._get_smb_conn()
                        attr = conn.getAttributes(self._smb_share_name, smb_path)
                    elif NotConnectedError and isinstance(exc, NotConnectedError):
                        print("[debug] SMB not connected, reconnecting and retrying getAttributes")
                        self._reset_smb_conn(use_netbios=self._smb_use_netbios)
                        conn = self._get_smb_conn()
                        attr = conn.getAttributes(self._smb_share_name, smb_path)
                    elif isinstance(exc, BrokenPipeError):
                        print("[debug] SMB broken pipe, reconnecting and retrying getAttributes")
                        self._reset_smb_conn(use_netbios=self._smb_use_netbios)
                        conn = self._get_smb_conn()
                        attr = conn.getAttributes(self._smb_share_name, smb_path)
                    else:
                        raise
                mtime = attr.last_write_time
            key = (name, mtime, self.dither_method, self.dither_strength, self.rotate_deg,
                   self.enhance, self.contrast, self.color, self.brightness, self.sharpness, self.gamma, "smb")
        else:
            mtime = os.path.getmtime(path)
            key = (path, mtime, self.dither_method, self.dither_strength, self.rotate_deg,
                   self.enhance, self.contrast, self.color, self.brightness, self.sharpness, self.gamma)
        cached = self._cache.get(key)
        if cached:
            print(f"[debug] cache hit for {os.path.basename(path)}")
            return cached
        print(f"[debug] cache miss for {os.path.basename(path)}; converting")
        if self.smb_direct:
            name = path
            smb_path = f"/{self.smb_dir}/{name}" if self.smb_dir else f"/{name}"
            buf = io.BytesIO()
            try:
                conn.retrieveFile(self._smb_share_name, smb_path, buf)
            except Exception as exc:
                if ProtocolError and isinstance(exc, ProtocolError) and not self._smb_use_netbios:
                    print("[debug] SMB2 protocol error, retrying with NetBIOS (SMB1)")
                    self._reset_smb_conn(use_netbios=True)
                    conn = self._get_smb_conn()
                    buf = io.BytesIO()
                    conn.retrieveFile(self._smb_share_name, smb_path, buf)
                elif SMBTimeout and isinstance(exc, SMBTimeout):
                    print("[debug] SMB timeout, reconnecting and retrying retrieveFile")
                    self._reset_smb_conn(use_netbios=self._smb_use_netbios)
                    conn = self._get_smb_conn()
                    buf = io.BytesIO()
                    conn.retrieveFile(self._smb_share_name, smb_path, buf)
                elif NotConnectedError and isinstance(exc, NotConnectedError):
                    print("[debug] SMB not connected, reconnecting and retrying retrieveFile")
                    self._reset_smb_conn(use_netbios=self._smb_use_netbios)
                    conn = self._get_smb_conn()
                    buf = io.BytesIO()
                    conn.retrieveFile(self._smb_share_name, smb_path, buf)
                elif isinstance(exc, BrokenPipeError):
                    print("[debug] SMB broken pipe, reconnecting and retrying retrieveFile")
                    self._reset_smb_conn(use_netbios=self._smb_use_netbios)
                    conn = self._get_smb_conn()
                    buf = io.BytesIO()
                    conn.retrieveFile(self._smb_share_name, smb_path, buf)
                else:
                    raise
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
            if self.rotate_deg:
                img = img.rotate(self.rotate_deg, expand=True)
            img = _fit_center_crop(img, W, H)
            if self.enhance:
                img = _enhance_for_spectra6(img, self.contrast, self.color, self.brightness, self.sharpness, self.gamma)
            arr = np.asarray(img, dtype=np.float32)
            if self.dither_method == "none":
                idx8 = image_to_palette_indices(arr)
            elif self.dither_method == "floyd":
                idx8 = floyd_steinberg_dither_indices(arr, self.dither_strength)
            else:
                raise ValueError(self.dither_method)
            data = pack_indices(idx8)
        else:
            data = jpeg_to_packed(path, self.dither_method, self.dither_strength, self.rotate_deg,
                                  self.enhance, self.contrast, self.color, self.brightness, self.sharpness, self.gamma)
        self._write_debug_preview(path, data)
        self._cache[key] = data
        return data

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=80)
    parser.add_argument("--base-url", required=True, help="z.B. http://192.168.1.182:8000")
    parser.add_argument("--images-dir", default="images")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--interval-sec", type=int, default=30)
    parser.add_argument("--heartbeat-interval-sec", type=int, default=0,
                        help="include heartbeat_interval_sec in version.json (0 disables)")
    parser.add_argument("--dither", choices=["none", "floyd"], default="floyd")
    parser.add_argument("--dither-strength", type=float, default=1.0)
    parser.add_argument("--rotate-deg", type=int, default=0,
                        help="rotate input images by N degrees (e.g. 90, 180, 270)")
    parser.add_argument("--enhance", action="store_true",
                        help="apply Spectra 6 preprocessing (contrast/gamma/color/etc)")
    parser.add_argument("--enhance-contrast", type=float, default=1.15)
    parser.add_argument("--enhance-color", type=float, default=1.05)
    parser.add_argument("--enhance-brightness", type=float, default=1.02)
    parser.add_argument("--enhance-sharpness", type=float, default=1.1)
    parser.add_argument("--enhance-gamma", type=float, default=1.1)
    parser.add_argument("--debug-preview-dir", default=None)
    parser.add_argument("--scan-interval-sec", type=int, default=30,
                        help="periodically scan images dir and precompute packed data (0 disables)")
    parser.add_argument("--reset-index", action="store_true",
                        help="set reset_index=true in version.json (one-time on client)")
    parser.add_argument("--smb-share", default=None,
                        help="SMB share to mount (e.g. //server/share or smb://server/share)")
    parser.add_argument("--smb-mount", default=None,
                        help="mount point for SMB share (e.g. /Volumes/Bilderrahmen)")
    parser.add_argument("--smb-user", default=None)
    parser.add_argument("--smb-pass", default=None)
    parser.add_argument("--smb-domain", default=None,
                        help="optional SMB domain (Linux only)")
    parser.add_argument("--smb-guest", action="store_true")
    parser.add_argument("--smb-direct", action="store_true",
                        help="read images directly over SMB without mounting")
    parser.add_argument("--smb-dir", default="",
                        help="optional subdirectory within the SMB share")
    args = parser.parse_args()

    if args.smb_share and args.smb_direct:
        if not SMBConnection:
            raise SystemExit("pysmb is required for SMB direct mode (pip install pysmb)")
    elif args.smb_share:
        if not args.smb_mount:
            raise SystemExit("--smb-mount is required when --smb-share is set (or use --smb-direct)")
        os.makedirs(args.smb_mount, exist_ok=True)
        share = args.smb_share
        if share.startswith("smb://"):
            share = "//" + share[len("smb://"):]

        system = platform.system().lower()
        if "darwin" in system:
            user = "guest" if args.smb_guest else (args.smb_user or "")
            if args.smb_guest:
                auth = "guest:@"
            elif args.smb_user and args.smb_pass is not None:
                auth = f"{args.smb_user}:{args.smb_pass}@"
            elif args.smb_user:
                auth = f"{args.smb_user}@"
            else:
                auth = ""
            if share.startswith("//"):
                share = share[2:]
            smb_url = f"//{auth}{share}"
            cmd = ["mount_smbfs", smb_url, args.smb_mount]
        else:
            opts = []
            if args.smb_guest:
                opts.append("guest")
            if args.smb_user:
                opts.append(f"username={args.smb_user}")
            if args.smb_pass is not None:
                opts.append(f"password={args.smb_pass}")
            if args.smb_domain:
                opts.append(f"domain={args.smb_domain}")
            cmd = ["mount", "-t", "cifs", share, args.smb_mount]
            if opts:
                cmd += ["-o", ",".join(opts)]

        print(f"[debug] mounting SMB share {args.smb_share} -> {args.smb_mount}")
        subprocess.run(cmd, check=True)

        if args.images_dir == "images":
            args.images_dir = args.smb_mount

    if not args.smb_direct and not os.path.isdir(args.images_dir):
        raise SystemExit(f"images dir not found: {args.images_dir}")

    httpd = Server((args.host, args.port), Handler,
                   args.images_dir, args.base_url, args.version, args.interval_sec, args.heartbeat_interval_sec,
                   args.dither, args.dither_strength, args.rotate_deg,
                   args.enhance, args.enhance_contrast, args.enhance_color, args.enhance_brightness,
                   args.enhance_sharpness, args.enhance_gamma, args.debug_preview_dir,
                   args.scan_interval_sec, args.smb_direct, args.smb_share, args.smb_user,
                   args.smb_pass, args.smb_domain, args.smb_guest, args.smb_dir,
                   args.reset_index)
    httpd.precompute_next_image(None)
    httpd.start_image_watcher()
    print(
        "[debug] enhance defaults: "
        f"contrast={args.enhance_contrast}, "
        f"color={args.enhance_color}, "
        f"brightness={args.enhance_brightness}, "
        f"sharpness={args.enhance_sharpness}, "
        f"gamma={args.enhance_gamma} "
        f"(enabled={args.enhance})"
    )
    print(f"Serving on {args.host}:{args.port}")
    print(f"version.json -> {httpd.base_url}/version.json")
    httpd.serve_forever()

if __name__ == "__main__":
    main()
