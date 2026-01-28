# frame-server

Kleiner HTTP-Server fuer den Bilderrahmen. Er liefert `version.json` und die Bilder
als gepackte Bytes (Palette + Dithering) aus. Neue Bilder werden beim Aufruf von
`/version.json` automatisch in die Liste aufgenommen (max. 8 Dateien).

## Usage

### Lokaler Ordner

```bash
python3 updater.py \
  --base-url http://192.168.1.154:80 \
  --images-dir images \
  --version 2 \
  --interval-sec 15
```

### SMB direkt (ohne Mount)

```bash
python3 updater.py \
  --base-url http://192.168.1.154:80 \
  --smb-share //NAS-Speicher.fritz.box/Bilderrahmen \
  --smb-direct \
  --smb-user anni \
  --smb-pass <PASSWORT> \
  --smb-domain WORKGROUP
```

### Relevante Optionen

- `--dither` / `--dither-strength`: Dithering einstellen (`none` oder `floyd`).
- `--rotate-deg`: Rotation vor dem Resizing (z.B. `90`, `180`).
- `--mirror-horizontal`: Bild horizontal spiegeln.
- `--enhance` + `--enhance-*`: Vorverarbeitung fuer Spectra-6 (Kontrast/Gamma/etc).
- `--scan-interval-sec`: Hintergrund-Scan fuer das Vorrechnen (0 = aus).
- `--reset-index`: setzt in der ersten `version.json` einmalig `reset_index=true`.
- `--heartbeat-interval-sec`: schreibt `heartbeat_interval_sec` in `version.json` (0 = aus).
- `--smb-direct`: SMB-Share direkt lesen (kein Mount).

## Beispiel (unten)

```bash
python3 updater.py \
  --base-url http://192.168.1.154:80 \
  --images-dir images \
  --version 2 \
  --interval-sec 15 \
  --rotate-deg 90 \
  --mirror-horizontal \
  --enhance \
  --enhance-contrast 1.2 \
  --enhance-color 1.1 \
  --enhance-brightness 1.05 \
  --enhance-sharpness 1.15 \
  --enhance-gamma 1.08 \
  --heartbeat-interval-sec 60 \
  --scan-interval-sec 0 \
  --reset-index
```


python3 updater.py --base-url http://...:80 --version 2 --interval-sec 1800 --rotate-deg 90   --enhance --enhance-contrast 1.2 --enhance-color 1.1 --enhance-brightness 1.05 --enhance-sharpness 1.15 --enhance-gamma 1.08 --scan-interval-sec 0 --smb-share //.../Ordner --smb-direct --smb-user USER --smb-pass PASS  --smb-domain WORKGROUP --reset-index


docker run --rm \
  -p 80:80 \
  -v "$PWD/previews:/app/previews" \
  ghcr.io/svolgmann/frame-server:sha-b38581e \
  --base-url http://IP:80 \
  --version 2 \
  --interval-sec 1800 \
  --debug-preview-dir previews \
  --rotate-deg 90 \
  --enhance \
  --enhance-contrast 1.2 \
  --enhance-color 1.1 \
  --enhance-brightness 1.05 \
  --enhance-sharpness 1.15 \
  --enhance-gamma 1.08 \
  --scan-interval-sec 0 \
  --smb-share //PFAD/ORDNER \
  --smb-direct \
  --smb-user USER \
  --smb-pass PASS \
  --smb-domain WORKGROUP \
  --reset-index
