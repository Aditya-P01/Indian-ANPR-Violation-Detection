# Traffic Violation Detection with License Plate Recognition (ANPR)

Detects vehicles at a traffic junction, tracks them across frames, flags **red-light jumping,
over-speeding and wrong-way driving**, and reads their **number plates** with OCR.
Outputs an annotated demo video, a violations log (CSV + snapshots) and an accuracy report.

**Tech stack:** Python · OpenCV · YOLOv8 (Ultralytics) · ByteTrack · EasyOCR · Pandas

```
video ─► YOLOv8 (car/bike/bus/truck) ─► ByteTrack IDs ─► rule engine ─► violations.csv + snapshots
                                              │
                                              └► plate detector (YOLOv8) ─► EasyOCR ─► format fix ─► multi-frame vote
```

## Features
- **Vehicle detection + tracking** - pretrained YOLOv8n (COCO) with ByteTrack, so every vehicle keeps one ID.
- **Plate localisation** - a YOLOv8 plate detector you train (`train_plate_detector.py`); if none is
  present it falls back to EasyOCR's text detector, so the project runs out of the box.
- **OCR tuned for plates** - upscaling, CLAHE contrast, automatic deskew of tilted plates
  (Hough-line angle), retry on a binarised image when the first read fails, 1- and 2-line plates,
  alphanumeric allow-list.
- **Plate-format correction** - fixes look-alike errors (O↔0, I↔1, S↔5, B↔8 ...) using the Indian
  plate pattern `SS 00 AA 0000` (switch to `format: none` for other countries).
- **Multi-frame voting** - each vehicle is read many times; reads are weighted by OCR confidence,
  format validity and plate size, then combined **character by character**, so several partly-wrong
  reads can still produce the fully correct plate.
- **Violations**
  - *Red light*: vehicle crosses the stop line while the signal is RED (manual schedule, timed cycle,
    or automatic lamp-colour detection).
  - *Over-speed*: time between two lines a known distance apart → km/h.
  - *Wrong way*: crosses the speed lines in the reverse order.
- **Evidence** - annotated video, per-violation snapshots, best plate crop per vehicle, CSV logs.
- **Evaluation** - plate exact-match / character accuracy, violation precision / recall / F1,
  detector mAP, all written to `outputs/accuracy_report.md`.

## Results

Demo: Mumbai junction, 4K handheld footage (Pexels), laptop CPU only.
Video: [`demo/mumbai_violations_demo.mp4`](demo/mumbai_violations_demo.mp4) · full write-up: [`REPORT.md`](REPORT.md)

| | Result |
|---|---|
| Plate detector (YOLOv8n, held-out test set, 882 images) | **96.6 % precision, 85.7 % recall, 93.2 % mAP@0.5** |
| Red-light violations (checked by eye) | **3 / 3 detected, 0 false alarms** |
| Plates read exactly right (12 readable plates) | 3 / 12 (25 %) |
| Plate precision (reported plates that are real) | 50 % (OCR-only baseline: 43 %) |
| Second clip, Bhopal (plain fonts): plates read exactly | 2 / 3, 93 % character accuracy |
| Speed | 1.8 FPS on 4K input, CPU only |

The plate detector is strong; most remaining errors come from OCR on decorative taxi fonts,
small plates and motion blur. See the report for the error analysis and next steps.

## Project structure
```
anpr/
  geometry.py        virtual lines + crossing test
  plate_text.py      plate clean-up, Indian-format correction, voting
  ocr.py             EasyOCR wrapper + pre-processing
  plate_detector.py  YOLO plate detector / OCR fallback
  signal.py          traffic-signal state
  violations.py      track state + rule engine
  draw.py            video overlay
main.py              run the pipeline on a video
evaluate.py          accuracy report
train_plate_detector.py
tools/pick_points.py         click stop line / speed lines on your video
tools/convert_coco_to_yolo.py  convert a COCO plate dataset (Roboflow / Hugging Face) to YOLO format
tools/convert_voc_to_yolo.py convert a VOC plate dataset to YOLO format
tools/make_synthetic_video.py  synthetic clip for a quick smoke test
tests/test_core.py   unit tests
config.yaml          all settings
```

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```
The YOLOv8n and EasyOCR weights download automatically on first run.

## Quick start
1. **Get a traffic video** - record a junction on your phone (from a footbridge, so plates are visible),
   or download a free clip from Pexels / Pixabay (search "traffic India", "traffic junction").
   Save it as `data/traffic.mp4`.
2. **Mark the lines** on your video:
   ```bash
   python tools/pick_points.py --video data/traffic.mp4
   ```
   Click: stop line (2 points) → speed line A → speed line B → (optional) traffic-light box.
   Paste the printed values into `config.yaml`, and set `distance_m` to the real distance between A and B
   (count lane markings: in India a broken line segment + gap is usually ~9 m, or measure on Google Maps).
3. **Set the signal** in `config.yaml` → `rules.red_light.signal`:
   `manual` with the seconds when the light is red, `timed`, or `auto` if the lamp is visible.
4. **Run**
   ```bash
   python main.py                 # add --display for a live window, --max-frames 300 for a quick test
   ```
5. **Check results** in `outputs/`: `annotated.mp4`, `violations.csv`, `vehicles.csv`,
   `violations/*.jpg`, `plates/*.jpg`.

Smoke test without any video:
```bash
python tools/make_synthetic_video.py --out data/synthetic.mp4
python main.py --source data/synthetic.mp4
```

## Better accuracy: train the plate detector
The detector in `weights/plate_yolov8n.pt` was trained on the
[keremberke/license-plate-object-detection](https://huggingface.co/datasets/keremberke/license-plate-object-detection)
dataset (Roboflow, CC BY 4.0: 6,176 train / 1,765 val / 882 test images, one class `license_plate`).
To reproduce:
```bash
# download data/train.zip, valid.zip, test.zip from the dataset page and unzip each into
# datasets/hf_raw/train, datasets/hf_raw/valid, datasets/hf_raw/test
python tools/convert_coco_to_yolo.py --src datasets/hf_raw --dst datasets/plates
python train_plate_detector.py --data datasets/plates/data.yaml --epochs 20 --imgsz 416 --fraction 0.5 --device cpu
```
That CPU setting (half the train set, 416 px) takes about 2-3 hours on a 16-core laptop CPU.
On Google Colab (free GPU) use the full set: `--epochs 50 --imgsz 640 --fraction 1.0` (~20 min),
then set `plate_imgsz: 640` in `config.yaml`.
Weights are picked up automatically by `main.py`; test-set precision / recall / mAP are saved to
`weights/plate_metrics.json` and included in the accuracy report.
If a run is interrupted, continue it with `python train_plate_detector.py --data datasets/plates/data.yaml --resume`.

Other datasets: `tools/convert_voc_to_yolo.py` handles Pascal-VOC sets such as Kaggle
"Car License Plate Detection". Adding some **Indian** plate images (e.g. Roboflow Universe, search
"indian number plate") and fine-tuning from `weights/plate_yolov8n.pt` improves results on Indian roads.

**Bigger vehicle model:** set `models.vehicle: yolov8s.pt` in `config.yaml` for better detection of
small / distant vehicles (about 2-3x slower on CPU).

## Accuracy report
1. Run `main.py`, then look at `outputs/annotated.mp4` and `outputs/plates/`.
2. Copy `data/ground_truth_template.csv` to `data/ground_truth.csv` and write the **real** plate of every
   readable vehicle, plus its violation if it actually committed one.
3. ```bash
   python evaluate.py --gt data/ground_truth.csv
   # optional: OCR accuracy on labelled plate images
   python evaluate.py --crops data/plate_crops --labels data/plate_crops/labels.csv
   ```
4. Open `outputs/accuracy_report.md` - it contains plate recognition accuracy, character accuracy,
   violation precision/recall/F1, false positives/negatives, detector mAP and limitations.

## Tests
```bash
python -m unittest discover tests -v
```

## Limitations
- Plates narrower than ~30 px, motion blur, glare and night footage hurt OCR.
- Speed depends on the measured distance between lines (expect a few km/h error).
- Red-light detection needs the signal state; `auto` works only if the lamp is visible in a fixed spot.
- CPU processing runs at a few FPS - fine for recorded video, not real-time.
