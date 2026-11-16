# Traffic Violation Detection with License Plate Recognition

**Computer Vision Project - Assignment 2 (Recognise license plates using OCR)**

## 1. Objective

Detect vehicles that violate traffic rules at a junction from video footage, and identify them by
reading their number plates with OCR.

Deliverables covered:

| Deliverable | Where |
|---|---|
| Demo video showing detection of violations | `demo/mumbai_violations_demo.mp4` (+ `demo/bhopal_plates_demo.mp4`) |
| Accuracy report on detection | Sections 4 and 6 of this report + `demo/*_accuracy_report.md` |
| Violation evidence | `demo/mumbai_violations.csv` + one snapshot JPG per violation in `demo/` |
| Working code | `main.py`, `anpr/`, `train_plate_detector.py`, `evaluate.py` |

## 2. Tech stack

| Part | Tool |
|---|---|
| Language | Python 3 |
| Image / video processing | OpenCV |
| Vehicle detection | YOLOv8n (Ultralytics), pretrained on COCO (car, motorcycle, bus, truck) |
| Multi-object tracking | ByteTrack |
| Plate detection | YOLOv8n fine-tuned on a licence-plate dataset (trained in this project) |
| OCR | EasyOCR (CRAFT text detector + CRNN recogniser) |
| Logs and evaluation | Pandas, CSV, Markdown report |

## 3. System design

```
 video frame (4K)
     │  resize to 1280 px
     ▼
 YOLOv8n vehicle detector ──► ByteTrack (one ID per vehicle)
     │                              │
     │                              ├─► rule engine (stop line, speed lines, signal state)
     │                              │        └─► violations.csv + snapshot JPGs
     ▼                              │
 crop vehicle from the ORIGINAL 4K frame
     ▼
 YOLOv8n plate detector ─► plate crop ─► deskew + contrast ─► EasyOCR
     ▼
 Indian-format correction (O↔0, I↔1, S↔5, B↔8 by position)
     ▼
 multi-frame, character-level voting per vehicle ─► final plate
```

### 3.1 Vehicle detection and tracking
Every frame is resized to 1280 px wide and passed to YOLOv8n with only the four vehicle classes
enabled. ByteTrack links detections across frames so each vehicle keeps a single ID. The
bottom-centre of the box (where the wheels touch the road) is used as the vehicle's position for
all line-crossing rules.

### 3.2 Plate detection
A YOLOv8n model was fine-tuned on the open *keremberke/license-plate-object-detection* dataset
(Roboflow, CC BY 4.0; 6,176 train / 1,765 validation / 882 test images, one class). Because the
plate is searched only inside each vehicle crop, false plates on shop signs and road signs are
largely avoided. Horizontal flipping was switched off during training, because mirrored text is
not a realistic plate.

The vehicle crop is taken from the original 4K frame (not the resized one), which gives the OCR
about three times more pixels per character.

### 3.3 OCR and plate-text correction
1. **Pre-processing:** grayscale, upscale to ~96 px height, bilateral filter, CLAHE contrast.
2. **Deskew:** the long near-horizontal edges of the plate (border, text baseline) are found
   with a Hough transform and the plate is rotated level.
3. **EasyOCR** with an A-Z/0-9 allow-list; text boxes are merged top-to-bottom and
   left-to-right so two-line plates (common on two-wheelers) read correctly.
4. **Retry:** if no valid plate is read, the same crop is retried as an Otsu-binarised image.
5. **Format correction:** Indian plates follow `SS DD [A-AAA] NNNN` (state, district, series,
   number). Each character is fixed according to its position, e.g. a `0` in the state part
   becomes `O`, an `S` in the number part becomes `5`. Unknown state codes are penalised.

### 3.4 Multi-frame voting
A vehicle is visible for many frames, so its plate is read up to every 3rd frame. Each read is
weighted by OCR confidence, whether it matches the plate format, and plate size (closer plates are
more reliable). The final plate is decided **character by character** across all valid reads, so
several partly wrong reads can still give the fully correct plate. OCR stops for a vehicle once
4 identical valid reads are found, which saves time.

### 3.5 Violation rules
| Violation | Rule |
|---|---|
| Red-light jumping | The vehicle's wheel point crosses the stop line while the signal is RED. |
| Over-speeding | Time taken between two lines a known distance apart gives the speed in km/h. |
| Wrong way | The vehicle crosses the two speed lines in the reverse order. |

The signal state can come from a manual schedule, a fixed timed cycle, or automatic colour
detection of the lamp in a fixed region of the frame.

Each violation is saved with the vehicle ID, plate, type, time, speed and a snapshot image.

## 4. Plate detector training

| Item | Value |
|---|---|
| Dataset | keremberke/license-plate-object-detection (Roboflow, CC BY 4.0) |
| Images | 6,176 train / 1,765 validation / 882 test, 1 class `license_plate` |
| Model | YOLOv8n, starting from COCO-pretrained weights |
| Hardware | Laptop CPU only (Intel Core Ultra 7 155H), no GPU |
| Schedule | Stage 1: 3 epochs on 50 % of the train set at 416 px (lr 0.01).<br>Stage 2: 4 fine-tuning epochs from the stage-1 weights (lr 0.002, no warm-up). |
| Augmentation | Ultralytics defaults (mosaic, HSV, scale, translate), **horizontal flip off** |

Validation mAP50 per epoch: 0.909, 0.841, 0.898 (stage 1), then 0.847, 0.885, 0.914, 0.922 (stage 2).
The best weights were scored once on the **held-out test split**, which was never used in training:

| Metric (test split, 882 images, 902 plates) | Value |
|---|---|
| Precision | **96.6 %** |
| Recall | **85.7 %** |
| mAP@0.5 | **93.2 %** |
| mAP@0.5:0.95 | **57.7 %** |

The detector almost never marks something that is not a plate (high precision); the plates it
misses are mostly very small or heavily angled ones. mAP@0.5:0.95 is lower because it also asks
for tight box edges, which matters less here: the crop is padded before OCR anyway.

## 5. Demo setup

| Item | Mumbai junction (main demo) | Bhopal intersection (second test) |
|---|---|---|
| Source | Pexels video 33604950 (free licence) | Pexels video 34430025 (free licence) |
| Footage | 4K, 30 fps, 19.2 s, handheld | 4K, 60 fps, 16.1 s, panning |
| Rules | Red light, over-speed (30 km/h, lines 8 m apart) | Plate reading only (camera pans, so fixed lines are meaningless) |
| Signal | Manual schedule: RED from 9.7 s (read frame by frame from the visible signal) | - |
| Config | `configs/mumbai.yaml` | `configs/bhopal.yaml` |

Output of the main demo: `outputs/mumbai/annotated_mumbai.mp4`, plus `violations.csv`,
`vehicles.csv`, a snapshot for each violation and the best plate crop for each vehicle.

### Ground truth
Ground truth was made by hand. Every plate crop found in the clip (sampled every 0.5 s at full 4K
resolution) was read by eye, and every vehicle was checked against the stop line and signal in the
annotated video. 12 plates were fully readable by a person; 3 more were left out because even a
person cannot read them for certain (one digit unclear, heavy blur, or cut off by the frame edge).
File: `data/ground_truth_mumbai.csv`.

## 6. Results (Mumbai demo)

### 6.1 Violation detection

| Metric | Value |
|---|---|
| True red-light violations (checked by eye) | 3 |
| Detected | 3 |
| True positives / false positives / false negatives | 3 / 0 / 0 |
| **Precision / recall / F1** | **100 % / 100 % / 100 %** |

All three violators are taxis that cross the stop line after the light turns red at 9.7 s
(at 10.2 s, 11.4 s and 14.5 s). No vehicle was flagged that had crossed before the light turned
red. No over-speeding was flagged: the two vehicles whose speed could be measured were moving at
10 km/h and 7 km/h, which is realistic for a busy junction.

### 6.2 Plate recognition

| Metric | Trained YOLO detector (final) | OCR-only baseline |
|---|---|---|
| Ground-truth plates | 12 | 12 |
| Plates read exactly right | 3 (25 %) | 3 (25 %) |
| Plates reported (valid format) | 6 | 7 |
| Plate precision (reported plates that are real) | **50 %** | 43 % |
| Wrong plates reported | 3 | 4 |
| Character accuracy | 64 % | 69 % |

Correctly read: `MH01BM3571`, `MH01DR3762` (green EV plate on the electric bus) and
`MH03AT3430` (one of the red-light violators, read the same way in 10 separate frames).

The baseline uses EasyOCR's own text detector on the whole vehicle. It reads the same 3 plates, but
it also reads road-sign text such as "KARVE ROAD" (from the *Maharshi Karve Road* sign) as text on
vehicles. The trained plate detector removes that kind of error, which is why precision goes up.

### 6.3 Plate detector vs. end-to-end plate reading
The plate **detector** is strong (93 % mAP@0.5 on the test set): in the contact sheets, almost every
visible plate in the clip has a box around it. The weak link is the **OCR** step on these plates:

| Problem seen in this clip | Example |
|---|---|
| Decorative dotted fonts on taxis | `MH.01.BT.6915` written with dots and curly serifs |
| Small, far plates (< 60 px wide) | Plates behind the first row of cars |
| Motion blur from the handheld camera | `MH01AT1825` read as `MH014745` |
| Short visibility | Many plates are readable for less than 1 s before the car turns away |

Even with these problems, the character-level voting picks a plate only when it matches the
Indian format with a real state code, so most bad reads are dropped instead of being reported.

### 6.4 Second test: Bhopal (MP plates)
Plate reading only (`configs/bhopal.yaml`, demo `demo/bhopal_plates_demo.mp4`). Three plates are
fully readable by a person: `MP04ZN2562`, `MP04CQ0991`, `MP09WF8336`
(`data/ground_truth_bhopal.csv`).

| Metric | Value |
|---|---|
| Plates read exactly right | 2 / 3 (67 %) |
| Character accuracy | 93 % |
| Plate precision | 33 % (2 of 6 reported plates) |

Plain modern fonts read much better than Mumbai's decorative taxi plates. Of the 4 wrong plates,
3 are near-misses of real plates (`AP04ZN2562`, `MP01CO0991`, `MP09WF8386`). They appear because the
tracker split one car into two IDs, and the second ID collected fewer reads. The fourth wrong plate,
`BH0P4141`, was the word "BHOPAL" painted on a school bus, accepted because `BH` was in the
state-code list. Bharat-series plates actually use a different layout (`22BH1234AA`), so `BH` was
removed from the list after this run (`anpr/plate_text.py`). That read is now rejected.

### 6.5 Processing speed
On the laptop CPU (no GPU) the full pipeline ran at **1.8 frames per second** on the Mumbai 4K clip
(577 frames in about 5.5 minutes). That is fine for recorded footage, but a GPU would be needed for
live use.

## 7. Limitations

- **OCR is the weakest stage.** EasyOCR is a general text reader. Fancy fonts, blur and small plates
  cause most of the errors, not the plate detector.
- **Handheld camera.** The virtual stop line is fixed in the image. When the camera moves, the line
  moves relative to the road. A fixed CCTV camera would make line-crossing exact.
- **Signal state.** The signal here was read from the video frame by frame and given as a schedule.
  The light that is visible faces the camera, so it was treated as the signal for the approach
  being monitored. A real system would get the signal phase from the signal controller.
- **Tracker ID switches.** Pedestrians and cyclists often block the taxis, so one vehicle can get a
  new ID (67 IDs were created in the 19 s clip, far more than the number of real vehicles). This splits plate reads across IDs and lowers accuracy.
- **Speed calibration.** Line distance (8 m) was estimated from the zebra crossing, so speeds are
  approximate (a few km/h).
- **Small training run.** The plate detector was trained for only 7 CPU epochs on half the data.

## 8. Future work

1. **Plate-specific OCR:** train a CRNN / PARSeq recogniser on Indian plate crops instead of the
   general EasyOCR model. This is the biggest expected gain.
2. **Indian plate data:** fine-tune the detector on Indian plates (two-wheelers, green EV plates,
   fancy fonts).
3. **Full training on a GPU:** 50+ epochs at 640 px on the full dataset (about 20 minutes on a free
   Colab GPU).
4. **Fixed CCTV footage** and automatic signal-colour detection.
5. **Re-identification** to merge split tracks of the same vehicle.

## 9. How to reproduce

```bash
pip install -r requirements.txt
python -m unittest discover tests                         # 14 unit tests
python main.py --config configs/mumbai.yaml               # demo video + CSVs
python evaluate.py --gt data/ground_truth_mumbai.csv --outputs outputs/mumbai
```

Videos are not stored in the repository because of their size. Download them from Pexels
(video IDs 33604950 and 34430025) into `data/` as `mumbai_4k.mp4` and `bhopal_4k.mp4`.
