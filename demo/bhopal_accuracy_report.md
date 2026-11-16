# Accuracy Report - Traffic Violation Detection & License Plate OCR

## Run summary

| Item | Value |
|---|---|
| Video | `data/bhopal_4k.mp4` (1280x720, 60.0 fps) |
| Frames processed | 969 |
| Processing speed | 0.12 FPS |
| Plate localisation | yolo |
| Vehicles tracked | 46 |
| Vehicles with a plate read | 12 |
| ... with a valid-format plate | 7 |
| Violations | RED_LIGHT: 0, OVERSPEED: 0, WRONG_WAY: 0 |

## Plate detector (YOLOv8n, held-out test split)

| Metric | Value |
|---|---|
| precision | 96.6% |
| recall | 85.7% |
| mAP50 | 93.2% |
| mAP50-95 | 57.7% |

## Plate recognition on the demo video

| Metric | Value |
|---|---|
| Ground-truth plates | 3 |
| Unique plates predicted | 6 |
| Exact-match recall | 66.7% |
| Near-match recall (<= 1 char error) | 66.7% |
| Character accuracy | 93.3% |
| Plate precision (predicted plates that are real) | 33.3% |
| False plate reads | 4 |

<details><summary>Per-plate results</summary>

| Ground truth | Best prediction | Edit distance |
|---|---|---|
| MP04CQ0991 | MP01CO0991 | 2 |
| MP04ZN2562 | MP04ZN2562 | 0 |
| MP09WF8336 | MP09WF8336 | 0 |

</details>

## Violation detection

| Metric | Value |
|---|---|
| Ground-truth violations | 0 |
| Detected violations | 0 |
| True positives | 0 |
| False positives | 0 |
| False negatives | 0 |
| Precision | 0.0% |
| Recall | 0.0% |
| F1 | 0.0% |

_A predicted violation is correct when its type matches and it is the same vehicle (same track ID, checked by eye in the annotated video) or its plate is within 1 character(s) of the true plate._

## Limitations

- Small, far-away, blurred or angled plates are often misread; accuracy drops sharply below ~30 px plate width.
- Night footage, glare, dirty or non-standard (fancy-font / handwritten) plates reduce OCR accuracy.
- Speed relies on a hand-measured distance between two lines; perspective and calibration errors of a few km/h are expected.
- Red-light detection needs the signal state (manual schedule, timed cycle or a visible lamp in a fixed ROI).
- Occlusion in dense traffic can swap tracker IDs, which may split one vehicle into two tracks.
- CPU-only processing is not real-time; a GPU or a smaller input width is needed for live use.
