# Accuracy Report - Traffic Violation Detection & License Plate OCR

## Run summary

| Item | Value |
|---|---|
| Video | `data/mumbai_4k.mp4` (1280x720, 30.0 fps) |
| Frames processed | 577 |
| Processing speed | 1.8 FPS |
| Plate localisation | yolo |
| Vehicles tracked | 67 |
| Vehicles with a plate read | 15 |
| ... with a valid-format plate | 6 |
| Violations | RED_LIGHT: 3, OVERSPEED: 0, WRONG_WAY: 0 |

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
| Ground-truth plates | 12 |
| Unique plates predicted | 6 |
| Exact-match recall | 25.0% |
| Near-match recall (<= 1 char error) | 25.0% |
| Character accuracy | 64.2% |
| Plate precision (predicted plates that are real) | 50.0% |
| False plate reads | 3 |

<details><summary>Per-plate results</summary>

| Ground truth | Best prediction | Edit distance |
|---|---|---|
| MH01AT1825 | MH014745 | 5 |
| MH01AT3305 | MH03AT3430 | 3 |
| MH01AT5336 | MH03AT3430 | 4 |
| MH01AT5973 | MH01BM3571 | 5 |
| MH01BM3571 | MH01BM3571 | 0 |
| MH01BT6915 | MH014745 | 5 |
| MH01DR3762 | MH01DR3762 | 0 |
| MH01EW0834 | MH014745 | 6 |
| MH02BQ9909 | MH01BM3571 | 6 |
| MH02BQ9955 | MH014745 | 6 |
| MH03AT3005 | MH03AT3430 | 3 |
| MH03AT3430 | MH03AT3430 | 0 |

</details>

## Violation detection

| Metric | Value |
|---|---|
| Ground-truth violations | 3 |
| Detected violations | 3 |
| True positives | 3 |
| False positives | 0 |
| False negatives | 0 |
| Precision | 100.0% |
| Recall | 100.0% |
| F1 | 100.0% |

_A predicted violation is correct when its type matches and it is the same vehicle (same track ID, checked by eye in the annotated video) or its plate is within 1 character(s) of the true plate._

| Type | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| RED_LIGHT | 3 | 0 | 0 | 100.0% | 100.0% |

## Limitations

- Small, far-away, blurred or angled plates are often misread; accuracy drops sharply below ~30 px plate width.
- Night footage, glare, dirty or non-standard (fancy-font / handwritten) plates reduce OCR accuracy.
- Speed relies on a hand-measured distance between two lines; perspective and calibration errors of a few km/h are expected.
- Red-light detection needs the signal state (manual schedule, timed cycle or a visible lamp in a fixed ROI).
- Occlusion in dense traffic can swap tracker IDs, which may split one vehicle into two tracks.
- CPU-only processing is not real-time; a GPU or a smaller input width is needed for live use.
