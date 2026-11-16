"""Accuracy report for the ANPR + violation pipeline.

Two evaluation modes (use one or both):

1. Video-level  (after running main.py)
   Watch outputs/annotated.mp4 / outputs/plates/ and write the TRUE plates
   (and any real violations) into data/ground_truth.csv:

       plate,violation
       MH12AB1234,RED_LIGHT
       KA05MN9999,
       DL3CAF0001,OVERSPEED

       python evaluate.py --gt data/ground_truth.csv

2. OCR on labelled plate images  (e.g. the test split of a plate dataset)
   A folder of plate crops + a CSV  `filename,plate`:

       python evaluate.py --crops data/plate_crops --labels data/plate_crops/labels.csv

The report is written to outputs/accuracy_report.md (+ JSON).
Plate-detector mAP is included automatically if weights/plate_metrics.json
exists (written by train_plate_detector.py).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def best_match(target: str, pool: List[str]) -> Tuple[Optional[str], int]:
    best, dist = None, 10**9
    for p in pool:
        d = levenshtein(target, p)
        if d < dist:
            best, dist = p, d
    return best, dist


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return dict(precision=round(p, 4), recall=round(r, 4), f1=round(f, 4))


def _clean(s) -> str:
    if not isinstance(s, str):
        return ""
    return "".join(c for c in s.upper() if c.isalnum())


def eval_video(gt_path: str, vehicles_csv: str, violations_csv: str, tol: int) -> dict:
    gt = pd.read_csv(gt_path, dtype=str).fillna("")
    veh = pd.read_csv(vehicles_csv, dtype=str).fillna("") if os.path.exists(vehicles_csv) else pd.DataFrame()
    vio = pd.read_csv(violations_csv, dtype=str).fillna("") if os.path.exists(violations_csv) else pd.DataFrame()

    gt_plates = sorted({_clean(p) for p in gt["plate"] if _clean(p)})
    # a vehicle only "has a plate" when its read matches the plate format; partial / junk text
    # (e.g. 4 random characters) is treated as "no plate read", not as a plate prediction
    if "valid_format" in veh:
        veh = veh[veh["valid_format"].astype(str).str.lower() == "true"]
    pred_plates = sorted({_clean(p) for p in veh.get("plate", []) if _clean(p)})

    # --- plate recognition ------------------------------------------------
    exact = near = 0
    char_err = char_total = 0
    per_plate = []
    for g in gt_plates:
        m, d = best_match(g, pred_plates)
        d_eff = d if m is not None else len(g)
        exact += int(d_eff == 0)
        near += int(d_eff <= tol)
        char_err += min(d_eff, len(g))
        char_total += len(g)
        per_plate.append(dict(ground_truth=g, best_prediction=m or "", edit_distance=d_eff))
    pred_correct = sum(1 for p in pred_plates if best_match(p, gt_plates)[1] == 0) if gt_plates else 0
    n_gt, n_pred = len(gt_plates), len(pred_plates)
    plate_metrics = dict(
        ground_truth_plates=n_gt,
        predicted_unique_plates=n_pred,
        exact_match_recall=round(exact / n_gt, 4) if n_gt else 0.0,
        near_match_recall=round(near / n_gt, 4) if n_gt else 0.0,
        near_match_tolerance=tol,
        character_accuracy=round(1 - char_err / char_total, 4) if char_total else 0.0,
        plate_precision=round(pred_correct / n_pred, 4) if n_pred else 0.0,
        false_plate_reads=n_pred - pred_correct,
    )

    # --- violation detection ---------------------------------------------
    # Optional `track_id` column in the ground truth: the vehicle's ID in the annotated video.
    # When given, a violation is matched to the vehicle itself, so detecting the violation and
    # reading the plate are scored separately (a missed plate is not counted as a missed violation).
    gt_v = []
    for _, row in gt.iterrows():
        for v in str(row.get("violation", "")).split(";"):
            v = v.strip().upper()
            if v:
                gt_v.append((_clean(row["plate"]), v, str(row.get("track_id", "")).strip()))
    pred_v = [(_clean(r.get("plate", "")), str(r.get("violation", "")).upper(), str(r.get("track_id", "")).strip())
              for _, r in vio.iterrows()]

    by_type = {}
    used = set()
    tp_all = fp_all = 0
    for i, (pp, pv, ptid) in enumerate(pred_v):
        hit = None
        for j, (gp, gv, gtid) in enumerate(gt_v):
            if j in used or gv != pv:
                continue
            if (gtid and ptid == gtid) or (pp and gp and levenshtein(pp, gp) <= tol):
                hit = j
                break
        t = by_type.setdefault(pv, dict(tp=0, fp=0, fn=0))
        if hit is None:
            t["fp"] += 1
            fp_all += 1
        else:
            used.add(hit)
            t["tp"] += 1
            tp_all += 1
    fn_all = 0
    for j, (_, gv, _) in enumerate(gt_v):
        if j not in used:
            by_type.setdefault(gv, dict(tp=0, fp=0, fn=0))["fn"] += 1
            fn_all += 1
    for t in by_type.values():
        t.update(prf(t["tp"], t["fp"], t["fn"]))
    violation_metrics = dict(ground_truth=len(gt_v), predicted=len(pred_v), tp=tp_all, fp=fp_all, fn=fn_all,
                             **prf(tp_all, fp_all, fn_all), by_type=by_type,
                             note="A predicted violation is correct when its type matches and it is the same "
                                  "vehicle (same track ID, checked by eye in the annotated video) or its plate "
                                  f"is within {tol} character(s) of the true plate.")
    return dict(plates=plate_metrics, per_plate=per_plate, violations=violation_metrics)


def eval_crops(crops_dir: str, labels_csv: str, fmt: str) -> dict:
    import cv2
    from anpr.ocr import PlateOCR

    labels = pd.read_csv(labels_csv, dtype=str).fillna("")
    ocr = PlateOCR(fmt=fmt, min_conf=0.0)
    rows, exact, err, total, unread = [], 0, 0, 0, 0
    for _, r in labels.iterrows():
        img = cv2.imread(os.path.join(crops_dir, r["filename"]))
        truth = _clean(r["plate"])
        read = ocr.read_plate(img) if img is not None else None
        pred = read.text if read else ""
        unread += int(not pred)
        d = levenshtein(truth, pred)
        exact += int(d == 0)
        err += min(d, len(truth))
        total += len(truth)
        rows.append(dict(filename=r["filename"], truth=truth, pred=pred, edit_distance=d,
                         conf=round(read.conf, 3) if read else 0.0))
    n = len(rows)
    return dict(images=n, exact_match_accuracy=round(exact / n, 4) if n else 0.0,
                character_accuracy=round(1 - err / total, 4) if total else 0.0,
                unread=unread, details=rows)


def pct(x) -> str:
    return f"{100 * float(x):.1f}%"


def write_report(out_dir: str, video: Optional[dict], crops: Optional[dict], det: Optional[dict],
                 summary: Optional[dict]) -> str:
    L = ["# Accuracy Report - Traffic Violation Detection & License Plate OCR", ""]
    if summary:
        L += ["## Run summary", "",
              "| Item | Value |", "|---|---|",
              f"| Video | `{summary.get('source')}` ({summary.get('resolution')}, {summary.get('video_fps')} fps) |",
              f"| Frames processed | {summary.get('frames')} |",
              f"| Processing speed | {summary.get('processing_fps')} FPS |",
              f"| Plate localisation | {summary.get('plate_mode')} |",
              f"| Vehicles tracked | {summary.get('vehicles')} |",
              f"| Vehicles with a plate read | {summary.get('vehicles_with_plate')} |",
              f"| ... with a valid-format plate | {summary.get('vehicles_with_valid_plate')} |",
              f"| Violations | " + ", ".join(f"{k}: {v}" for k, v in summary.get("violations", {}).items()) + " |",
              ""]
    if det:
        L += [f"## Plate detector (YOLOv8n, held-out {det.get('split', 'val')} split)", "",
              "| Metric | Value |", "|---|---|"]
        for k in ("precision", "recall", "mAP50", "mAP50-95"):
            if k in det:
                L.append(f"| {k} | {pct(det[k])} |")
        L.append("")
    if video:
        p, v = video["plates"], video["violations"]
        L += ["## Plate recognition on the demo video", "",
              "| Metric | Value |", "|---|---|",
              f"| Ground-truth plates | {p['ground_truth_plates']} |",
              f"| Unique plates predicted | {p['predicted_unique_plates']} |",
              f"| Exact-match recall | {pct(p['exact_match_recall'])} |",
              f"| Near-match recall (<= {p['near_match_tolerance']} char error) | {pct(p['near_match_recall'])} |",
              f"| Character accuracy | {pct(p['character_accuracy'])} |",
              f"| Plate precision (predicted plates that are real) | {pct(p['plate_precision'])} |",
              f"| False plate reads | {p['false_plate_reads']} |", "",
              "<details><summary>Per-plate results</summary>", "",
              "| Ground truth | Best prediction | Edit distance |", "|---|---|---|"]
        L += [f"| {r['ground_truth']} | {r['best_prediction']} | {r['edit_distance']} |" for r in video["per_plate"]]
        L += ["", "</details>", "",
              "## Violation detection", "",
              "| Metric | Value |", "|---|---|",
              f"| Ground-truth violations | {v['ground_truth']} |",
              f"| Detected violations | {v['predicted']} |",
              f"| True positives | {v['tp']} |",
              f"| False positives | {v['fp']} |",
              f"| False negatives | {v['fn']} |",
              f"| Precision | {pct(v['precision'])} |",
              f"| Recall | {pct(v['recall'])} |",
              f"| F1 | {pct(v['f1'])} |", "",
              f"_{v['note']}_", ""]
        if v["by_type"]:
            L += ["| Type | TP | FP | FN | Precision | Recall |", "|---|---|---|---|---|---|"]
            L += [f"| {k} | {t['tp']} | {t['fp']} | {t['fn']} | {pct(t['precision'])} | {pct(t['recall'])} |"
                  for k, t in v["by_type"].items()]
            L.append("")
    if crops:
        L += ["## OCR on labelled plate images", "",
              "| Metric | Value |", "|---|---|",
              f"| Images | {crops['images']} |",
              f"| Exact-match accuracy | {pct(crops['exact_match_accuracy'])} |",
              f"| Character accuracy | {pct(crops['character_accuracy'])} |",
              f"| Unread | {crops['unread']} |", ""]
        bad = [r for r in crops["details"] if r["edit_distance"]][:15]
        if bad:
            L += ["Sample errors:", "", "| File | Truth | Predicted |", "|---|---|---|"]
            L += [f"| {r['filename']} | {r['truth']} | {r['pred'] or '(none)'} |" for r in bad]
            L.append("")
    L += ["## Limitations", "",
          "- Small, far-away, blurred or angled plates are often misread; accuracy drops sharply below ~30 px plate width.",
          "- Night footage, glare, dirty or non-standard (fancy-font / handwritten) plates reduce OCR accuracy.",
          "- Speed relies on a hand-measured distance between two lines; perspective and calibration errors of a few km/h are expected.",
          "- Red-light detection needs the signal state (manual schedule, timed cycle or a visible lamp in a fixed ROI).",
          "- Occlusion in dense traffic can swap tracker IDs, which may split one vehicle into two tracks.",
          "- CPU-only processing is not real-time; a GPU or a smaller input width is needed for live use.", ""]
    path = os.path.join(out_dir, "accuracy_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Accuracy report for ANPR + violations")
    ap.add_argument("--gt", help="ground-truth CSV for the demo video (plate,violation)")
    ap.add_argument("--outputs", default="outputs", help="folder with vehicles.csv / violations.csv")
    ap.add_argument("--crops", help="folder of labelled plate crops")
    ap.add_argument("--labels", help="CSV filename,plate for --crops")
    ap.add_argument("--format", default="india", help="plate format: india | none")
    ap.add_argument("--tolerance", type=int, default=1, help="max character errors for a near match")
    ap.add_argument("--det-metrics", default="weights/plate_metrics.json")
    args = ap.parse_args()
    if not args.gt and not args.crops:
        ap.error("give --gt and/or --crops/--labels")

    os.makedirs(args.outputs, exist_ok=True)
    video = eval_video(args.gt, os.path.join(args.outputs, "vehicles.csv"),
                       os.path.join(args.outputs, "violations.csv"), args.tolerance) if args.gt else None
    crops = eval_crops(args.crops, args.labels, args.format) if args.crops else None
    det = None
    if os.path.exists(args.det_metrics):
        with open(args.det_metrics, encoding="utf-8") as f:
            det = json.load(f)
    summary = None
    sp = os.path.join(args.outputs, "run_summary.json")
    if os.path.exists(sp):
        with open(sp, encoding="utf-8") as f:
            summary = json.load(f)

    path = write_report(args.outputs, video, crops, det, summary)
    with open(os.path.join(args.outputs, "accuracy_report.json"), "w", encoding="utf-8") as f:
        json.dump(dict(video=video, crops=crops, detector=det, run=summary), f, indent=2)
    print(f"Report written to {path}")
    if video:
        print(json.dumps(video["plates"], indent=2))
        print(json.dumps({k: v for k, v in video["violations"].items() if k != "by_type"}, indent=2))
    if crops:
        print(json.dumps({k: v for k, v in crops.items() if k != "details"}, indent=2))


if __name__ == "__main__":
    main()
