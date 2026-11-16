"""Traffic violation detection + license plate recognition on a video.

    python main.py                              # uses config.yaml
    python main.py --source data/junction.mp4   # override the input video
    python main.py --display                    # live preview window

Outputs (in outputs/):
    annotated.mp4     demo video with boxes, plates, speeds, signal and alerts
    violations.csv    one row per violation (plate, type, time, speed, snapshot)
    vehicles.csv      one row per tracked vehicle (plate, OCR confidence, speed)
    violations/       JPG snapshot of every violation
    plates/           best plate crop for each vehicle (handy for ground truth)
    run_summary.json  processing stats
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import cv2
import pandas as pd
import yaml

from anpr.draw import draw_hud, draw_lines, draw_track
from anpr.ocr import PlateOCR
from anpr.plate_detector import PlateFinder
from anpr.signal import SignalController
from anpr.violations import OVERSPEED, RED_LIGHT, WRONG_WAY, RuleEngine, Track

# COCO class ids used by the pretrained YOLO model
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resize_keep(frame, width: int):
    if not width or frame.shape[1] == width:
        return frame
    h = int(frame.shape[0] * width / frame.shape[1])
    return cv2.resize(frame, (width, h), interpolation=cv2.INTER_AREA)


def safe_name(s: str) -> str:
    return "".join(c for c in s if c.isalnum()) or "UNKNOWN"


def run(cfg: dict) -> dict:
    from ultralytics import YOLO

    vcfg, mcfg, ocfg, outcfg = cfg["video"], cfg["models"], cfg["ocr"], cfg.get("output", {})
    out_dir = outcfg.get("dir", "outputs")
    snap_dir = os.path.join(out_dir, "violations")
    plate_dir = os.path.join(out_dir, "plates")
    for d in (out_dir, snap_dir, plate_dir):
        os.makedirs(d, exist_ok=True)

    source = vcfg["source"]
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if src_fps <= 1 or src_fps > 240:
        src_fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    ok, raw = cap.read()
    if not ok:
        raise SystemExit("Video has no frames")
    frame = resize_keep(raw, vcfg.get("resize_width", 1280))
    H, W = frame.shape[:2]
    # Detection/tracking runs on the resized frame, but plates are cropped from the
    # ORIGINAL frame: on 4K footage a plate is ~3x more pixels than after resizing.
    full_res_ocr = bool(ocfg.get("full_res_crops", True))
    max_crop_w = int(ocfg.get("max_crop_width", 960))

    print(f"[video] {source}  {W}x{H} @ {src_fps:.1f} fps, {total or '?'} frames")
    vehicle_model = YOLO(mcfg.get("vehicle", "yolov8n.pt"))
    ocr = PlateOCR(ocfg.get("languages", ["en"]), gpu=bool(ocfg.get("gpu", False)),
                   fmt=ocfg.get("format", "india"), min_conf=float(ocfg.get("min_confidence", 0.25)))
    finder = PlateFinder(ocr, mcfg.get("plate"), conf=float(mcfg.get("plate_conf", 0.25)),
                         device=mcfg.get("device", ""), min_plate_width=int(ocfg.get("min_plate_width", 30)),
                         imgsz=int(mcfg.get("plate_imgsz", 416)))
    signal = SignalController(cfg["rules"].get("red_light", {}).get("signal", {}))
    engine = RuleEngine(cfg["rules"], W, H)

    writer = cv2.VideoWriter(vcfg.get("output", os.path.join(out_dir, "annotated.mp4")),
                             cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (W, H))

    tracks: Dict[int, Track] = {}
    violations: List[dict] = []
    counts = {RED_LIGHT: 0, OVERSPEED: 0, WRONG_WAY: 0}
    alerts: List[str] = []
    alert_until: List[float] = []
    every_n = int(ocfg.get("every_n_frames", 3))
    plate_fmt = ocfg.get("format", "india")
    min_vw = int(ocfg.get("min_vehicle_width", 60))
    max_frames = vcfg.get("max_frames")
    display = bool(vcfg.get("display", False))

    track_kwargs = dict(persist=True, classes=list(VEHICLE_CLASSES), conf=float(mcfg.get("vehicle_conf", 0.35)),
                        tracker=mcfg.get("tracker", "bytetrack.yaml"), verbose=False)
    if mcfg.get("device"):
        track_kwargs["device"] = mcfg["device"]

    frame_idx = 0
    t_start = time.time()
    fps_smooth = 0.0
    while ok:
        t0 = time.time()
        t_video = frame_idx / src_fps
        sig = signal.state(t_video, frame) if engine.red_enabled else "OFF"
        clean_frame = frame.copy()

        res = vehicle_model.track(frame, **track_kwargs)[0]
        active_ids = []
        ocr_jobs = []
        if res.boxes is not None and res.boxes.id is not None:
            xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
            ids = res.boxes.id.cpu().numpy().astype(int)
            clss = res.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), tid, c in zip(xyxy, ids, clss):
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, W - 1), min(y2, H - 1)
                tr = tracks.get(tid)
                if tr is None:
                    tr = tracks[tid] = Track(int(tid), VEHICLE_CLASSES.get(int(c), "vehicle"), frame_idx)
                    tr.voter.fmt = plate_fmt
                tr.box = (x1, y1, x2, y2)
                tr.last_frame = frame_idx
                anchor = ((x1 + x2) / 2.0, float(y2))  # bottom-centre = where the wheels touch the road
                for v in engine.update(tr, anchor, t_video, sig):
                    counts[v] += 1
                    rec = dict(track_id=tr.track_id, violation=v, frame=frame_idx, time_s=round(t_video, 2),
                               vehicle=tr.cls_name, signal=sig,
                               speed_kmph=round(tr.speed_kmph, 1) if tr.speed_kmph else None, snapshot="")
                    if outcfg.get("save_snapshots", True):
                        snap = clean_frame.copy()
                        cv2.rectangle(snap, (x1, y1), (x2, y2), (40, 40, 230), 3)
                        cv2.putText(snap, f"{v}  #{tr.track_id}  t={t_video:.1f}s", (x1, max(y1 - 10, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 230), 2, cv2.LINE_AA)
                        p = os.path.join(snap_dir, f"{v}_id{tr.track_id}_f{frame_idx}.jpg")
                        cv2.imwrite(p, snap)
                        rec["snapshot"] = p
                    violations.append(rec)
                    alerts.append(f"{v.replace('_', ' ')}  #{tr.track_id}")
                    alert_until.append(t_video + 3.0)
                    print(f"[violation] t={t_video:6.1f}s  #{tr.track_id:<4} {tr.cls_name:<10} {v}")
                tr.last_pt = anchor
                active_ids.append(tid)

                # schedule OCR for this vehicle?
                if (x2 - x1) >= min_vw and frame_idx - tr.last_ocr_frame >= every_n \
                        and not tr.voter.is_locked():
                    ocr_jobs.append(tr)

        if ocr_jobs:
            src = raw if full_res_ocr else clean_frame
            s = src.shape[1] / float(W)          # resized-frame -> source scale
            crops, scales = [], []
            for t in ocr_jobs:
                x1, y1, x2, y2 = (int(round(v * s)) for v in t.box)
                crop = src[y1:y2, x1:x2]
                k = 1.0
                if crop.shape[1] > max_crop_w:   # keep OCR time bounded on huge crops
                    k = max_crop_w / crop.shape[1]
                    crop = cv2.resize(crop, (max_crop_w, max(int(crop.shape[0] * k), 1)), interpolation=cv2.INTER_AREA)
                crops.append(crop)
                scales.append(s * k)             # crop pixel = frame pixel * (s * k)
            for tr, read, sk in zip(ocr_jobs, finder.read(crops), scales):
                tr.last_ocr_frame = frame_idx
                if read is None:
                    continue
                # bigger (closer) plates are more reliable: weight reads by plate width
                pw = read.crop.shape[1] if read.crop is not None else 100
                size_w = min(max(pw / 100.0, 0.5), 1.5)
                tr.voter.add(read.text, read.conf, read.valid, read.score * size_w)
                if read.box is not None and read.valid:
                    bx1, by1, bx2, by2 = (int(v / sk) for v in read.box)
                    tr.plate_box = (tr.box[0] + bx1, tr.box[1] + by1, tr.box[0] + bx2, tr.box[1] + by2)
                    tr.plate_box_frame = frame_idx
                if read.text == tr.plate and read.crop is not None and read.score > tr.best_plate_score:
                    tr.best_plate_score = read.score
                    tr.best_plate_crop = read.crop

        # expire alerts
        keep = [i for i, until in enumerate(alert_until) if until > t_video]
        alerts = [alerts[i] for i in keep]
        alert_until = [alert_until[i] for i in keep]

        draw_lines(frame, engine.lines(), sig)
        for tid in active_ids:
            tr = tracks[tid]
            # plate box positions go stale as the vehicle moves: show them only right after a read
            draw_track(frame, tr, show_plate_box=frame_idx - tr.plate_box_frame < every_n)
        dt = time.time() - t0
        fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / max(dt, 1e-6)) if fps_smooth else 1.0 / max(dt, 1e-6)
        draw_hud(frame, sig, counts, len(tracks), fps_smooth, alerts, finder.mode)
        writer.write(frame)

        if display:
            cv2.imshow("ANPR - traffic violations (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 30 == 0:
            pct = f"{100 * frame_idx / total:5.1f}%" if total else ""
            print(f"[progress] frame {frame_idx}/{total or '?'} {pct}  {fps_smooth:.1f} fps  "
                  f"vehicles={len(tracks)} violations={len(violations)}")
        if max_frames and frame_idx >= int(max_frames):
            break
        ok, raw = cap.read()
        if ok:
            frame = resize_keep(raw, vcfg.get("resize_width", 1280))

    cap.release()
    writer.release()
    if display:
        cv2.destroyAllWindows()

    # ---- final plate per vehicle (multi-frame vote) ------------------------
    vehicle_rows = []
    for tr in tracks.values():
        plate, conf, agree = tr.voter.best()
        crop_path = ""
        if plate and tr.best_plate_crop is not None and outcfg.get("save_plate_crops", True):
            crop_path = os.path.join(plate_dir, f"id{tr.track_id}_{safe_name(plate)}.jpg")
            cv2.imwrite(crop_path, tr.best_plate_crop)
        vehicle_rows.append(dict(
            track_id=tr.track_id, vehicle=tr.cls_name, plate=plate, plate_conf=round(conf, 3),
            agreeing_reads=agree, total_reads=tr.voter.n_reads,
            valid_format=bool(tr.voter.valid.get(plate, False)) if plate else False,
            speed_kmph=round(tr.speed_kmph, 1) if tr.speed_kmph else None,
            first_seen_s=round(tr.first_frame / src_fps, 2), last_seen_s=round(tr.last_frame / src_fps, 2),
            frames_seen=tr.last_frame - tr.first_frame + 1,
            violations=";".join(tr.violations), plate_crop=crop_path))
    for v in violations:
        tr = tracks[v["track_id"]]
        plate, conf, _ = tr.voter.best()
        v["plate"] = plate or "UNREAD"
        v["plate_conf"] = round(conf, 3)
        if v["speed_kmph"] is None and tr.speed_kmph:
            v["speed_kmph"] = round(tr.speed_kmph, 1)

    vcols = ["track_id", "plate", "plate_conf", "violation", "vehicle", "time_s", "frame", "signal",
             "speed_kmph", "snapshot"]
    pd.DataFrame(violations, columns=vcols).to_csv(os.path.join(out_dir, "violations.csv"), index=False)
    pd.DataFrame(vehicle_rows).sort_values("track_id").to_csv(os.path.join(out_dir, "vehicles.csv"), index=False) \
        if vehicle_rows else pd.DataFrame().to_csv(os.path.join(out_dir, "vehicles.csv"), index=False)

    elapsed = time.time() - t_start
    summary = dict(source=str(source), frames=frame_idx, resolution=f"{W}x{H}", video_fps=src_fps,
                   processing_seconds=round(elapsed, 1), processing_fps=round(frame_idx / max(elapsed, 1e-6), 2),
                   plate_mode=finder.mode, vehicles=len(tracks),
                   vehicles_with_plate=sum(1 for r in vehicle_rows if r["plate"]),
                   vehicles_with_valid_plate=sum(1 for r in vehicle_rows if r["valid_format"]),
                   violations=counts)
    with open(os.path.join(out_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n[done]", json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Traffic violation detection with license plate OCR")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", help="override video source")
    ap.add_argument("--output", help="override annotated video path")
    ap.add_argument("--plate-model", help="override plate detector weights ('none' = OCR fallback)")
    ap.add_argument("--out-dir", help="override output folder (CSV, snapshots, plates)")
    ap.add_argument("--display", action="store_true", help="show live preview window")
    ap.add_argument("--max-frames", type=int, help="process only the first N frames")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.source:
        cfg["video"]["source"] = args.source
    if args.output:
        cfg["video"]["output"] = args.output
    if args.plate_model:
        cfg["models"]["plate"] = None if args.plate_model.lower() == "none" else args.plate_model
    if args.out_dir:
        cfg.setdefault("output", {})["dir"] = args.out_dir
        if not args.output:
            cfg["video"]["output"] = os.path.join(args.out_dir, os.path.basename(cfg["video"]["output"]))
    if args.display:
        cfg["video"]["display"] = True
    if args.max_frames:
        cfg["video"]["max_frames"] = args.max_frames
    run(cfg)


if __name__ == "__main__":
    main()
