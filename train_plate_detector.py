"""Train a YOLOv8 license-plate detector on a YOLO-format dataset.

Dataset options (free):
  * Roboflow Universe - search "license plate", export as "YOLOv8"
    -> unzip to datasets/plates/ (it already contains data.yaml)
  * Kaggle "Car License Plate Detection" (andrewmvd, Pascal-VOC XML)
    -> python tools/convert_voc_to_yolo.py --src <unzipped> --dst datasets/plates

Then:
    python train_plate_detector.py --data datasets/plates/data.yaml --epochs 50

CPU training works but is slow (use --epochs 20 --imgsz 512 to start);
Google Colab with a free GPU trains 50 epochs in ~15 minutes.
The best weights are copied to weights/plate_yolov8n.pt (used by main.py)
and validation metrics to weights/plate_metrics.json (used by evaluate.py).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to data.yaml")
    ap.add_argument("--model", default="yolov8n.pt", help="starting weights")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="", help='"" auto, "cpu", "0"')
    ap.add_argument("--fraction", type=float, default=1.0, help="use only this fraction of the train set")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=10, help="stop early after N epochs without improvement")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run")
    ap.add_argument("--lr0", type=float, default=0.01, help="initial learning rate (use ~0.002 to fine-tune)")
    ap.add_argument("--warmup", type=float, default=3.0, help="warm-up epochs (0 when fine-tuning)")
    ap.add_argument("--out", default="weights/plate_yolov8n.pt")
    args = ap.parse_args()

    from ultralytics import YOLO

    run_dir = os.path.abspath(os.path.join("runs", "plate", "train"))
    last = os.path.join(run_dir, "weights", "last.pt")
    if args.resume and os.path.exists(last):
        YOLO(last).train(resume=True)
    else:
        model = YOLO(args.model)
        kwargs = dict(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                      project=os.path.dirname(run_dir), name="train", exist_ok=True,
                      patience=args.patience, fraction=args.fraction, workers=args.workers,
                      lr0=args.lr0, warmup_epochs=args.warmup,
                      # plates are text: never mirror them
                      fliplr=0.0)
        if args.device:
            kwargs["device"] = args.device
        model.train(**kwargs)

    best = os.path.join(run_dir, "weights", "best.pt")
    if not os.path.exists(best):
        raise SystemExit(f"Training finished but {best} was not found")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    shutil.copy(best, args.out)
    print(f"Copied best weights to {args.out}")

    # score on the held-out test split if the dataset has one, else on val
    import yaml

    with open(args.data, encoding="utf-8") as f:
        has_test = "test" in (yaml.safe_load(f) or {})
    split = "test" if has_test else "val"
    metrics = YOLO(args.out).val(data=args.data, imgsz=args.imgsz, split=split, verbose=False,
                                 project=os.path.dirname(run_dir), name="eval", exist_ok=True)
    box = metrics.box
    out = dict(split=split, precision=float(box.mp), recall=float(box.mr), mAP50=float(box.map50),
               **{"mAP50-95": float(box.map)})
    path = os.path.join(os.path.dirname(args.out) or ".", "plate_metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"Saved metrics to {path}")


if __name__ == "__main__":
    main()
