"""Train a YOLO resistor detector.

Examples:
    python train.py
    python train.py --epochs 8 --batch 4 --imgsz 320 --fraction 0.1
    python train.py --weights runs/detect/resistor_detector/weights/best.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a YOLO resistor-detection model."
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_DIR / "dataset" / "data.yaml",
        help="Path to the YOLO dataset YAML file.",
    )

    parser.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_DIR / "yolo11n.pt",
        help="Pretrained weights or a previous checkpoint.",
    )

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)

    parser.add_argument(
        "--fraction",
        type=float,
        default=0.1,
        help="Fraction of the training dataset to use (0 < fraction <= 1).",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help="Training device: 'cpu' or a CUDA device such as '0'.",
    )

    parser.add_argument(
        "--project",
        type=Path,
        default=PROJECT_DIR / "runs" / "detect",
    )

    parser.add_argument(
        "--name",
        default="resistor_detector_fast",
        help="Name of the training run.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data_path = args.data.expanduser().resolve()
    weights_path = args.weights.expanduser().resolve()
    output_dir = args.project.expanduser().resolve()

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset configuration was not found: {data_path}"
        )

    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Model weights were not found: {weights_path}"
        )

    if args.epochs < 1:
        raise ValueError("epochs must be at least 1.")

    if args.imgsz < 32:
        raise ValueError("imgsz must be at least 32.")

    if args.batch < 1:
        raise ValueError("batch must be at least 1.")

    if args.workers < 0:
        raise ValueError("workers cannot be negative.")

    if not 0 < args.fraction <= 1:
        raise ValueError("fraction must be greater than 0 and at most 1.")

    print(f"Dataset: {data_path}")
    print(f"Starting weights: {weights_path}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Training dataset fraction: {args.fraction * 100:.0f}%")

    model = YOLO(str(weights_path))

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        fraction=args.fraction,
        project=str(output_dir),
        name=args.name,
        exist_ok=True,
        plots=True,
    )

    save_dir = Path(model.trainer.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    last_weights = save_dir / "weights" / "last.pt"

    print("\nTraining finished.")
    print(f"Best model: {best_weights}")
    print(f"Last checkpoint: {last_weights}")


if __name__ == "__main__":
    main()