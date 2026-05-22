"""
finetune.py — Standalone YOLOv8 fine-tuning CLI script.

Usage:
    python finetune.py --data path/to/data.yaml [options]

This script is self-contained and does NOT import from backend.app or any
other project-specific module. It requires only ultralytics and stdlib packages.
"""

import argparse
import shutil
import sys
import traceback
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print(
        "ultralytics package required: pip install ultralytics",
        file=sys.stderr,
    )
    sys.exit(1)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Fine-tune the YOLOv8 car damage detection model with new labeled data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage — overwrites best.pt in place (will prompt for confirmation):
  python finetune.py --data /path/to/dataset/data.yaml

  # Full options — no confirmation prompt, custom output path:
  python finetune.py \\
      --data /path/to/dataset/data.yaml \\
      --base-model best.pt \\
      --output new_best.pt \\
      --epochs 50 \\
      --imgsz 640 \\
      --batch 16 \\
      --device 0 \\
      --no-confirm
""",
    )

    parser.add_argument(
        "--data",
        required=True,
        metavar="DATA_YAML",
        help="Path to the YOLOv8 data.yaml file (required). Must exist.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        metavar="WEIGHTS",
        help=(
            'Path to the base model weights file '
            '(default: "best.pt" in the same directory as this script).'
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="OUTPUT_PATH",
        help=(
            'Path where the fine-tuned model will be written '
            '(default: "best.pt" in the same directory as this script, '
            'overwriting the base model in place).'
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        metavar="N",
        help="Number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        metavar="PX",
        help="Input image size in pixels (default: 640).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        metavar="N",
        help="Training batch size (default: 16). Reduce to 8 or 4 if CUDA OOM.",
    )
    parser.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help=(
            'Device to train on: "cpu", "0" (first GPU), "cuda:0", etc. '
            "(default: auto-detect)."
        ),
    )
    parser.add_argument(
        "--project",
        default="runs/train",
        metavar="DIR",
        help='Parent directory for training run output (default: "runs/train").',
    )
    parser.add_argument(
        "--name",
        default="finetune_run",
        metavar="NAME",
        help='Name of the training run subdirectory (default: "finetune_run").',
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help=(
            "Skip the overwrite confirmation prompt when --output resolves to the "
            "same path as --base-model."
        ),
    )

    return parser.parse_args(argv)


def resolve_path(path_str, base_dir):
    """Resolve path_str relative to base_dir if it is not absolute."""
    p = Path(path_str)
    if not p.is_absolute():
        p = base_dir / p
    return p.resolve()


def main(argv=None):
    args = parse_args(argv)

    script_dir = Path(__file__).parent.resolve()

    # --- Resolve and validate base model path ---
    base_model_str = args.base_model if args.base_model is not None else "best.pt"
    base_model_path = resolve_path(base_model_str, script_dir)

    if not base_model_path.exists():
        print(
            f"ERROR: base model not found: {base_model_path}",
            file=sys.stderr,
        )
        print(
            "  Ensure best.pt is in the project root or pass --base-model with an explicit path.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Resolve and validate data.yaml path ---
    data_path = Path(args.data).resolve()

    if not data_path.exists():
        print(
            f"ERROR: data.yaml not found: {data_path}",
            file=sys.stderr,
        )
        print(
            "  Run: python finetune.py --help  for usage examples.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Resolve output path ---
    output_str = args.output if args.output is not None else "best.pt"
    output_path = resolve_path(output_str, script_dir)

    # --- Overwrite confirmation ---
    if output_path == base_model_path and not args.no_confirm:
        response = input(
            f"This will overwrite the existing model at {base_model_path}. Continue? [y/N]: "
        ).strip().lower()
        if response != "y":
            print("Aborted.")
            sys.exit(0)

    # --- Print training summary ---
    print("=" * 60)
    print("Fine-tuning parameters:")
    print(f"  Base model  : {base_model_path}")
    print(f"  Data yaml   : {data_path}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Image size  : {args.imgsz}")
    print(f"  Batch size  : {args.batch}")
    print(f"  Device      : {args.device if args.device is not None else 'auto'}")
    print(f"  Project dir : {args.project}")
    print(f"  Run name    : {args.name}")
    print(f"  Output path : {output_path}")
    print("=" * 60)

    # --- Load model and train ---
    try:
        model = YOLO(str(base_model_path))

        train_kwargs = dict(
            data=str(data_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            name=args.name,
            exist_ok=True,
        )
        # Only pass device if explicitly provided — do not pass device=None to YOLO
        if args.device is not None:
            train_kwargs["device"] = args.device

        model.train(**train_kwargs)

    except Exception:
        print("\nERROR: Training failed:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # --- Locate best weights produced by training ---
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    if not best_weights.exists():
        print(
            f"ERROR: Training completed but best.pt was not found at: {best_weights}",
            file=sys.stderr,
        )
        print(
            "  Check the runs/ directory for partial output.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Copy best weights to output path ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(best_weights), str(output_path))
    print(f"\nFine-tuning complete. New model saved to: {output_path}")


if __name__ == "__main__":
    main()
