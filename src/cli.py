from __future__ import annotations

import argparse

from .training import DEFAULT_OUTPUT_DIR, run_experiments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run potato illness forecasting experiments")
    parser.add_argument("--excel-path", default="Golitcino72-17d2_CLEAN.xlsx")
    parser.add_argument("--models", nargs="+", default=["logreg"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[2])
    parser.add_argument("--window-sizes", nargs="+", type=int, default=[7])
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--tune-trials", type=int, default=1)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--upload-dataset", action="store_true")
    parser.add_argument("--dataset-slug", default=None)
    parser.add_argument("--no-y", action="store_true", help="Exclude y1/y2/y3/y4 and their derived features")
    parser.add_argument("--invert-classes", action="store_true", help="Invert final binary predictions")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = run_experiments(
        excel_path=args.excel_path,
        models=args.models,
        horizons=args.horizons,
        window_sizes=args.window_sizes,
        output_dir=args.output_dir,
        top_n=args.top_n,
        tune_trials=args.tune_trials,
        upload_dataset=args.upload_dataset,
        dataset_slug=args.dataset_slug,
        no_y=args.no_y,
        invert_classes=args.invert_classes,
    )
    for row in results:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
