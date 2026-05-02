from __future__ import annotations

import json
import shutil
import subprocess
import time
import zlib
from pathlib import Path
from typing import Any

import pandas as pd


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("_")


def experiment_dir(output_dir: str | Path, model_name: str, f1: float) -> Path:
    base = Path(output_dir) / f"{_safe_name(model_name)}_{int(round(f1 * 100)):02d}"
    path = base
    suffix = 2
    while path.exists():
        path = Path(f"{base}_{suffix}")
        suffix += 1
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_simple_png(path: str | Path) -> None:
    path = Path(path)
    width, height = 16, 16
    raw = b"".join(b"\x00" + b"\xff\xff\xff" * width for _ in range(height))
    def chunk(kind: bytes, data: bytes) -> bytes:
        return len(data).to_bytes(4, "big") + kind + data + zlib.crc32(kind + data).to_bytes(4, "big")
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", width.to_bytes(4,"big") + height.to_bytes(4,"big") + b"\x08\x02\x00\x00\x00") + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def write_config_yaml(path: str | Path, config: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        for key, value in config.items():
            fh.write(f"{key}: {json.dumps(value, ensure_ascii=False)}\n")


def save_experiment_artifacts(
    output_dir: str | Path,
    model_name: str,
    metrics: dict[str, float],
    predictions: pd.DataFrame,
    report: pd.DataFrame,
    config: dict[str, Any],
    splits: dict[str, list[int]],
) -> tuple[Path, Path]:
    run_dir = experiment_dir(output_dir, model_name, metrics["f1"])
    pd.DataFrame([metrics]).to_csv(run_dir / "metrics.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    report.to_csv(run_dir / "classification_report.csv")
    write_simple_png(run_dir / "classification_report.png")
    write_config_yaml(run_dir / "config_used.yaml", config)
    (run_dir / "train_val_test_years.json").write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")
    zip_root = Path(output_dir) / "zips"
    zip_root.mkdir(parents=True, exist_ok=True)
    zip_path = zip_root / f"{run_dir.name}.zip"
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=run_dir)
    return run_dir, zip_path


def update_dataset_metadata(persistent_dataset_dir: str | Path, dataset_slug: str) -> Path:
    path = Path(persistent_dataset_dir)
    path.mkdir(parents=True, exist_ok=True)
    metadata = {"id": dataset_slug, "title": dataset_slug.split("/")[-1]}
    metadata_path = path / "dataset-metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata_path


def copy_zip_to_persistent(zip_path: str | Path, persistent_dataset_dir: str | Path, dataset_slug: str) -> Path:
    update_dataset_metadata(persistent_dataset_dir, dataset_slug)
    destination = Path(persistent_dataset_dir) / Path(zip_path).name
    shutil.copy2(zip_path, destination)
    return destination


def upload_dataset_version(persistent_dataset_dir: str | Path, message: str = "new experiment results", attempts: int = 3) -> None:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            subprocess.run(
                ["kaggle", "datasets", "version", "-p", str(persistent_dataset_dir), "-m", message],
                check=True,
            )
            return
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"Kaggle dataset upload failed after {attempts} attempts") from last_error
