"""Reproducible evaluation metadata without recording API credentials."""

import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, default=str,
    ).encode("utf-8")).hexdigest()


def run_provenance(
    *, model: str | None, provider: str | None,
    model_revision: str | None = None,
    data_files: Iterable[str | Path] = (),
    scorer_files: Iterable[str | Path] = (),
    parameters: dict[str, Any] | None = None,
    dataset_revision: str | None = None,
) -> dict:
    versions = {}
    for package in ("my-agents-csy", "openai", "datasets", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "model": model,
        "model_revision": model_revision,
        "provider": provider,
        "dataset_revision": dataset_revision,
        "dataset_files_sha256": {str(Path(path).resolve()): sha256_file(path)
                                 for path in data_files},
        "scorer_files_sha256": {str(Path(path).resolve()): sha256_file(path)
                                for path in scorer_files},
        "parameters": parameters or {},
        "package_versions": versions,
    }
