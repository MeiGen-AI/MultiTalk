"""Lightweight filesystem readiness checks for multitalk generator."""

from __future__ import annotations

import importlib.util
import os
from types import ModuleType
from typing import Any, Dict, Iterable, List


DEFAULT_WEIGHT_EXTENSIONS: tuple[str, ...] = (
    ".safetensors",
    ".pth",
    ".pt",
    ".bin",
    ".gguf",
    ".ckpt",
)


def _repo_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _resolve_path(base_dir: str, path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(base_dir, path_value))


def _load_config_module(repo_dir: str) -> ModuleType:
    config_path = os.path.join(repo_dir, "config.py")
    spec = importlib.util.spec_from_file_location("multitalk_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load config module from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _collect_weight_stats(
    root_dir: str, extensions: tuple[str, ...]
) -> Dict[str, Any]:
    non_empty_count = 0
    empty_files: List[str] = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            lowered = filename.lower()
            if not lowered.endswith(extensions):
                continue
            file_path = os.path.join(dirpath, filename)
            try:
                file_size = os.path.getsize(file_path)
            except OSError:
                continue
            if file_size > 0:
                non_empty_count += 1
            else:
                empty_files.append(file_path)

    return {
        "non_empty_count": non_empty_count,
        "empty_files": empty_files,
    }


def healthcheck(weight_extensions: Iterable[str] | None = None) -> Dict[str, Any]:
    """Return multitalk readiness details without importing/loading models."""

    repo_dir = _repo_dir()
    checks: List[Dict[str, Any]] = []
    errors: List[str] = []
    warnings: List[str] = []

    extensions = tuple(
        ext.lower() for ext in (tuple(weight_extensions) if weight_extensions else DEFAULT_WEIGHT_EXTENSIONS)
    )

    required_files = (
        "cli.py",
        "generate_multitalk.py",
        "base_tts_template.json",
    )
    for filename in required_files:
        path = os.path.join(repo_dir, filename)
        ok = os.path.isfile(path)
        check: Dict[str, Any] = {"name": filename, "path": path, "ok": ok}
        if not ok:
            detail = f"Missing file: {filename}"
            check["detail"] = detail
            errors.append(detail)
        checks.append(check)

    try:
        config_module = _load_config_module(repo_dir)
    except Exception as exc:
        errors.append(f"Failed to load multitalk config.py: {exc}")
        return {"ok": False, "checks": checks, "errors": errors, "warnings": warnings}

    path_keys = ("CKPT_DIR", "WAV2VEC_DIR", "KOKORO_DIR")
    for key in path_keys:
        raw_value = getattr(config_module, key, "")
        resolved = _resolve_path(repo_dir, raw_value) if raw_value else ""

        ok = bool(resolved) and os.path.isdir(resolved)
        check = {"name": key, "path": resolved or "<unset>", "ok": ok}

        if not ok:
            detail = f"{key} is missing or not a directory"
            check["detail"] = detail
            errors.append(detail)
            checks.append(check)
            continue

        stats = _collect_weight_stats(resolved, extensions)
        check["weight_files_found"] = stats["non_empty_count"]
        check["empty_files"] = len(stats["empty_files"])

        if stats["non_empty_count"] == 0:
            detail = f"{key} contains no non-empty weight files matching {extensions}"
            check["detail"] = detail
            errors.append(detail)

        if stats["empty_files"]:
            preview = stats["empty_files"][:5]
            warnings.append(
                f"{key} has {len(stats['empty_files'])} empty weight files (showing up to 5): {preview}"
            )

        checks.append(check)

    tts_voice_raw = getattr(config_module, "TTS_VOICE", "")
    tts_voice_path = _resolve_path(repo_dir, tts_voice_raw) if tts_voice_raw else ""
    tts_ok = bool(tts_voice_path) and os.path.isfile(tts_voice_path)
    tts_check: Dict[str, Any] = {
        "name": "TTS_VOICE",
        "path": tts_voice_path or "<unset>",
        "ok": tts_ok,
    }

    if not tts_ok:
        detail = "TTS_VOICE is missing or not a file"
        tts_check["detail"] = detail
        errors.append(detail)
    else:
        try:
            size_bytes = os.path.getsize(tts_voice_path)
            tts_check["size_bytes"] = size_bytes
            if size_bytes <= 0:
                detail = "TTS_VOICE exists but is empty (0 bytes)"
                tts_check["detail"] = detail
                tts_check["ok"] = False
                errors.append(detail)
        except OSError as exc:
            detail = f"Failed to stat TTS_VOICE: {exc}"
            tts_check["detail"] = detail
            tts_check["ok"] = False
            errors.append(detail)

    checks.append(tts_check)

    return {
        "ok": len(errors) == 0,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
    }
