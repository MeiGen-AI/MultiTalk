"""
@file cli.py
@brief Backend-facing CLI wrapper for the MultiTalk generator.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid
from typing import Any, Dict, Tuple
from urllib.parse import urlparse

import requests

import config

# Characters per second for speech duration estimation (matches app/services/eta_service.py)
CHARS_PER_SECOND = 15.0


def _estimate_speech_duration_seconds(speech_text: str) -> float | None:
    """
    @brief Estimate video duration in seconds from speech text using character count.
    @param speech_text The speech text to estimate duration for.
    @return Estimated duration in seconds, or None if speech_text is empty/missing.
    """
    if not speech_text or not isinstance(speech_text, str):
        return None
    
    # Estimate duration: characters / characters_per_second
    duration = len(speech_text) / CHARS_PER_SECOND
    return duration


def _run_command_streaming(command: list[str], cwd: str) -> None:
    """
    @brief Run a subprocess while streaming stdout/stderr to the current process.
    @param command Command list to execute.
    @param cwd Working directory for the subprocess.
    @throws RuntimeError when the command exits non-zero.
    """

    proc = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        tail.append(line)
        if len(tail) > 200:
            tail.pop(0)

    rc = proc.wait()
    if rc != 0:
        tail_text = "".join(tail).strip()
        raise RuntimeError(
            f"multitalk generation failed with exit code {rc}. Last output:\n{tail_text}"
        )


def _resolve_path(base_dir: str, path_value: str) -> str:
    """
    @brief Resolve a path relative to the multitalk repo if needed.
    @param base_dir Base directory for relative resolution.
    @param path_value Path to resolve.
    @return Absolute path for the input value.
    """

    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(base_dir, path_value))


def _ensure_kokoro_weights(repo_dir: str) -> None:
    """
    @brief Ensure the Kokoro weights are reachable via `weights/Kokoro-82M` from repo_dir.
    @details MultiTalk's `generate_multitalk.py` uses a relative repo_id (`weights/Kokoro-82M`),
             so we provide a symlink to the absolute directory configured in `config.KOKORO_DIR`.
    @param repo_dir MultiTalk repo directory.
    """

    kokoro_dir = getattr(config, "KOKORO_DIR", "")
    if not kokoro_dir:
        return

    weights_dir = os.path.join(repo_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    link_path = os.path.join(weights_dir, "Kokoro-82M")

    if os.path.exists(link_path):
        return

    try:
        os.symlink(kokoro_dir, link_path)
    except Exception:
        # If symlinks are not permitted, fall back to doing nothing; the generator will error clearly.
        pass


def _select_avatar_assets(avatar_dir: str) -> Tuple[str, str]:
    """
    @brief Select the avatar configuration JSON and image file from a directory.
    @details Finds the first .json file (assumed to be base.json or config) and
             the first image file (.png, .jpg, .jpeg, or .webp) in the directory.
             Files are selected in sorted order, so naming is predictable.
    @param avatar_dir Directory containing avatar assets.
    @return Tuple of (json_config_path, image_file_path).
    @throws RuntimeError if required files are missing.
    """

    if not os.path.isdir(avatar_dir):
        raise RuntimeError(f"Avatar directory not found: {avatar_dir}")

    entries = sorted(os.listdir(avatar_dir))
    json_path = ""
    image_path = ""
    for name in entries:
        path = os.path.join(avatar_dir, name)
        if os.path.isdir(path):
            continue
        lower = name.lower()
        if lower.endswith(".json") and not json_path:
            json_path = path
        elif lower.endswith((".png", ".jpg", ".jpeg", ".webp")) and not image_path:
            image_path = path

    if not json_path or not image_path:
        raise RuntimeError(
            f"Avatar directory must contain one JSON config file and one image file (.png/.jpg/.jpeg/.webp): {avatar_dir}"
        )

    return json_path, image_path

def _resolve_voice_path(preferred_voice: str | None, base_dir: str) -> str:
    """
    @brief Resolve a preferred voice identifier to an absolute voice file path.
    @details Accepts a full relative path (e.g. ``weights/Kokoro-82M/voices/af_heart.pt``),
             a filename (``af_heart.pt``), or just a voice name (``af_heart``).
    @param preferred_voice Voice identifier supplied by the caller. May be *None*.
    @param base_dir Base directory for relative path resolution (multitalk repo root).
    @return Absolute path to the voice ``.pt`` file.
    """

    if not preferred_voice:
        return _resolve_path(base_dir, config.TTS_VOICE)

    # Already looks like a path (contains separator or ends with .pt)
    if "/" in preferred_voice or preferred_voice.endswith(".pt"):
        return _resolve_path(base_dir, preferred_voice)

    # Bare voice name → resolve inside the Kokoro voices directory.
    voice_path = f"weights/Kokoro-82M/voices/{preferred_voice}.pt"
    return _resolve_path(base_dir, voice_path)


def _build_input_from_template(
    data: Dict[str, Any], base_dir: str, avatar_image_path: str
) -> Dict[str, Any]:
    """
    @brief Build an input payload using the checked-in base TTS template.
    @details Used when the caller supplies a ``avatar`` name (found in S3 bucket under ``avatars/``)
             The template's ``cond_image``, ``tts_audio.text``, and ``tts_audio.human1_voice``
             are replaced with the caller-provided values.
    @param data Raw job data dictionary.
    @param base_dir Multitalk repo directory (for path resolution).
    @param avatar_image_path Absolute path to the downloaded avatar image.
    @return Payload dictionary ready for multitalk ``input_json``.
    @throws RuntimeError when required fields are missing.
    """

    template_path = os.path.join(base_dir, "base_tts_template.json")
    payload = _load_json(template_path)

    # Replace the avatar image with the downloaded one.
    payload["cond_image"] = avatar_image_path

    if "cond_audio" not in payload:
        payload["cond_audio"] = {}

    # Speech text is mandatory.
    speech_text = data.get("speech_text")
    if not speech_text:
        raise RuntimeError("speech_text is required for multitalk TTS mode")

    # Resolve the voice file path.
    preferred_voice = data.get("preferredVoice")
    voice_path = _resolve_voice_path(preferred_voice, base_dir)

    payload["tts_audio"] = {
        "text": speech_text,
        "human1_voice": voice_path,
    }

    return payload


def _build_input_payload(
    data: Dict[str, Any], base_dir: str,
) -> Dict[str, Any]:
    """
    @brief Build the input payload expected by the multitalk generator.
    @details Transforms avatar config JSON into the multitalk generator format:
             - Extracts prompt and voice from avatar JSON
             - Resolves image path to absolute path
             - Converts voice name to path: weights/Kokoro-82M/voices/{voice}.pt
             - Combines with speech_text from job data into tts_audio
    @param data Raw job data from the backend JSON file.
    @param base_dir Base directory for resolving paths (repo directory).
    @return Payload dictionary ready for multitalk input_json.
    @throws RuntimeError when required fields are missing.
    """

    # Extract required fields from avatar config
    prompt = data.get("video_prompt")  # Fallback to default prompt if not specified in avatar config
    if not prompt:
        raise RuntimeError(f"Job data must contain 'video_prompt' field: {data}")
    
    voice = data.get("kokoro_voice")
    if not voice:
        raise RuntimeError(f"Job data must contain 'kokoro_voice' field: {data}")

    # Get speech text from job data
    speech_text = data.get("speech_text")
    if not speech_text:
        raise RuntimeError(f"Job data must contain 'speech_text' field: {data}")
    
    avatar_path = data.get("avatar_path")
    if not avatar_path:
        raise RuntimeError(f"Job data must contain 'avatar_path' field: {data}")

    # Build the payload in the expected format
    payload = {
        "prompt": prompt,
        "cond_image": _resolve_path(base_dir, avatar_path),
        "tts_audio": {
            "text": speech_text,
            "human1_voice": _resolve_path(base_dir, f"weights/Kokoro-82M/voices/{voice}.pt"),
        },
        "cond_audio": {},
    }
    
    return payload


def _load_json(path: str) -> Dict[str, Any]:
    """
    @brief Load JSON content from disk.
    @param path File path to read.
    @return Parsed JSON dictionary.
    @throws RuntimeError if JSON cannot be read.
    """

    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        raise RuntimeError(f"Failed to read JSON from {path}: {exc}") from exc


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    """
    @brief Write JSON content to disk.
    @param path File path to write.
    @param payload JSON-serializable dictionary.
    """

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def main() -> None:
    """
    @brief CLI entrypoint for backend-triggered multitalk generation.
    @throws RuntimeError on invalid input or generation failure.
    """

    parser = argparse.ArgumentParser(
        description="Backend wrapper for MultiTalk generation",
    )
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--work-dir", default=None)
    args = parser.parse_args()

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    
    work_dir = args.work_dir
    if work_dir is None:
        print('WARNING: no --work-dir specified, using default "backend_runs/{job_id}"')
        work_dir = os.path.join(repo_dir, "backend_runs", args.job_id)
    os.makedirs(work_dir, exist_ok=True)

    input_json_path = os.path.join(work_dir, f"cond.json")
    audio_save_dir = os.path.join(work_dir, "audio")

    data = _load_json(args.data)
    payload = _build_input_payload(
        data=data,
        base_dir=repo_dir,
    )
    _write_json(input_json_path, payload)

    ckpt_dir = config.CKPT_DIR
    wav2vec_dir = config.WAV2VEC_DIR
    if not ckpt_dir or not wav2vec_dir:
        raise RuntimeError("Missing CKPT_DIR or WAV2VEC_DIR in config.py")

    ckpt_dir = _resolve_path(repo_dir, ckpt_dir)
    wav2vec_dir = _resolve_path(repo_dir, wav2vec_dir)

    _ensure_kokoro_weights(repo_dir)

    # Calculate frame_num and max_frames_num based on video duration
    FPS = 25.0  # Video frames per second
    speech_text = data.get("speech_text", "")
    video_duration_seconds = (
        _estimate_speech_duration_seconds(speech_text) if speech_text else None
    )

    # Choose the largest safe frame_num for this text, with safety margin.
    # Keep it in [33, 81] and enforce frame_num = 4n+1.
    MIN_FRAMES = 33   # 4*8 + 1
    MAX_FRAMES = 81   # 4*20 + 1

    if video_duration_seconds is not None:
        # Estimated frames for the (TTS) audio, with a small safety margin
        frames_estimated = int(video_duration_seconds * FPS)
        frames_target = int(frames_estimated * 0.9) # 10% safety margin

        # Clamp into [MIN_FRAMES, MAX_FRAMES]
        frames_target = max(MIN_FRAMES, min(MAX_FRAMES, frames_target))

        # frame_num must be 4n+1 -> round DOWN to nearest 4n+1 <= frames_target
        remainder = (frames_target - 1) % 4
        frame_num = frames_target - remainder
        if frame_num < MIN_FRAMES:
            frame_num = MIN_FRAMES
    else:
        # Unknown duration: fall back to the most conservative (max) clip length
        frame_num = MAX_FRAMES

    # # Calculate max_frames_num for longer videos
    mode = "streaming"
    # if mode == "clip":
    #     max_frames_num = frame_num
    # else:
    #     # Streaming mode: calculate frames needed based on video duration
    #     if video_duration_seconds is not None:
    #         # Calculate frames needed: duration * fps
    #         frames_needed = int(video_duration_seconds * FPS)
    #         # Ensure it's at least frame_num
    #         max_frames_num = max(frame_num, frames_needed)
    #         # Round up to next 4n+1 if needed (to match frame_num pattern)
    #         remainder = (max_frames_num - 1) % 4
    #         if remainder != 0:
    #             max_frames_num = max_frames_num + (4 - remainder)
    #     else:
    #         max_frames_num = 1000  # default for streaming when duration unknown
    max_frames_num = 2000

    if frames_estimated > max_frames_num:
        raise RuntimeError(f"Estimated frames {frames_estimated} is greater than max_frames_num {max_frames_num}. "
                            "Max runtime will be exceeded")

    command = [
        sys.executable,
        os.path.join(repo_dir, "generate_multitalk.py"),
        "--ckpt_dir",
        ckpt_dir,
        "--wav2vec_dir",
        wav2vec_dir,
        "--input_json",
        input_json_path,
        "--sample_steps",
        str(data.get("sample_steps", getattr(config, "SAMPLE_STEPS", 40))),
        "--mode",
        mode,
        "--num_persistent_param_in_dit",
        str(data.get("num_persistent_param_in_dit", 0)),
        "--audio_mode",
        "tts",
        "--audio_save_dir",
        audio_save_dir,
        "--save_file",
        os.path.splitext(args.output)[0],
        "--frame_num",
        str(frame_num),
    ]

    # Add max_frames_num argument for streaming mode
    if mode == "streaming":
        command.extend(["--max_frames_num", str(max_frames_num)])

    if data.get("use_teacache", True):
        command.append("--use_teacache")

    try:
        _run_command_streaming(command, cwd=repo_dir)
    finally:
        try:
            shutil.rmtree(work_dir)
            # print(f"Skipping cleanup of work_dir: {work_dir}\n command ran \n {command}")
        except Exception:
            pass


if __name__ == "__main__":
    main()

