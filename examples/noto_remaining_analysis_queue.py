from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

POLL_SECONDS = 30
STILL_ACTIVE = 259
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def process_is_running(process_id: int) -> bool:
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(
        PROCESS_QUERY_LIMITED_INFORMATION,
        False,
        process_id,
    )
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def read_status(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def run_step(
    name: str,
    script: Path,
    output_dir: Path,
    workers: int,
    queue_root: Path,
) -> None:
    status_path = queue_root / "queue_status.json"
    stdout_path = queue_root / f"{name}.stdout.log"
    stderr_path = queue_root / f"{name}.stderr.log"
    atomic_json(
        status_path,
        {
            "status": "running",
            "step": name,
            "script": str(script),
            "started": time.time(),
        },
    )
    command = [
        sys.executable,
        "-u",
        "-B",
        str(script),
        "--output-dir",
        str(output_dir),
        "--workers",
        str(workers),
    ]
    with stdout_path.open("a", encoding="utf-8") as stdout_file:
        with stderr_path.open("a", encoding="utf-8") as stderr_file:
            completed = subprocess.run(
                command,
                cwd=script.parent.parent,
                stdout=stdout_file,
                stderr=stderr_file,
                check=False,
            )
    if completed.returncode != 0:
        atomic_json(
            status_path,
            {
                "status": "failed",
                "step": name,
                "return_code": completed.returncode,
                "stderr": str(stderr_path),
                "updated": time.time(),
            },
        )
        raise RuntimeError(
            f"{name} failed with return code {completed.returncode}."
        )


def main(
    output_dir: Path,
    mechanism_process_id: int,
    workers: int,
) -> None:
    queue_root = output_dir / "remaining_analysis_queue_v1"
    queue_root.mkdir(parents=True, exist_ok=True)
    queue_status = queue_root / "queue_status.json"
    mechanism_status = (
        output_dir / "mechanism_separated_capability_marginal_v2" / "status.json"
    )

    while True:
        status = read_status(mechanism_status)
        if status.get("status") == "completed":
            break
        running = process_is_running(mechanism_process_id)
        atomic_json(
            queue_status,
            {
                "status": "waiting",
                "waiting_for": "M0-M4 full-grid mechanism analysis",
                "mechanism_process_id": mechanism_process_id,
                "mechanism_process_running": running,
                "mechanism_status": status,
                "updated": time.time(),
            },
        )
        if not running:
            atomic_json(
                queue_status,
                {
                    "status": "blocked",
                    "reason": (
                        "Mechanism process exited without a completed status."
                    ),
                    "mechanism_process_id": mechanism_process_id,
                    "mechanism_status": status,
                    "updated": time.time(),
                },
            )
            raise RuntimeError(
                "Mechanism process exited without completing its grid."
            )
        time.sleep(POLL_SECONDS)

    repo = Path(__file__).resolve().parent.parent
    steps = (
        (
            "selected_sensitivity_separated_capability_marginal",
            repo / "examples" / "noto_selected_sensitivity_full_grid.py",
        ),
        (
            "stage2_joint_full_grid",
            repo / "examples" / "noto_stage2_joint_full_grid.py",
        ),
    )
    for name, script in steps:
        run_step(name, script, output_dir, workers, queue_root)

    atomic_json(
        queue_status,
        {
            "status": "completed",
            "completed_steps": [name for name, _ in steps],
            "updated": time.time(),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mechanism-process-id", required=True, type=int)
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    main(
        Path(arguments.output_dir),
        arguments.mechanism_process_id,
        arguments.workers,
    )
