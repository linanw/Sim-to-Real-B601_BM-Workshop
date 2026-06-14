# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preflight checks for Isaac Sim launch scripts."""

from __future__ import annotations

import os
import subprocess
import sys


def _get_driver_info() -> tuple[str, str] | None:
    """Return the first visible GPU name and NVIDIA driver version."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,name",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None

    first_gpu = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    if not first_gpu:
        return None

    parts = [part.strip() for part in first_gpu.split(",", maxsplit=1)]
    if len(parts) != 2:
        return None
    return parts[1], parts[0]


def guard_known_bad_isaacsim_driver() -> None:
    """Stop early on NVIDIA driver branches known to crash Isaac Sim 5.1 RTX startup."""
    if os.getenv("SIM_TO_REAL_SKIP_DRIVER_PREFLIGHT"):
        return

    driver_info = _get_driver_info()
    if driver_info is None:
        return

    gpu_name, driver_version = driver_info
    if not driver_version.startswith("595."):
        return

    message = f"""
[ERROR]: Refusing to launch Isaac Sim with NVIDIA driver {driver_version} on {gpu_name}.

Isaac Sim 5.1 is known to crash in the RTX scene database startup path
(`librtx.scenedb.plugin.so`) with the 595.xx driver branch. This matches
the native segfault seen before Python reaches the SO-101 task code.

Install the Isaac Sim 5.1 tested Linux driver branch instead, for example
580.65.06, then rerun this command. To bypass this guard anyway, set:

    export SIM_TO_REAL_SKIP_DRIVER_PREFLIGHT=1
"""
    print(message.strip(), file=sys.stderr)
    raise SystemExit(1)
