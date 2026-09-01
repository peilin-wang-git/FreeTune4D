"""Lightweight CUDA device discovery without initializing CUDA in the GUI."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import shutil
import subprocess
import json
import sys


GIB = 1024**3


@dataclass(frozen=True)
class DeviceInfo:
    physical_index: int | None
    name: str
    total_bytes: int
    free_bytes: int
    utilization_percent: int | None = None
    device_type: str = "cuda"
    available: bool = True
    supported: bool = True

    @property
    def key(self) -> str:
        return "cpu" if self.device_type == "cpu" else f"cuda:{self.physical_index}"

    @property
    def total_gib(self) -> float:
        return self.total_bytes / GIB

    @property
    def free_gib(self) -> float:
        return self.free_bytes / GIB

    @property
    def low_memory(self) -> bool:
        # A conservative warning, not a claimed pipeline requirement.
        return self.device_type == "cuda" and (
            self.free_gib < 1.0 or self.free_bytes < self.total_bytes * 0.05
        )

    @property
    def display_name(self) -> str:
        if self.device_type == "cpu":
            return "CPU — slower" if self.supported else "CPU — unavailable in current backend"
        return (
            f"GPU {self.physical_index} — {self.name} — "
            f"{self.free_gib:.2f}/{self.total_gib:.2f} GiB free"
        )


def cpu_device() -> DeviceInfo:
    """Return the always-present CPU option with an honest support state."""
    return DeviceInfo(
        physical_index=None,
        name="CPU",
        total_bytes=0,
        free_bytes=0,
        device_type="cpu",
        available=True,
        supported=True,
    )


def detect_cuda_devices(timeout: float = 4.0) -> list[DeviceInfo]:
    """Query nvidia-smi, gracefully returning no GPUs when unavailable."""
    executable = shutil.which("nvidia-smi")
    if not executable:
        return _detect_with_torch_subprocess(timeout)
    command = [
        executable,
        "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return _detect_with_torch_subprocess(timeout)

    devices: list[DeviceInfo] = []
    try:
        rows = csv.reader(result.stdout.splitlines(), skipinitialspace=True)
        for row in rows:
            if len(row) < 4:
                continue
            utilization = int(row[4].strip()) if len(row) > 4 and row[4].strip().isdigit() else None
            devices.append(DeviceInfo(
                physical_index=int(row[0].strip()),
                name=row[1].strip(),
                total_bytes=int(row[2].strip()) * 1024**2,
                free_bytes=int(row[3].strip()) * 1024**2,
                utilization_percent=utilization,
            ))
    except (ValueError, csv.Error):
        return _detect_with_torch_subprocess(timeout)
    return sorted(devices, key=lambda device: device.physical_index)


def _detect_with_torch_subprocess(timeout: float) -> list[DeviceInfo]:
    """Fallback discovery isolated from the long-lived GUI process."""
    program = """
import json, torch
items = []
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            free, total = torch.cuda.mem_get_info(index)
        items.append({"index": index, "name": torch.cuda.get_device_name(index), "total": total, "free": free})
print(json.dumps(items))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True,
            timeout=timeout, check=True,
        )
        data = json.loads(result.stdout.strip())
        return [DeviceInfo(
            physical_index=int(item["index"]), name=str(item["name"]),
            total_bytes=int(item["total"]), free_bytes=int(item["free"]),
        ) for item in data]
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return []


def select_device(selection: str, devices: list[DeviceInfo]) -> DeviceInfo | None:
    """Resolve ``auto`` or a stable cuda key to a detected physical GPU."""
    if selection == "auto":
        compatible_gpus = [
            device for device in devices
            if device.device_type == "cuda" and device.available and device.supported
        ]
        if compatible_gpus:
            return max(compatible_gpus, key=lambda device: device.free_bytes)
        return next(
            (device for device in devices if device.device_type == "cpu" and device.available and device.supported),
            None,
        )
    return next((device for device in devices if device.key == selection), None)
