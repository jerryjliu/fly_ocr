import json
import platform
import shutil
import psutil

free = shutil.disk_usage(".").free / 2**30
report = {
    "platform": platform.platform(), "architecture": platform.machine(),
    "ram_gib": round(psutil.virtual_memory().total / 2**30, 2),
    "free_disk_gib": round(free, 2), "planned_cache_gib": 5,
    "reserve_gib": 10, "compiler": shutil.which("clang++") or shutil.which("g++"),
    "pdftoppm": shutil.which("pdftoppm"), "ffmpeg": shutil.which("ffmpeg"),
}
print(json.dumps(report, indent=2))
if free < report["planned_cache_gib"] + report["reserve_gib"]:
    raise SystemExit("Insufficient disk reserve; configure a different cache location.")
