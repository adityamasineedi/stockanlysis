"""Pull persistent data from the Railway volume to local data/.

Uses `railway volume files download` (works while the service is stopped).
Does not require SSH into a running container.

Usage:
  uv run python scripts/pull_railway_data.py
  uv run python scripts/pull_railway_data.py --volume stockanlysis-volume
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from stockbot.config import DB_PATH, PORTFOLIO_DIR, PROJECT_ROOT

logger = logging.getLogger(__name__)

DEFAULT_VOLUME_NAME = "stockanlysis-volume"

# Remote paths are relative to the volume root (mounted at /app/data in the container).
REMOTE_FILES: tuple[tuple[str, Path], ...] = (
    ("/db/analyses.sqlite3", DB_PATH),
    ("/portfolio/prescan_outcomes.jsonl", PORTFOLIO_DIR / "prescan_outcomes.jsonl"),
)


def _railway_bin() -> str:
    path = shutil.which("railway") or shutil.which("railway.exe")
    if path is None:
        raise RuntimeError(
            "Railway CLI not found. Install: https://docs.railway.com/guides/cli"
        )
    return path


def _resolve_volume_id(railway: str, volume_name: str) -> str:
    proc = subprocess.run(
        [railway, "volume", "list", "--json"],
        capture_output=True,
        text=True,
        check=False,
        cwd=PROJECT_ROOT,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"railway volume list failed: {proc.stderr.strip()}")
    payload = json.loads(proc.stdout)
    volumes = payload.get("volumes") or payload
    if isinstance(volumes, dict):
        volumes = volumes.get("volumes", [])
    for item in volumes:
        if item.get("name") == volume_name or item.get("id") == volume_name:
            return str(item["id"])
    raise RuntimeError(f"Volume not found: {volume_name}")


def download_file(railway: str, volume_id: str, remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        railway,
        "volume",
        "files",
        "-v",
        volume_id,
        "download",
        remote,
        str(local),
        "--overwrite",
    ]
    logger.info("Downloading %s -> %s", remote, local)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=PROJECT_ROOT)
    if proc.returncode != 0:
        raise RuntimeError(
            f"download {remote} failed: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    if not local.exists() or local.stat().st_size == 0:
        raise RuntimeError(f"download {remote} produced empty file at {local}")


def pull_railway_data(
    *,
    volume_name: str = DEFAULT_VOLUME_NAME,
    files: tuple[tuple[str, Path], ...] = REMOTE_FILES,
) -> list[Path]:
    railway = _railway_bin()
    volume_id = _resolve_volume_id(railway, volume_name)
    downloaded: list[Path] = []
    for remote, local in files:
        download_file(railway, volume_id, remote, local)
        downloaded.append(local)
    return downloaded


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Sync Railway volume data to local data/")
    parser.add_argument(
        "--volume",
        default=DEFAULT_VOLUME_NAME,
        help=f"Railway volume name or ID (default: {DEFAULT_VOLUME_NAME})",
    )
    args = parser.parse_args()
    try:
        paths = pull_railway_data(volume_name=args.volume)
    except Exception as exc:
        print(f"Pull failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    for path in paths:
        size_kb = path.stat().st_size / 1024
        print(f"OK  {path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
