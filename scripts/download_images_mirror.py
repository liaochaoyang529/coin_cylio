#!/usr/bin/env python3
"""Download and verify Coin Challenge images from an HF-compatible mirror."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="e-zorzi/images_coin_challenge")
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--output-dir", type=Path, default=Path("images"))
    parser.add_argument("--staging-dir", type=Path, default=Path(".images-download"))
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def required_filenames() -> list[str]:
    filenames = set()
    for jsonl_path in Path(".").glob("episodes_*.jsonl"):
        with jsonl_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                episode = json.loads(line)
                filenames.add(Path(episode["path"]).name)
                filenames.update(Path(item["path"]).name for item in episode["distractors"])
    if not filenames:
        raise RuntimeError("No episode JSONL files found.")
    return sorted(filenames)


def verify_image(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()


def main() -> int:
    args = parse_args()
    filenames = required_filenames()
    args.staging_dir.mkdir(parents=True, exist_ok=True)

    def download(filename: str) -> str:
        destination = args.staging_dir / filename
        if destination.exists():
            try:
                verify_image(destination)
                return "cached"
            except Exception:
                destination.unlink()

        url = (
            f"{args.endpoint.rstrip('/')}/datasets/{args.repo_id}/resolve/main/"
            f"{quote(filename)}?download=true"
        )
        temporary = destination.with_suffix(".part")
        for attempt in range(5):
            try:
                request = Request(url, headers={"User-Agent": "coin-challenge-downloader"})
                with urlopen(request, timeout=90) as response, temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)
                verify_image(temporary)
                os.replace(temporary, destination)
                return "downloaded"
            except Exception:
                temporary.unlink(missing_ok=True)
                if attempt == 4:
                    raise
                time.sleep(attempt + 1)
        raise AssertionError("unreachable")

    downloaded = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download, name): name for name in filenames}
        for index, future in enumerate(as_completed(futures), start=1):
            if future.result() == "downloaded":
                downloaded += 1
            if index % 25 == 0 or index == len(filenames):
                print(f"processed {index}/{len(filenames)}", flush=True)

    for filename in filenames:
        verify_image(args.staging_dir / filename)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        os.replace(args.staging_dir / filename, args.output_dir / filename)
    args.staging_dir.rmdir()
    print(f"Installed {len(filenames)} verified images ({downloaded} downloaded this run).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
