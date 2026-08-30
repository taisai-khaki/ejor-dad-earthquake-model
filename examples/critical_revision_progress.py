from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=10.0)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    while True:
        rows = []
        for path in sorted(root.rglob("status.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            block_root = path.parent
            checkpoint_count = len(list(block_root.rglob("*.json"))) if block_root.exists() else 0
            rows.append({"path": str(path.relative_to(root)), "status": payload.get("status"), "block": payload.get("block"), "rho": payload.get("rho"), "completed": payload.get("completed", payload.get("completed_rho", payload.get("rows"))), "total": payload.get("total", payload.get("total_cube_rows")), "checkpoint_count": checkpoint_count, "updated_at_epoch": payload.get("updated_at_epoch")})
        for checkpoint_dir in sorted(root.glob("continuous_bb/radii/*/corner_checkpoints")):
            count = len(list(checkpoint_dir.glob("*.json")))
            rows.append({"path": str(checkpoint_dir.relative_to(root)), "status": "checkpointing", "block": "continuous_bb", "rho": checkpoint_dir.parent.name, "completed": count, "total": 647, "checkpoint_count": count, "updated_at_epoch": checkpoint_dir.stat().st_mtime})
        summary = {"updated_at_epoch": time.time(), "root": str(root), "blocks": rows}
        root.mkdir(parents=True, exist_ok=True)
        (root / "progress_snapshot.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

