"""Write an exact machine-readable inventory for the active environment."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = PROJECT_ROOT / "requirements-lock.txt"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "provenance" / "environment_inventory_python313_v1.json"
)


def main() -> int:
    output = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_OUTPUT
    packages = sorted(
        (
            {"name": distribution.metadata["Name"], "version": distribution.version}
            for distribution in importlib.metadata.distributions()
            if distribution.metadata["Name"]
        ),
        key=lambda item: item["name"].casefold(),
    )
    inventory = {
        "schema_version": 1,
        "record": "data002_python_environment_inventory_v1",
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "requirements_lock": {
            "path": "requirements-lock.txt",
            "sha256": hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest(),
        },
        "packages": packages,
        "package_count": len(packages),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
