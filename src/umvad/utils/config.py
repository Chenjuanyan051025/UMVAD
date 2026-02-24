from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to load YAML configs.") from exc
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif suffix == ".json":
        import json

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {path}")
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping/dict.")
    return data

