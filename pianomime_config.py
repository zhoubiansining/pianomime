from __future__ import annotations

import os
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility.
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "baseline.toml"


def _raw_load(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _format_value(value: Any, context: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        previous = None
        current = value
        for _ in range(5):
            if current == previous:
                break
            previous = current
            current = current.format(**context)
        return current
    if isinstance(value, list):
        return [_format_value(item, context) for item in value]
    if isinstance(value, dict):
        return {key: _format_value(item, context) for key, item in value.items()}
    return value


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path or os.environ.get("CONFIG_FILE", DEFAULT_CONFIG_PATH)).expanduser()
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    cfg = _raw_load(config_path)

    paths = dict(cfg.get("paths", {}))
    context = {
        "home": str(Path.home()),
        "project_root": str(PROJECT_ROOT),
        "project_parent": str(PROJECT_ROOT.parent),
    }
    for _ in range(5):
        changed = False
        for key, value in list(paths.items()):
            if isinstance(value, str):
                expanded = value.format(**(context | paths))
                if expanded != value:
                    paths[key] = expanded
                    changed = True
        context |= {key: str(value) for key, value in paths.items()}
        if not changed:
            break
    cfg["paths"] = paths
    context |= {key: str(value) for key, value in paths.items()}
    return _format_value(cfg, context)


def section(config: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping):
            return {}
        value = value.get(key, {})
    return dict(value) if isinstance(value, Mapping) else {}


def shell_default(name: str, value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        value = "1" if value else "0"
    elif isinstance(value, (list, tuple)):
        value = " ".join(str(item) for item in value)
    else:
        value = str(value)
    return f'if [[ -z "${{{name}:-}}" ]]; then export {name}={shlex.quote(value)}; fi'


def cli_args_from_mapping(values: Mapping[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in values.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            args.append(flag if value else "--no-" + key.replace("_", "-"))
        elif isinstance(value, (list, tuple)):
            args.append(flag)
            args.extend(str(item) for item in value)
        elif value is not None:
            args.extend([flag, str(value)])
    return args
