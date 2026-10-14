import os
from pathlib import Path

import yaml

DATA_DIR_KEY = "DATA_DIR"
DATA_PLACEHOLDER = "$DATA_DIR"
USER_CONFIG_FILENAME = "data/configs/user_config.yaml"


def _resolve_user_config_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    user_config_path = repo_root / USER_CONFIG_FILENAME
    if not user_config_path.is_file():
        assert False, f"Missing {USER_CONFIG_FILENAME} at {user_config_path}."
    return user_config_path


def _load_data_dir() -> Path:
    user_config_path = _resolve_user_config_path()
    user_config = yaml.safe_load(user_config_path.read_text()) or {}
    if DATA_DIR_KEY not in user_config:
        assert False, f"DATA_DIR not set in {user_config_path}."
    data_dir = Path(os.path.expandvars(str(user_config[DATA_DIR_KEY]))).expanduser()
    if not data_dir.is_absolute():
        assert False, "DATA_DIR must be an absolute path."
    if not data_dir.exists():
        assert False, f"DATA_DIR path does not exist: {data_dir}"
    if not data_dir.is_dir():
        assert False, f"DATA_DIR is not a directory: {data_dir}"
    return data_dir


def _apply_data_dir(value, data_dir: Path):
    if isinstance(value, dict):
        return {k: _apply_data_dir(v, data_dir) for k, v in value.items()}
    if isinstance(value, list):
        return [_apply_data_dir(item, data_dir) for item in value]
    if isinstance(value, tuple):
        return tuple(_apply_data_dir(item, data_dir) for item in value)
    if isinstance(value, Path):
        return Path(str(value).replace(DATA_PLACEHOLDER, str(data_dir)))
    if isinstance(value, str):
        return value.replace(DATA_PLACEHOLDER, str(data_dir))
    return value


def resolve_path(path_value) -> Path:
    data_dir = _load_data_dir()
    resolved_value = _apply_data_dir(path_value, data_dir)

    if isinstance(resolved_value, str):
        resolved_path = Path(os.path.expandvars(resolved_value)).expanduser()
    elif isinstance(resolved_value, Path):
        resolved_path = Path(os.path.expandvars(str(resolved_value))).expanduser()
    else:
        assert False, f"Path must be a string or Path, got {type(resolved_value)}"

    #if not resolved_path.is_absolute():
    #    assert False, f"Resolved path must be absolute: {resolved_path}"

    return resolved_path


def load_config(config_path) -> dict:
    path = Path(config_path)
    if not path.is_file():
        assert False, f"Config file does not exist: {path}"
    data_dir = _load_data_dir()
    config = yaml.safe_load(path.read_text())
    return _apply_data_dir(config, data_dir)