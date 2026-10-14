"""Build a dataset yaml (motion files with sampling weights) from a create_dataset config.

    python -m scripts.create_dataset_yaml --config data/configs/create_dataset_config_g1.yaml

See parc/util/create_dataset.py for the config keys. With compute_preprocessing_data,
every motion file read is re-saved in place with its heightfield mask data.
"""
import argparse
import os
import sys
from pathlib import Path

# Ensure the repository root is on the import path so "parc" resolves consistently
# across platforms (editable installs on Windows sometimes fail to place the repo
# ahead of site-packages when invoking this script directly).
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import parc.util.create_dataset as create_dataset
import parc.util.path_loader as path_loader


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build a dataset yaml from folders of motion files.")
    parser.add_argument("--config", default="data/configs/create_dataset_config.yaml",
                        help="create_dataset config yaml (default: %(default)s)")
    cfg_path = parser.parse_args(argv).config
    print("loading create dataset config from", cfg_path)

    try:
        config = path_loader.load_config(cfg_path)
    except (IOError, AssertionError) as exc:
        print("error opening file:", cfg_path)
        print(exc)
        print("Current working directory:", os.getcwd())
        return 1

    create_dataset.create_dataset_yaml_from_config(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
