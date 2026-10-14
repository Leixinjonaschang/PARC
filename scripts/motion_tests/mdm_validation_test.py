"""MDM validation test: generates motions along the path of each test terrain and saves
them as MS files (one per generated motion, sorted by loss).

    python -m scripts.motion_tests.mdm_validation_test --config scripts/motion_tests/mdm_validation_test.yaml

Config (paths may use $DATA_DIR):
    input_terrains  yaml whose `terrains` is an MS file path or a list of them; each file
                    needs terrain data and misc_data["path_nodes"] (legacy terrain pickles:
                    convert them with scripts/convert_legacy_motions.py)
    mdm_model       MDM checkpoint (.ckpt)
    output_folder   where <terrain name>_<i>.pkl are written
    device          optional; default cuda:0 if available, else cpu
    use_ema_model   optional; default False
    mdm_path        MDMPathSettings fields
    mdm_gen         MDMGenSettings fields
"""
import argparse
import os
import sys
import time
from typing import List

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch  # noqa: E402

import parc.motion_generator.gen_util as gen_util  # noqa: E402
import parc.motion_generator.mdm as mdm  # noqa: E402
import parc.motion_synthesis.procgen.mdm_path as mdm_path  # noqa: E402
import parc.util.file_io as file_io  # noqa: E402
import parc.util.file_io_helper as file_io_helper  # noqa: E402
import parc.util.path_loader as path_loader  # noqa: E402
import parc.util.terrain_util as terrain_util  # noqa: E402

DEFAULT_CONFIG = "scripts/motion_tests/mdm_validation_test.yaml"
cpu_device = "cpu"


def load_config(cfg_path) -> dict:
    return path_loader.load_config(path_loader.resolve_path(cfg_path))


def load_terrain_file_list(input_terrains) -> List[str]:
    """Reads the `terrains` entry of the input_terrains yaml: one path or a list of paths."""
    terrain_cfg = path_loader.load_config(path_loader.resolve_path(input_terrains))
    terrain_files = terrain_cfg["terrains"]
    if isinstance(terrain_files, str):
        terrain_files = [terrain_files]
    return [str(path_loader.resolve_path(f)) for f in terrain_files]


def load_terrain_and_path_nodes(terrain_file, device=cpu_device):
    """Returns (terrain, path_nodes [num_nodes, 3]) from an MS file."""
    file_data = file_io_helper.load_ms_file(terrain_file, device=device)
    if file_data.terrain_data is None:
        raise ValueError("{}: no terrain data".format(terrain_file))
    misc_data = file_data.misc_data if file_data.misc_data is not None else dict()
    if file_io_helper.PATH_NODES_KEY not in misc_data:
        raise ValueError("{}: no path_nodes in misc_data".format(terrain_file))
    terrain = terrain_util.SubTerrain.from_ms_terrain_data(file_data.terrain_data, device=device)
    return terrain, misc_data[file_io_helper.PATH_NODES_KEY]


def apply_settings(settings, settings_config):
    for key in settings_config:
        setattr(settings, key, settings_config[key])
    return settings


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate MDM motions along the paths of test terrains.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    print("loading mdm validation config from", args.config)
    config = load_config(args.config)

    device = config.get("device") or ("cuda:0" if torch.cuda.is_available() else cpu_device)
    mdm_model = mdm.MDM.load_checkpoint(config["mdm_model"], device=device,
                                        use_ema_model=config.get("use_ema_model", False))

    output_folder = str(path_loader.resolve_path(config["output_folder"]))
    os.makedirs(output_folder, exist_ok=True)

    terrain_files = load_terrain_file_list(config["input_terrains"])

    mdm_path_settings = apply_settings(mdm_path.MDMPathSettings(), config["mdm_path"])
    mdm_gen_settings = apply_settings(gen_util.MDMGenSettings(), config["mdm_gen"])

    torch.manual_seed(0)
    for terrain_file in terrain_files:
        terrain_name = os.path.splitext(os.path.basename(terrain_file))[0]
        try:
            terrain, path_nodes = load_terrain_and_path_nodes(terrain_file)
        except file_io.MSFileFormatError as exc:
            sys.exit("error: {}".format(exc))
        print("Generating for:", terrain_file)

        start_time = time.time()

        gen_motion_frames, out_terrains, info = mdm_path.generate_frames_until_end_of_path(
            path_nodes=path_nodes,
            terrain=terrain,
            char_model=mdm_model._kin_char_model,
            mdm_model=mdm_model,
            prev_frames=None,
            mdm_path_settings=mdm_path_settings,
            mdm_gen_settings=mdm_gen_settings,
            add_noise_to_loss=False,
            verbose=True,
            slice_terrain=False)

        end_time = time.time()
        print("generation time:", end_time - start_time)

        zfill_num = len(str(len(gen_motion_frames)))
        for i in range(len(gen_motion_frames)):
            motion_frames = gen_motion_frames[i].get_copy(cpu_device)
            if len(motion_frames.root_pos.shape) == 3: # [1, num_frames, ...]
                motion_frames = motion_frames.squeeze(0)
            out_terrain = out_terrains[i].torch_copy()
            out_terrain.set_device(cpu_device)
            save_path = os.path.join(output_folder, terrain_name + "_" + str(i).zfill(zfill_num) + ".pkl")
            file_io_helper.save_ms_file(save_path,
                                        motion_frames=motion_frames,
                                        fps=mdm_model._sequence_fps,
                                        loop_mode="CLAMP",
                                        terrain=out_terrain,
                                        path_nodes=path_nodes.cpu())
            print("Saved", save_path)


if __name__ == "__main__":
    main()
