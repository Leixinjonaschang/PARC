import yaml

import envs.ig_parkour.ig_parkour_env as ig_parkour_env
from util.logger import Logger

def build_env(env_file, num_envs, device, visualize, debug_visuals=False):
    env_config = load_env_file(env_file)

    # Override debug_visuals if provided (config has nested "env" key)
    if debug_visuals:
        if "env" in env_config:
            env_config["env"]["debug_visuals"] = debug_visuals
        else:
            env_config["debug_visuals"] = debug_visuals

    env_name = env_config["env_name"]
    Logger.print("Building {} env".format(env_name))

    if (env_name == ig_parkour_env.IGParkourEnv.NAME):
        env = ig_parkour_env.IGParkourEnv(config=env_config, 
                                          num_envs=num_envs, 
                                          device=device,
                                          visualize=visualize)
    else:
        assert(False), "Unsupported env: {}".format(env_name)

    return env

def load_env_file(file):
    with open(file, "r") as stream:
        env_config = yaml.safe_load(stream)
    return env_config
