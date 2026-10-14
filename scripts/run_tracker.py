import numpy as np
import os
import shutil
import sys
import time
from parc.motion_tracker.envs import env_builder
import parc.motion_tracker.learning.agent_builder as agent_builder
import parc.motion_tracker.util.arg_parser as arg_parser
from parc.util.logger import Logger
import random
import torch
from parc.util import path_loader
from parc.motion_tracker.learning.dm_motion_recorder import record_dm_motions

def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)
    return

def load_args(argv):
    args = arg_parser.ArgParser()
    args.load_args(argv[1:])

    arg_file = args.parse_string("arg_file", "")
    if (arg_file != ""):
        succ = args.load_file(arg_file)
        assert succ, Logger.print("Failed to load args from: " + arg_file)

    return args

def build_env(args, num_envs, device, visualize):

    env_file = args.parse_string("env_config")
    # --debug_visuals True draws the heightfield samples and contacts (needs --visualize True)
    debug_visuals = args.parse_bool("debug_visuals", False)
    env = env_builder.build_env(path_loader.resolve_path(env_file), num_envs, device, visualize,
                                debug_visuals=debug_visuals)
    return env

def build_agent(agent_file, env, device):
    agent = agent_builder.build_agent(agent_file, env, device)
    return agent

def train(agent, max_samples, out_model_file, int_output_dir, logger_type, log_file):
    agent.train_model(max_samples=max_samples, out_model_file=out_model_file, 
                      int_output_dir=int_output_dir, logger_type=logger_type,
                      log_file=log_file)
    return

def test(agent, test_episodes):
    result = agent.test_model(num_episodes=test_episodes)
    Logger.print("Mean Return: {}".format(result["mean_return"]))
    Logger.print("Mean Episode Length: {}".format(result["mean_ep_len"]))
    Logger.print("Episodes: {}".format(result["num_eps"]))

    for key in result:
        if "test_mean" in key:
            Logger.print(key + ": {}".format(result[key]))
    return result

def create_output_dirs(out_model_file, int_output_dir):
    output_dir = os.path.dirname(out_model_file)
    if (output_dir != "" and (not os.path.exists(output_dir))):
        os.makedirs(output_dir, exist_ok=True)

    if (int_output_dir != "" and (not os.path.exists(int_output_dir))):
        os.makedirs(int_output_dir, exist_ok=True)
    return

def copy_file_to_dir(in_path, out_filename, output_dir):
    out_file = os.path.join(output_dir, out_filename)
    shutil.copy(in_path, out_file)
    return

def set_rand_seed(args):
    rand_seed_key = "rand_seed"

    if (args.has_key(rand_seed_key)):
        rand_seed = args.parse_int(rand_seed_key)
    else:
        rand_seed = np.uint64(time.time() * 256)
        
    print("Setting seed: {}".format(rand_seed))
    random.seed(rand_seed)
    np.random.seed(np.uint64(rand_seed % (2**32)))
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)
    return

def run(args):
    mode = args.parse_string("mode", "train")
    num_envs = args.parse_int("num_envs", 1)
    device = args.parse_string("device", 'cuda:0')
    visualize = args.parse_bool("visualize", True)
    logger_type = args.parse_string("logger", "tb")
    log_file = args.parse_string("log_file", "output/log.txt")
    out_model_file = args.parse_string("out_model_file", "output/model.pt")
    int_output_dir = args.parse_string("int_output_dir", "")
    model_file = args.parse_string("model_file", "")

    set_rand_seed(args)
    set_np_formatting()

    create_output_dirs(out_model_file, int_output_dir)

    env = build_env(args, num_envs, device, visualize)
    agent_file = args.parse_string("agent_config")
    agent = build_agent(path_loader.resolve_path(agent_file), env, device)

    if (model_file != ""):
        agent.load(path_loader.resolve_path(model_file))

    test_predictor = args.parse_bool("test_predictor", False)
    if (test_predictor):
        # The actor uses the future-pose predictor's latent instead of the encoder's.
        if (mode == "train"):
            raise ValueError("--test_predictor is only supported in test and record modes.")
        if (not hasattr(agent, "set_predictor_test_mode")):
            raise ValueError("--test_predictor needs an agent config with obs_encoders and a future-pose predictor.")
        Logger.print("Predictor test mode: the actor uses the predicted future-pose latent.")
        agent.set_predictor_test_mode(True)

    if (mode == "train"):
        max_samples = args.parse_int("max_samples", np.iinfo(np.int64).max)
        train(agent=agent, max_samples=max_samples, out_model_file=out_model_file,
              int_output_dir=int_output_dir, logger_type=logger_type, log_file=log_file)
    elif (mode == "test"):
        test_episodes = args.parse_int("test_episodes", np.iinfo(np.int64).max)
        test(agent=agent, test_episodes=test_episodes)
    elif (mode == "record"):
        record_dm_motions(agent)
    else:
        assert(False), "Unsupported mode: {}".format(mode)

    return

def main(argv):
    args = load_args(argv)
    run(args)
    return

if __name__ == "__main__":
    main(sys.argv)