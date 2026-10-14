import yaml

import parc.motion_tracker.learning.dm_ppo_agent as dm_ppo_agent
import parc.motion_tracker.learning.dm_ppo_fp_agent as dm_ppo_fp_agent
import parc.motion_tracker.learning.ppo_agent as ppo_agent
import parc.util.path_loader as path_loader
from parc.util.logger import Logger


def build_agent(agent_file, env, device):
    agent_config = path_loader.load_config(path_loader.resolve_path(agent_file))
    
    agent_name = agent_config["agent_name"]
    agent_class = get_agent_class(agent_config)
    Logger.print("Building {} agent ({})".format(agent_name, agent_class.__name__))

    agent = agent_class(config=agent_config, env=env, device=device)

    num_params = agent.calc_num_params()
    Logger.print("Total parameter count: {}".format(num_params))

    return agent

def get_agent_class(agent_config):
    agent_name = agent_config["agent_name"]

    if (agent_name == ppo_agent.PPOAgent.NAME):
        agent_class = ppo_agent.PPOAgent
    elif (agent_name == dm_ppo_fp_agent.FuturePoseDMPPOAgent.NAME):
        agent_class = dm_ppo_fp_agent.FuturePoseDMPPOAgent
    elif (agent_name == dm_ppo_agent.DMPPOAgent.NAME):
        # As on develop, a DM_PPO config with observation encoders (e.g. dm_agent_g1.yaml)
        # gets the future-pose agent.
        model_config = agent_config.get("model", None) or dict()
        if (model_config.get("obs_encoders", None)):
            agent_class = dm_ppo_fp_agent.FuturePoseDMPPOAgent
        else:
            agent_class = dm_ppo_agent.DMPPOAgent
    else:
        raise ValueError("Unsupported agent: {}".format(agent_name))

    return agent_class
