import yaml

import parc.motion_tracker.learning.dm_ppo_agent as dm_ppo_agent
import parc.motion_tracker.learning.ppo_agent as ppo_agent
import parc.util.path_loader as path_loader
from parc.util.logger import Logger


def build_agent(agent_file, env, device):
    agent_config = path_loader.load_config(path_loader.resolve_path(agent_file))
    
    agent_name = agent_config["agent_name"]
    Logger.print("Building {} agent".format(agent_name))

    if (agent_name == ppo_agent.PPOAgent.NAME):
        agent = ppo_agent.PPOAgent(config=agent_config, env=env, device=device)
    elif (agent_name == dm_ppo_agent.DMPPOAgent.NAME):
        agent = dm_ppo_agent.DMPPOAgent(config=agent_config, env=env, device=device)
    else:
        assert(False), "Unsupported agent: {}".format(agent_name)

    num_params = agent.calc_num_params()
    Logger.print("Total parameter count: {}".format(num_params))

    return agent