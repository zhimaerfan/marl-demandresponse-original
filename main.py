#%% Imports

from agents.dqn import DQN
from agents.ppo import PPO
from agents.tarmac_ppo import TarMAC_PPO
from agents.mappo import MAPPO
from agents.ddpg import MADDPG
from agents.tarmac.a2c_acktr import A2C_ACKTR as TARMAC
from train_dqn import train_dqn
from train_ppo import train_ppo
from train_mappo import train_mappo
from train_ddpg import train_ddpg
from train_tarmac import train_tarmac
from train_tarmacPPO import train_tarmac_ppo
from config import config_dict
from cli import cli_train
from env.MA_DemandResponse import MADemandResponseEnv
from utils import adjust_config_train, render_and_wandb_init, normStateDict

import os
import random

os.environ["WANDB_SILENT"] = "true"

def main():
    # 先载入cli.py的parser参数
    opt = cli_train()
    # 根据cli.py的parser参数,更改 config_dict 的配置。
    adjust_config_train(opt, config_dict)
    render, log_wandb, wandb_run = render_and_wandb_init(opt, config_dict)

    # Create environment
    random.seed(opt.env_seed)
    env = MADemandResponseEnv(config_dict)
    obs_dict = env.reset()
    print(obs_dict[0])  # efan, 原来print(obs_dict)
    # Select agent
    agents = {"ppo": PPO, "mappo": MAPPO, "dqn": DQN, "tarmac": TARMAC, "maddpg": MADDPG, "tarmac_ppo": TarMAC_PPO}

    num_state = len(normStateDict(obs_dict[next(iter(obs_dict))], config_dict))
    print("Number of states: {}".format(num_state))
    # TODO num_state = env.observation_space.n
    # TODO num_action = env.action_space.n
    agent = agents[opt.agent_type](config_dict, opt, num_state=num_state, wandb_run = wandb_run) # num_state, num_action
    
    # Start training
    train = {"ppo": train_ppo, "mappo": train_mappo, "dqn": train_dqn, "tarmac": train_tarmac, "maddpg": train_ddpg, "tarmac_ppo": train_tarmac_ppo}
    train[opt.agent_type](env, agent, opt, config_dict, render, log_wandb, wandb_run)

#%%

if __name__ == "__main__":
    main()