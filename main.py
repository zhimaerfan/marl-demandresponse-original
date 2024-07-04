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
    print("\n------------------------------------未归一化的状态------------------------------------\n", obs_dict)  # efan, 原来print(obs_dict)
    # Select agent
    agents = {"ppo": PPO, "mappo": MAPPO, "dqn": DQN, "tarmac": TARMAC, "maddpg": MADDPG, "tarmac_ppo": TarMAC_PPO}

    # Efan's 初始化各智能体最大自身状态和最大消息状态数，以保证各智能体状态数一致 
    max_self_num_state = -1
    max_message_num_state = -1
    max_agent_message_count = -1  # 最大智能体消息数量
    init_state = True  # 是否返回状态
    # 循环遍历每个智能体的观察值
    for sDict in obs_dict.values():  # 若使用obs_dict.values()来遍历则不包含索引
        _, self_state, temp_messages = normStateDict(sDict, config_dict, init_state = init_state)
        max_self_num_state = max(max_self_num_state, len(self_state))

        # 计算每个智能体的最大消息状态数
        max_temp_message_num_state = max(len(message) for message in temp_messages)
        max_message_num_state = max(max_message_num_state, max_temp_message_num_state)
        max_agent_message_count = max(max_agent_message_count, len(temp_messages)) # 来自几个智能体的消息

    # 更新配置字典
    max_num_state = max_self_num_state + max_message_num_state * max_agent_message_count
    config_dict["default_env_prop"]["cluster_prop"]["max_num_state"] = max_num_state
    config_dict["default_env_prop"]["cluster_prop"]["max_self_num_state"] = max_self_num_state
    config_dict["default_env_prop"]["cluster_prop"]["max_message_num_state"] = max_message_num_state
    config_dict["default_env_prop"]["cluster_prop"]["all_agent_ids"] = env.all_agent_ids
    config_dict["default_env_prop"]["cluster_prop"]["all_nb_agents"] = env.all_nb_agents 

    num_state = max_num_state
    print("Number of states: {}".format(num_state))
    # TODO num_state = env.observation_space.n
    # TODO num_action = env.action_space.n
    # agent = agents[opt.agent_type](config_dict, opt, num_state=num_state, wandb_run = wandb_run) # num_state, num_action
    agent = TarMAC_PPO(config_dict, opt, num_obs_hvac=num_state, num_obs_station=num_state, wandb_run = wandb_run) # num_state, num_action
    
    # Start training
    train = {"ppo": train_ppo, "mappo": train_mappo, "dqn": train_dqn, "tarmac": train_tarmac, "maddpg": train_ddpg, "tarmac_ppo": train_tarmac_ppo}
    # train_tarmac_ppo(env, agent, opt, config_dict, render, log_wandb, wandb_run)
    train[opt.agent_type](env, agent, opt, config_dict, render, log_wandb, wandb_run)
#%%

if __name__ == "__main__":
    main()