#%% Imports

from config import config_dict
from cli import cli_train
from agents.tarmac_ppo import TarMAC_PPO
from env.MA_DemandResponse import MADemandResponseEnv
from metrics import Metrics
from utils import (
    adjust_config_train,
    normStateDict,
    saveActorNetDict,
    render_and_wandb_init,
    test_tarmac_ppo_agent,
)

import os
import random
import numpy as np
from collections import namedtuple
import wandb
import torch
# efan
import datetime

#%% Functions


def train_tarmac_ppo(env, agent, opt, config_dict, render, log_wandb, wandb_run):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 生成一个随机且唯一的标识符, 用于唯一标识某个特定的实验、模型保存文件或其他需要唯一标识的场景。
    id_rng = np.random.default_rng()
    unique_ID = str(int(id_rng.random() * 1000000))
    # Efan's
    current_time = datetime.datetime.now().strftime("-%Y%m%d-%H:%M:%S-")  

    # Training configuration
    nb_time_steps = config_dict["training_prop"]["nb_time_steps"]
    time_steps_per_episode = int(nb_time_steps / config_dict["training_prop"]["nb_tr_episodes"])
    time_steps_per_epoch = int(nb_time_steps / config_dict["training_prop"]["nb_tr_epochs"])
    time_steps_train_log = int(nb_time_steps / config_dict["training_prop"]["nb_tr_logs"])
    time_steps_test_log = int(nb_time_steps / config_dict["training_prop"]["nb_test_logs"])
    time_steps_per_saving_actor = int(nb_time_steps / (config_dict["training_prop"]["nb_inter_saving_actor"] + 1))

    # Initialize render, if applicable
    if render:
        from env.renderer import Renderer
        renderer = Renderer(env.hvac_nb_agents)

    # Initialize variables
    Transition = namedtuple(
        # "Transition", ["state", "action", "a_log_prob", "reward", "next_state", "done"]
        "Transition", ["state", "discrete_action", "continuous_action", "discrete_a_log_prob", "continuous_action_log_prob", "continuous_mean", "continuous_std", "reward", "next_state", "done"]
    )

    metrics = Metrics() #wangli 

    # Get first observation
    obs_dict = env.reset()

    for t in range(nb_time_steps):

        # Render observation
        if render:
            renderer.render(obs_dict)


        #### Passing actor one shot
        # Select action with probabilities
        obs_all = np.array([normStateDict(obs_dict[k], config_dict) for k in obs_dict.keys()]) 
        # Efan's 根据异构智能体的id来选择动作
        discrete_actions, discrete_action_probs, continuous_actions, continuous_action_log_probs, continuous_means, continuous_stds = agent.select_actions(obs_all, all_agent_ids=list(obs_dict.keys()))
        # 原来的
        # action = {k: actions_and_probs[0][k] for k in obs_dict.keys()}
        # action_prob = {k: actions_and_probs[1][k] for k in obs_dict.keys()}
        # Efan 提取每个智能体的动作和概率
        action = {}
        discrete_action = {}
        discrete_action_prob = {}
        continuous_action = {}
        continuous_action_log_prob = {}
        continuous_mean = {}
        continuous_std = {}
        discrete_action_index = 0
        continuous_action_index = 0  # 索引用于访问连续动作和对应概率
        for agent_id in obs_dict.keys():
            # 对于HVAC智能体，使用离散动作
            if isinstance(agent_id, int):  # Efan's 有问题action_prob没被使用?
                action[agent_id] = discrete_actions[discrete_action_index].flatten()
                discrete_action[agent_id] = discrete_actions[discrete_action_index].flatten()  # 提取动作并转换为一维数组或单个值
                discrete_action_prob[agent_id] = discrete_action_probs[discrete_action_index].flatten()    # 提取概率并转换为一维数组或单个值
                discrete_action_index += 1
            elif 'charging_station' in agent_id:
                action[agent_id] = continuous_actions[continuous_action_index].flatten()
                continuous_action[agent_id] = continuous_actions[continuous_action_index].flatten()  # 提取动作并转换为一维数组
                continuous_action_log_prob[agent_id] = continuous_action_log_probs[continuous_action_index].flatten()
                # 这里简化了处理，连续动作的“概率”以均值和标准差的形式给出
                continuous_mean[agent_id] = continuous_means[continuous_action_index].flatten()  # 提取连续动作的均值
                continuous_std[agent_id] = continuous_stds[continuous_action_index].flatten()
                continuous_action_index += 1

        # Take action and get new transition
        next_obs_dict, rewards_dict, dones_dict, info_dict = env.step(action)

        # Render next observation
        if render and t >= opt.render_after:
            renderer.render(next_obs_dict)

        # Check if the episode is done
        done = t % time_steps_per_episode == time_steps_per_episode - 1

        for k in obs_dict.keys():
            if isinstance(k, int):
                # HVAC智能体处理离散动作
                agent.store_transition(
                    Transition(
                        state=normStateDict(obs_dict[k], config_dict),  # Efan's 这里有错误, 存的应该是discrete_action而不是discrete_actions
                        discrete_action=discrete_action[k],  # 使用针对HVAC的离散动作
                        continuous_action=None,  # HVAC没有连续动作
                        discrete_a_log_prob=discrete_action_prob[k],  # 使用离散动作概率
                        continuous_action_log_prob=None,
                        continuous_mean=None,  # HVAC没有连续动作的均值和标准差
                        continuous_std=None,  # HVAC没有连续动作的均值和标准差
                        reward=rewards_dict[k],
                        next_state=normStateDict(next_obs_dict[k], config_dict),
                        done=done
                    ),
                    k
                )
            elif 'charging_station' in k:
                # EV station智能体处理连续动作
                agent.store_transition(
                    Transition(
                        state=normStateDict(obs_dict[k], config_dict),
                        discrete_action=None,  # EV station没有离散动作
                        continuous_action=continuous_action[k],  # 使用针对EV station的连续动作
                        discrete_a_log_prob=None,  # EV station没有离散动作概率
                        continuous_action_log_prob=continuous_action_log_prob[k],
                        continuous_mean=continuous_mean[k],  # 使用连续动作的均值和标准差
                        continuous_std=continuous_std[k],  
                        reward=rewards_dict[k],
                        next_state=normStateDict(next_obs_dict[k], config_dict),
                        done=done
                    ),
                    k
                )
            # Update metrics
            metrics.update(k, next_obs_dict, rewards_dict, env)

        # Set next state as current state
        obs_dict = next_obs_dict
        
        # New episode, reset environment
        if done:
            print(f"New episode at time {t}")
            obs_dict = env.reset()

        # Epoch: update agent
        if (
            t % time_steps_per_epoch == time_steps_per_epoch - 1
            and len(next(iter(agent.buffer.values()))) >= agent.batch_size
        ):
            print(f"Updating agent at time {t}")
            agent.update(t)

        # Log train statistics
        if t % time_steps_train_log == time_steps_train_log - 1:  # Log train statistics
            logged_metrics = metrics.log(t, time_steps_train_log)
            if log_wandb:
                wandb_run.log(logged_metrics)
            metrics.reset()

        # Test policy
        if t % time_steps_test_log == time_steps_test_log - 1:  # Test policy
            print(f"Testing at time {t}")
            metrics_test = test_tarmac_ppo_agent(agent, env, config_dict, opt, t)
            if log_wandb:
                wandb_run.log(metrics_test)
            else:
                print("Training step - {}".format(t))

        if opt.save_actor_name and t % time_steps_per_saving_actor == 0 and t != 0:
            # efan
            path = os.path.join(".", "actors", opt.save_actor_name + current_time + unique_ID)
            # path = os.path.join(".", "actors", opt.save_actor_name + unique_ID)
            saveActorNetDict(agent, path, t)
            if log_wandb:
                wandb.save(os.path.join(path, "actor" + str(t) + ".pth"))

    if render:
        renderer.__del__(obs_dict)

    if opt.save_actor_name:
        # efan
        path = os.path.join(".", "actors", opt.save_actor_name + current_time + unique_ID)
        # path = os.path.join(".", "actors", opt.save_actor_name + unique_ID)
        saveActorNetDict(agent, path)
        if log_wandb:
            wandb.save(os.path.join(path, "actor.pth"))


#%% Train

if __name__ == "__main__":
    import os

    os.environ["WANDB_SILENT"] = "true"
    opt = cli_train()
    adjust_config_train(opt, config_dict)
    render, log_wandb, wandb_run = render_and_wandb_init(opt, config_dict)
    random.seed(opt.env_seed)
    env = MADemandResponseEnv(config_dict)
    agent = TarMAC_PPO(config_dict, opt)
    train_tarmac_ppo(env, agent, opt, config_dict, render, log_wandb, wandb_run)
