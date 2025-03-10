#%% Imports

from config import config_dict
from cli import cli_train
from agents.ddpg import MADDPG
from env.MA_DemandResponse import MADemandResponseEnv
from metrics import Metrics

from plotting import colorPlotTestAgentHouseTemp
from utils import (
    normStateDict,
    # testAgentHouseTemperature,
    saveDDPGDict,
    adjust_config_train,
    render_and_wandb_init,
    test_ppo_agent,
)
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from collections import namedtuple
import wandb
# efan
import datetime

#%% Functions


def train_ddpg(env, agent, opt, config_dict, render, log_wandb, wandb_run):
    # id_rng = np.random.default_rng()
    # unique_ID = str(int(id_rng.random() * 1000000))
    # maddpg = MADDPG(
    #     dim_info,
    #     config_dict,
    #     opt
    # )
    maddpg = agent
    id_rng = np.random.default_rng()
    unique_ID = str(int(id_rng.random() * 1000000))
    # efan
    current_time = datetime.datetime.now().strftime("-%Y%m%d-%H:%M:%S-")

    nb_time_steps = config_dict["training_prop"]["nb_time_steps"]
    nb_tr_episodes = config_dict["DDPG_prop"]["episode_num"]
    nb_tr_epochs = config_dict["training_prop"]["nb_tr_epochs"]
    nb_tr_logs = config_dict["training_prop"]["nb_tr_logs"]
    nb_test_logs = config_dict["training_prop"]["nb_test_logs"]
    nb_inter_saving_actor = config_dict["training_prop"]["nb_inter_saving_actor"]

    # Initialize render, if applicable
    if render:
        from env.renderer import Renderer

        renderer = Renderer(env.nb_agents)

    # Initialize variables
    # Transition = namedtuple("Transition", ["state", "action", "a_log_prob", "reward", "next_state", "done"])
    time_steps_per_episode = int(nb_time_steps / nb_tr_episodes)
    time_steps_per_epoch = int(nb_time_steps / nb_tr_epochs)
    time_steps_train_log = int(nb_time_steps / nb_tr_logs)
    time_steps_test_log = int(nb_time_steps / nb_test_logs)
    time_steps_per_saving_actor = int(
        nb_time_steps / (nb_inter_saving_actor + 1)
    )
    metrics = Metrics()
    step = 0  # global step counter
    # agent_num = env.num_agents
    # reward of each episode of each agent
    # 字典推导式是Python中一种简洁且高效的创建字典的方法:{key_expression: value_expression for item in iterable}
    # 这里的key_expression是字典中键的表达式，value_expression是字典中值的表达式，iterable是一个可迭代对象，item是从iterable中取出的每个元素。
    episode_rewards = {
        agent_id: np.zeros(config_dict["DDPG_prop"]["episode_num"])
        for agent_id in range(opt.nb_agents)
    }

    for episode in range(config_dict["DDPG_prop"]["episode_num"]):
        print(f"New episode at time {step}")
        obs = env.reset()
        if render:
            renderer.render(obs)
        # iter(obs): 创建一个迭代器，用于遍历字典obs。
        # next(iter(obs)): 从迭代器中获取下一个元素，也就是字典obs的第一个键。
        # obs[next(iter(obs))]: 使用获取到的键来从字典obs中获取对应的值。
        # normStateDict(obs[next(iter(obs))], config_dict): 调用函数normStateDict，传入上一步获取到的值和字典config_dict作为参数。
        obs_ = normStateDict(obs[next(iter(obs))], config_dict)
        # 使用字典推导式为每个智能体创建了一个观察字典obs_dict。字典的键是智能体的ID（从0到opt.nb_agents - 1），值都是obs_。这意味着在这个时刻，所有智能体共享相同的观察值obs_。这种设计可能是基于环境的特性，其中所有智能体在某些情况下可能会接收相同的观察信息。如果每个智能体应该有不同的观察值，那么这段代码可能需要进行修改。
        obs_dict = {
            agent_id: obs_  # env.action_space(agent_id).sample()
            for agent_id in range(opt.nb_agents)
        }
        agent_reward = {
            agent_id: 0 for agent_id in range(opt.nb_agents)
        }  # agent reward of the current episode
        for s in range(time_steps_per_episode):  # interact with the env for an episode
            step += 1
            if step < config_dict["DDPG_prop"]["random_steps"]:
                action = {
                    agent_id: np.random.randint(
                        0, 2
                    )  # env.action_space(agent_id).sample()
                    for agent_id in range(opt.nb_agents)
                }
            else:
                action = maddpg.select_action(obs_dict)

            next_obs, reward, done, info = env.step(action)
            if render and step >= opt.render_after:
                renderer.render(next_obs)
            # env.render()
            next_obs_ = normStateDict(next_obs[next(iter(next_obs))], config_dict)
            next_obs_dict = {
                agent_id: next_obs_  # env.action_space(agent_id).sample()
                for agent_id in range(opt.nb_agents)
            }
            maddpg.push(obs_dict, action, reward, next_obs_dict, done)

            for k in obs_dict.keys():
                metrics.update(k, obs, next_obs, reward, env)

            for agent_id, r in reward.items():  # update reward
                agent_reward[agent_id] += r

            if (
                step >= config_dict["DDPG_prop"]["random_steps"]
                and step % time_steps_per_epoch == 0
            ):  # learn every few steps
                # maddpg.update(opt.batch_size, opt.gamma)
                print(f"Updating agent at time {step}")
                maddpg.update()
                maddpg.update_target()

            obs = next_obs

        # episode finishes
        for agent_id, r in agent_reward.items():  # record reward
            episode_rewards[agent_id][episode] = r

        if (episode + 1) % 100 == 0:  # print info every 100 episodes
            message = f"episode {episode + 1}, "
            sum_reward = 0
            for agent_id, r in agent_reward.items():  # record reward
                message += f"{agent_id}: {r:>4f}; "
                sum_reward += r
            message += f"sum reward: {sum_reward}"
            print(message)
            if log_wandb:
                wandb_run.log({"sum_reward": sum_reward})
        # Log train statistics
        if (
            step % time_steps_train_log == time_steps_train_log - 1
        ):  # Log train statistics
            # print("Logging stats at time {}".format(t))
            logged_metrics = metrics.log(step, time_steps_train_log)
            if log_wandb:
                wandb_run.log(logged_metrics)
            metrics.reset()

        # Test policy
        if step % time_steps_test_log == time_steps_test_log - 1:  # Test policy
            print(f"Testing at time {step}")
            metrics_test = test_ppo_agent(agent, env, config_dict, opt, step)
            if log_wandb:
                wandb_run.log(metrics_test)
            else:
                print("Training step - {}".format(step))

        if (
            opt.save_actor_name
            and step % time_steps_per_saving_actor == 0
            and step != 0
        ):
            # efan
            # path = os.path.join(".", "actors", opt.save_actor_name + current_time + unique_ID)
            path = os.path.join(".", "actors", opt.save_actor_name + unique_ID)
            saveDDPGDict(agent, path, step)
            if log_wandb:
                wandb.save(os.path.join(path, "actor" + str(step) + ".pth"))
    if render:
        renderer.__del__(obs)
    maddpg.save(episode_rewards)  # save model
    return episode_rewards


def get_running_reward(arr: np.ndarray, window=100):
    """calculate the running reward, i.e. average of last `window` elements from rewards"""
    running_reward = np.zeros_like(arr)
    for i in range(window - 1):
        running_reward[i] = np.mean(arr[: i + 1])
    for i in range(window - 1, len(arr)):
        running_reward[i] = np.mean(arr[i - window + 1 : i + 1])
    return running_reward


#%% Train

if __name__ == "__main__":
    # import os

    os.environ["WANDB_SILENT"] = "true"
    opt = cli_train()
    adjust_config_train(opt, config_dict)
    # render, log_wandb, wandb_run = render_and_wandb_init(opt, config_dict)
    random.seed(opt.env_seed)
    # env = MADemandResponseEnv(config_dict)
    # agent = PPO(config_dict, opt)
    # train_ppo(env, agent, opt, config_dict, render, log_wandb, wandb_run)
    # from easydict import EasyDict

    # opt = EasyDict(vars(opt))
    # opt.env_name = "MA_DemandResponse"
    # opt.episode_num = 10000 #
    # opt.episode_length = 25 #
    # opt.random_steps = 100 #
    # opt.soft_tau = 0.02 #
    # opt.gamma = 0.95 #
    # opt.buffer_capacity = int(1e6) #
    # opt.batch_size = 64 #
    # opt.actor_lr = 1e-2 #
    # opt.critic_lr = 1e-2 #
    # opt.learn_interval = 100 #
    env_dir = os.path.join("./ddpg_results")
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
    total_files = len([file for file in os.listdir(env_dir)])
    result_dir = os.path.join(env_dir, f"{total_files + 1}")
    os.makedirs(result_dir)
    opt.result_dir = result_dir

    env = MADemandResponseEnv(config_dict)
    obs_dict = env.reset()
    num_state = len(normStateDict(obs_dict[next(iter(obs_dict))], config_dict))
    # dim_info = get_dim_info(opt, env, num_state)
    episode_rewards = train_ddpg(
        env,
        config_dict,
        opt,
    )

    # training finishes, plot reward
    fig, ax = plt.subplots()
    x = range(1, config_dict["DDPG_prop"]["episode_num"] + 1)
    for agent_id, reward in episode_rewards.items():
        ax.plot(x, reward, label=agent_id)
        ax.plot(x, get_running_reward(reward))
    ax.legend()
    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
    title = f"training result of maddpg solve {opt.env_name}"
    ax.set_title(title)
    plt.savefig(os.path.join(result_dir, title))
