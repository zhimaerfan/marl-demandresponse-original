# from apt import ProblemResolver
from cmath import nan
from env import *
from agents import *
from config import config_dict
from utils import get_actions, adjust_config_deploy, normStateDict
from wandb_setup import wandb_setup
from copy import deepcopy
import warnings
import os
import random
import time
import numpy as np
import pandas as pd
import argparse
import wandb
from cli import cli_deploy


os.environ["WANDB_SILENT"] = "true"

agents_dict = {
    "BangBang": BangBangController,
    "DeadbandBangBang": DeadbandBangBangController,
    "Basic": BasicController,
    "AlwaysOn": AlwaysOnController,
    "PPO": PPOAgent,
    "MAPPO": PPOAgent,
    "DQN": DQNAgent,
    "GreedyMyopic": GreedyMyopic,
    "MPC": MPCController,
    "MADDPG": DDPGAgent,
    "TarmacPPO": TarmacPPOAgent,
}


# CLI arguments

opt = cli_deploy(agents_dict)
adjust_config_deploy(opt, config_dict)

log_wandb = not opt.no_wandb
if opt.render:
    from env.renderer import Renderer

    renderer = Renderer(opt.hvac_nb_agents)
if opt.log_metrics_path != "":
    df_metrics = pd.DataFrame()

# Creating environment
random.seed(opt.env_seed)
nb_time_steps = opt.nb_time_steps

if log_wandb:
    wandb_run = wandb_setup(opt, config_dict)

env = MADemandResponseEnv(config_dict)
obs_dict = env.reset()

# Efan's 需要修改吗?状态数都不一样了
num_state = len(normStateDict(obs_dict[next(iter(obs_dict))], config_dict))

if opt.log_metrics_path != "":
    df_metrics = pd.DataFrame()
time_steps_log = int(opt.nb_time_steps / opt.nb_logs)
hvac_nb_agents = config_dict["default_env_prop"]["cluster_prop"]["hvac_nb_agents"]
houses = env.cluster.houses

if opt.agent != "TarmacPPO":
    actors = {}
    for house_id in houses.keys():
        agent_prop = {"id": house_id}

        if opt.actor_name:
            agent_prop["actor_name"] = opt.actor_name
            agent_prop["net_seed"] = opt.net_seed

        actors[house_id] = agents_dict[opt.agent](agent_prop, config_dict, num_state=num_state)

else:
    agent_prop = {"net_seed" : opt.net_seed, "actor_name" : opt.actor_name}
    actors = TarmacPPOAgent(agent_prop, config_dict, num_state=num_state)

obs_dict = env.reset()


cumul_temp_offset = 0
cumul_temp_error = 0
max_temp_error = 0
cumul_signal_offset = 0
cumul_signal_error = 0
cumul_OD_temp = 0
cumul_signal = 0
cumul_cons = 0

cumul_squared_error_sig = 0
cumul_squared_error_temp = 0
cumul_squared_max_error_temp = 0

actions = get_actions(actors, obs_dict)
t1_start = time.process_time() 

for i in range(nb_time_steps):
    obs_dict, _, _, info = env.step(actions)
    actions = get_actions(actors, obs_dict)
    if opt.log_metrics_path != "" and i >= opt.start_stats_from:
        df = pd.DataFrame(obs_dict).transpose()
 
        df["temperature_difference"] = df["house_temp"] - df["house_target_temp"]
        df["temperature_error"] = np.abs(df["house_temp"] - df["house_target_temp"])
        # 温度偏差（temp_diff）：计算每个房屋的当前温度与其目标温度之间的差异，并取所有房屋的平均值。这反映了智能体在保持温度目标上的总体表现。
        temp_diff = df["temperature_difference"].mean() 
        # 温度误差（temp_err）：计算每个房屋的当前温度与其目标温度之间差异的绝对值，并取所有房屋的平均值。这是一个关于智能体性能的误差指标，越低越好。
        temp_err = df["temperature_error"].mean()
        # 室内温度（air_temp）和质量温度（mass_temp）：分别表示集群内所有房屋的平均室内温度和室内质量的平均温度。
        air_temp = df["house_temp"].mean()
        mass_temp = df["house_mass_temp"].mean()
        # 目标温度（target_temp）：集群内所有房屋设定的平均目标温度。
        target_temp = df["house_target_temp"].mean()
        # 室外温度（OD_temp）：集群内所有房屋的共同室外温度。
        OD_temp = df["OD_temp"][0]
        # 总调节信号（signal）：控制HVAC功率输出的信号，反映了调节系统发出的指令。
        signal = df["grid_active_reg_signal"][0]
        # 功率消耗（consumption）：集群HVAC系统的总功率消耗
        consumption = df["cluster_hvac_active_power"][0]
        row = pd.DataFrame({"temp_diff":temp_diff, "temp_err":temp_err, "air_temp":air_temp, "mass_temp":mass_temp,"target_temp":target_temp, "OD_temp":OD_temp, "signal": signal, "consumption":consumption}, index=[config_dict["default_env_prop"]["time_step"]*i])
        df_metrics = pd.concat([df_metrics,row])
        

    if opt.render and i >= opt.render_after:
        renderer.render(obs_dict)
    # Max temperature error(max_temp_error_houses)：在当前记录间隔内遇到的最大温度误差。
    max_temp_error_houses = 0
    for k in obs_dict.keys():
        temp_error = obs_dict[k]["house_temp"] - obs_dict[k]["house_target_temp"]
        cumul_temp_offset += temp_error / env.hvac_nb_agents
        cumul_temp_error += np.abs(temp_error) / env.hvac_nb_agents
        if np.abs(temp_error) > max_temp_error:
            max_temp_error = np.abs(temp_error)
        if np.abs(temp_error) > max_temp_error_houses:
            max_temp_error_houses = np.abs(temp_error)

        if i >= opt.start_stats_from:
            cumul_squared_error_temp += temp_error**2
            
    if i>= opt.start_stats_from:
        cumul_squared_max_error_temp += max_temp_error_houses**2
    cumul_OD_temp += obs_dict[0]["OD_temp"]
    cumul_signal += obs_dict[0]["grid_active_reg_signal"]
    cumul_cons += obs_dict[0]["cluster_hvac_active_power"]
    
    # 信号误差（signal_error）：总调节信号与实际集群HVAC功率之间的差异。
    signal_error = obs_dict[0]["grid_active_reg_signal"] - obs_dict[0]["cluster_hvac_active_power"]
    cumul_signal_offset += signal_error
    cumul_signal_error += np.abs(signal_error)

    if i >= opt.start_stats_from:
        cumul_squared_error_sig += signal_error**2

    if i % time_steps_log == time_steps_log - 1:  # Log train statistics
        # print("Logging stats at time {}".format(t))

        #print("Average absolute noise: {} W".format(env.power_grid.cumulated_abs_noise / env.power_grid.nb_steps ))


        # Mean temperature offset(mean_temp_offset)在日志记录间隔内，房屋温度与目标温度差异的平均值。
        mean_temp_offset = cumul_temp_offset / time_steps_log
        # Mean temperature error(mean_temp_error)在日志记录间隔内，房屋温度与目标温度差异的绝对值的平均值。
        mean_temp_error = cumul_temp_error / time_steps_log
        # Mean signal offset(mean_signal_offset)在日志记录间隔内，调节信号与实际集群HVAC功率之间差异的平均值。
        mean_signal_offset = cumul_signal_offset / time_steps_log
        # Mean signal error(mean_signal_error)在日志记录间隔内，调节信号与实际集群HVAC功率之间差异的绝对值的平均值。
        mean_signal_error = cumul_signal_error / time_steps_log
        # Mean outside temperature(mean_OD_temp)在日志记录间隔内，室外温度的平均值。
        mean_OD_temp = cumul_OD_temp / time_steps_log
        # Mean signal(mean_signal)在日志记录间隔内，调节信号的平均值。
        mean_signal = cumul_signal / time_steps_log
        # Mean consumption(mean_consumption)在日志记录间隔内，集群HVAC功率消耗的平均值。
        mean_consumption = cumul_cons / time_steps_log

        # 循环中的逐步累积计算, 计算到目前为止的累积误差的均方根误差（RMSE）等
        if i >= opt.start_stats_from:
            # RMSE signal per agent（rmse_sig_per_ag）：计算所有代理的信号误差的均方根误差，并除以代理数量来获取每个代理的平均值。这是评估信号跟踪性能的关键指标。
            rmse_sig_per_ag = np.sqrt(cumul_squared_error_sig/(i-opt.start_stats_from))/env.hvac_nb_agents
            # RMSE temperature（rmse_temp）：计算所有代理的温度误差的均方根误差，再除以代理数量来获取平均值。这反映了智能体在维持目标温度方面的准确性。
            rmse_temp = np.sqrt(cumul_squared_error_temp/((i-opt.start_stats_from)*env.hvac_nb_agents))
            # RMS Max Error temperature(rms_max_error_temp)：在当前记录间隔内，房屋遇到的最大温度误差的均方根，这是对极端情况性能的量度。
            rms_max_error_temp = np.sqrt(cumul_squared_max_error_temp/(i-opt.start_stats_from))
        else:
            rmse_sig_per_ag = nan
            rmse_temp = nan 
            rms_max_error_temp = nan

        if log_wandb:
            wandb_run.log(
                {
                    "RMSE signal per agent": rmse_sig_per_ag,
                    "RMSE temperature": rmse_temp,
                    "RMS Max Error temperature": rms_max_error_temp,
                    "Mean temperature offset": mean_temp_offset,
                    "Mean temperature error": mean_temp_error,
                    "Max temperature error": max_temp_error,
                    "Mean signal offset": mean_signal_offset,
                    "Mean signal error": mean_signal_error,
                    "Mean outside temperature": mean_OD_temp,
                    "Mean signal" : mean_signal,
                    "Mean consumption": mean_consumption,
                    # 当前时间步中的小时数，这可能用于分析一天中不同时间的性能。
                    "Time (hour)": obs_dict[0]["datetime"].hour,
                    # 当前的时间步数或迭代数。
                    "Time step": i,
                }
            )

        cumul_temp_offset = 0
        cumul_temp_error = 0
        max_temp_error = 0
        cumul_signal_offset = 0
        cumul_signal_error = 0
        cumul_OD_temp = 0
        cumul_signal = 0
        cumul_cons = 0
        print("Time step: {}".format(i))
        t1_stop = time.process_time()
        print("Elapsed time for {}% of steps: {} seconds.".format(int(np.round(float(i)/nb_time_steps*100)), int(t1_stop - t1_start))) 
# 循环结束时的最终计算：在模拟结束后，代码再次计算RMSE，这次使用的是从 opt.start_stats_from 到最后一个时间步的所有数据。这为整个模拟提供了最终的性能评估。
rmse_sig_per_ag = np.sqrt(cumul_squared_error_sig/(nb_time_steps-opt.start_stats_from))/env.hvac_nb_agents
rmse_temp = np.sqrt(cumul_squared_error_temp/((nb_time_steps-opt.start_stats_from)*env.hvac_nb_agents))
rms_max_error_temp = np.sqrt(cumul_squared_max_error_temp/(nb_time_steps-opt.start_stats_from))
print("RMSE Signal per agent: {} W".format(int(rmse_sig_per_ag)))
print("RMSE Temperature: {} C".format(rmse_temp))
print("RMS Max Error Temperature: {} C".format(rms_max_error_temp))


#print("Average absolute noise: {} W".format(env.power_grid.cumulated_abs_noise / env.power_grid.nb_steps ))
if log_wandb:
    wandb_run.log({
        "RMSE signal per agent": rmse_sig_per_ag,
        "RMSE temperature": rmse_temp,
        "RMS Max Error temperature": rms_max_error_temp,
        }
    )
if opt.log_metrics_path != "":
    df_metrics.to_csv(opt.log_metrics_path)