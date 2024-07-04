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
import datetime


os.environ["WANDB_SILENT"] = "true"

agents_dict = {
    "BangBang": BangBangController,
    "DeadbandBangBang": DeadbandBangBangController,
    "Basic": BasicController,
    "AlwaysOn": AlwaysOnController,
    "EvBangBang": EvBangBangController,
    "EvDeadbandBangBang": EvDeadbandBangBangController,
    "EvBasic": EvBasicController,
    "EvAlwaysOn": EvAlwaysOnController,
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



# Efan's 需要修改吗?状态数都不一样了
# num_state = len(normStateDict(obs_dict[next(iter(obs_dict))], config_dict))

if opt.log_metrics_path != "":
    df_metrics = pd.DataFrame()
time_steps_log = int(opt.nb_time_steps / opt.nb_logs)
hvac_nb_agents = config_dict["default_env_prop"]["cluster_prop"]["hvac_nb_agents"]
houses = env.cluster.houses
stations = env.cluster.stations

if opt.agent != "TarmacPPO":
    actors = {}
    for house_id in houses.keys():
        agent_prop = {"id": house_id}

        if opt.actor_name:
            agent_prop["actor_name"] = opt.actor_name
            agent_prop["net_seed"] = opt.net_seed

        actors[house_id] = agents_dict[opt.agent](agent_prop, config_dict, num_state=num_state)
        # actors[house_id] = AlwaysOnController(agent_prop, config_dict, num_state=num_state)
    for station_id in stations.keys():
        station_agent_prop = {"id": station_id}

        if opt.actor_name:
            station_agent_prop["actor_name"] = opt.actor_name
            station_agent_prop["net_seed"] = opt.net_seed
        actors[station_id] = agents_dict[opt.station_agent](station_agent_prop, config_dict, num_state=num_state)
else:
    agent_prop = {"net_seed" : opt.net_seed, "actor_name" : opt.actor_name}
    # actors = TarmacPPOAgent(agent_prop, config_dict, num_state=num_state)
    actors = TarmacPPOAgent(agent_prop, config_dict, num_obs_hvac=num_state, num_obs_station=num_state)

obs_dict = env.reset()


cumul_temp_offset = 0
cumul_temp_error = 0
max_temp_error = 0
cumul_hvac_active_signal_offset = 0
cumul_hvac_active_signal_error = 0
cumul_ev_active_signal_offset = 0
cumul_ev_active_signal_error = 0
cumul_ev_reactive_signal_offset = 0
cumul_ev_reactive_signal_error = 0

cumul_OD_temp = 0
cumul_hvac_active_signal = 0
cumul_ev_active_signal = 0
cumul_reactive_signal = 0
cumul_hvac_cons = 0
cumul_ev_cons = 0
cumul_cons = 0

cumul_squared_error_hvac_active_sig = 0
cumul_squared_error_ev_active_sig = 0
cumul_squared_error_reactive_sig = 0
cumul_squared_error_temp = 0
cumul_squared_max_error_temp = 0

actions, discrete_actions, continuous_actions = get_actions(actors, obs_dict)
t1_start = time.process_time() 

for i in range(nb_time_steps):
    obs_dict, _, _, info = env.step(actions)
    actions, discrete_actions, continuous_actions = get_actions(actors, obs_dict)
    if opt.log_metrics_path != "" and i >= opt.start_stats_from:  # 如果给了记录log地址, 并且大于初始时间步(初始化或预热期后开始记录数据), 则需要记录df. 需含有HVAC
        df = pd.DataFrame(obs_dict).transpose()
        if any(isinstance(x, int) for x in obs_dict.keys()) and any('charging_station' in str(x) for x in obs_dict.keys()):
            # HVAC相关计算
            df["temperature_difference"] = df["house_temp"] - df["house_target_temp"]
            df["temperature_error"] = np.abs(df["house_temp"] - df["house_target_temp"])
            temp_diff = df["temperature_difference"].mean()
            temp_err = df["temperature_error"].mean()
            air_temp = df["house_temp"].mean()
            mass_temp = df["house_mass_temp"].mean()
            target_temp = df["house_target_temp"].mean()
            OD_temp = df["OD_temp"][0]
            hvac_active_signal = df["grid_hvac_active_reg_signal"][0]
            hvac_consumption = df["cluster_hvac_active_power"][0]

            # EV相关计算
            charging_station_count = sum('charging_station' in str(x) for x in obs_dict.keys())
            soc_data = {}
            soc_target_data = {}
            ev_active_power_data = {}
            reactive_power_data = {}
            # 记录每个充电桩的SOC
            for count in range(charging_station_count):
                soc_column_name = f"soc{count+1}"
                soc_target_column_name = f"soc_target{count+1}"
                ev_active_power_column_name = f"ev active power{count+1}"
                ev_reactive_power_column_name = f"ev reactive power{count+1}"
                if df[f"battery_capacity"]["charging_station{}".format(count)] == 0:
                    soc_data[soc_column_name] = None
                    soc_target_data[soc_target_column_name] = None
                    ev_active_power_data[ev_active_power_column_name] = None
                    reactive_power_data[ev_reactive_power_column_name] = None
                else:
                    soc_data[soc_column_name] = df[f"current_battery_energy"]["charging_station{}".format(count)] / df[f"battery_capacity"]["charging_station{}".format(count)]
                    soc_target_data[soc_target_column_name] = df[f"soc_target_energy"]["charging_station{}".format(count)] / df[f"battery_capacity"]["charging_station{}".format(count)]
                    ev_active_power_data[ev_active_power_column_name] = df[f"current_ev_active_power"]["charging_station{}".format(count)]
                    reactive_power_data[ev_reactive_power_column_name] = df[f"current_ev_reactive_power"]["charging_station{}".format(count)]

            df['soc_error'] = df['current_battery_energy'] - df['soc_target_energy']
            soc_abs_diff = df['soc_error'].abs().mean()
            ev_consumption = df["cluster_ev_active_power"][0]
            total_ev_reactive_power = df['cluster_ev_reactive_power'][0]
            ev_active_signal = df["grid_ev_active_reg_signal"][0]
            ev_reactive_signal = df["grid_ev_reactive_reg_signal"][0]

            # 创建合并的数据行
            row_combined = pd.DataFrame({
                "temp_diff": temp_diff,
                "temp_err": temp_err,
                "air_temp": air_temp,
                "mass_temp": mass_temp,
                "target_temp": target_temp,
                "OD_temp": OD_temp,
                "HVAC Active Signal": hvac_active_signal,
                "EV Active Signal": ev_active_signal,
                "HVAC Consumption": hvac_consumption,
                "SOC Difference": soc_abs_diff,
                "EV Consumption": ev_consumption,
                "Total EV Reactive Power": total_ev_reactive_power,
                "EV Reactive Signal": ev_reactive_signal,
                **soc_data,
                **soc_target_data,
                **ev_active_power_data,
                **reactive_power_data
            }, index=[config_dict["default_env_prop"]["time_step"] * i])

            # 将合并的记录添加到性能数据的DataFrame中
            df_metrics = pd.concat([df_metrics, row_combined])
        elif any(isinstance(x, int) for x in obs_dict.keys()):
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
            # 总调节信号（signal）：控制HVAC功率输出的信号，反映了调节系统发出的指令。[0]表示选第一个智能体的信号,所有智能体都一样
            hvac_active_signal = df["grid_hvac_active_reg_signal"][0]
            # 功率消耗（consumption）：集群HVAC系统的总功率消耗
            hvac_consumption = df["cluster_hvac_active_power"][0]
            row_hvac = pd.DataFrame({"temp_diff":temp_diff, "temp_err":temp_err, "air_temp":air_temp, "mass_temp":mass_temp,"target_temp":target_temp, "OD_temp":OD_temp, "hvac_active_signal": hvac_active_signal, "hvac_consumption":hvac_consumption}, index=[config_dict["default_env_prop"]["time_step"]*i])
            df_metrics = pd.concat([df_metrics,row_hvac])
        elif any('charging_station' in str(x) for x in obs_dict.keys()):
            # 计算SOC误差
            # 提取充电桩数量
            charging_station_count = sum('charging_station' in str(x) for x in obs_dict.keys())
            soc_data = {}
            soc_target_data = {}
            ev_active_power_data = {}
            reactive_power_data = {}

            # 记录每个充电桩的SOC
            for count in range(charging_station_count):
                soc_column_name = f"soc{count+1}"
                soc_target_column_name = f"soc_target{count+1}"
                ev_active_power_column_name = f"ev active power{count+1}"
                ev_reactive_power_column_name = f"ev reactive power{count+1}"
                if df[f"battery_capacity"]["charging_station{}".format(count)] == 0:
                    soc_data[soc_column_name] = None
                    soc_target_data[soc_target_column_name] = None
                    ev_active_power_data[ev_active_power_column_name] = None
                    reactive_power_data[ev_reactive_power_column_name] = None
                else:
                    soc_data[soc_column_name] = df[f"current_battery_energy"]["charging_station{}".format(count)] / df[f"battery_capacity"]["charging_station{}".format(count)]
                    soc_target_data[soc_target_column_name] = df[f"soc_target_energy"]["charging_station{}".format(count)] / df[f"battery_capacity"]["charging_station{}".format(count)]
                    ev_active_power_data[ev_active_power_column_name] = df[f"current_ev_active_power"]["charging_station{}".format(count)]
                    reactive_power_data[ev_reactive_power_column_name] = df[f"current_ev_reactive_power"]["charging_station{}".format(count)]

            df['soc_error'] = df['current_battery_energy'] - df['soc_target_energy']
            
            soc_abs_diff = df['soc_error'].abs().mean()
            # 计算平均充放电功率
            average_charge_power = df['current_ev_active_power'].mean()
            # 总功率
            ev_consumption = df["cluster_ev_active_power"][0]
            ev_active_signal = df["grid_ev_active_reg_signal"][0]
            hvac_active_signal = df["grid_hvac_active_reg_signal"][0]
            total_ev_reactive_power = df['cluster_ev_reactive_power'][0]
            ev_reactive_signal = df["grid_ev_reactive_reg_signal"][0]
            # 创建一个用于记录的数据行
            row_ev = pd.DataFrame({
                "SOC Difference": soc_abs_diff,
                # "Average Charge/Discharge Power": average_charge_power,  # 跟Total Power重复了,没啥用
                "EV Consumption": ev_consumption,
                "EV Active Signal": ev_active_signal,
                "HVAC Active Signal": hvac_active_signal,
                "Total EV Reactive Power":total_ev_reactive_power,
                "EV Reactive Signal": ev_reactive_signal,
                **soc_data,
                **soc_target_data,
                **ev_active_power_data,
                **reactive_power_data,
            }, index=[config_dict["default_env_prop"]["time_step"] * i])
            # 将记录添加到性能数据的DataFrame中
            df_metrics = pd.concat([df_metrics, row_ev])
        

    if opt.render and i >= opt.render_after:
        renderer.render(obs_dict)
    # Max temperature error(max_temp_error_houses)：在当前记录间隔内遇到的最大温度误差。
    max_temp_error_houses = 0
    for agent_id, obs in obs_dict.items():
        if isinstance(agent_id, int):
            temp_error = obs_dict[agent_id]["house_temp"] - obs_dict[agent_id]["house_target_temp"]
            cumul_temp_offset += temp_error / env.hvac_nb_agents
            cumul_temp_error += np.abs(temp_error) / env.hvac_nb_agents
            if np.abs(temp_error) > max_temp_error:
                max_temp_error = np.abs(temp_error)
            if np.abs(temp_error) > max_temp_error_houses:
                max_temp_error_houses = np.abs(temp_error)

            if i >= opt.start_stats_from:
                cumul_squared_error_temp += temp_error**2
        elif 'charging_station' in agent_id:
            pass
            
    if any(isinstance(x, int) for x in obs_dict.keys()) and any('charging_station' in str(x) for x in obs_dict.keys()):  # 两种智能体都有
        if i>= opt.start_stats_from:
            cumul_squared_max_error_temp += max_temp_error_houses**2
        cumul_OD_temp += obs_dict[0]["OD_temp"]
        cumul_hvac_active_signal += obs_dict[0]["grid_hvac_active_reg_signal"]
        cumul_ev_active_signal += obs_dict["charging_station0"]["grid_ev_active_reg_signal"]
        cumul_reactive_signal += obs_dict["charging_station0"]["grid_ev_reactive_reg_signal"]
        cumul_hvac_cons += obs_dict[0]["cluster_hvac_active_power"]
        cumul_ev_cons += obs_dict["charging_station0"]["cluster_ev_active_power"]
        cumul_cons += obs_dict[0]["cluster_hvac_active_power"] + obs_dict["charging_station0"]["cluster_ev_active_power"]
        hvac_active_signal_error = obs_dict[0]["grid_hvac_active_reg_signal"] - obs_dict[0]["cluster_hvac_active_power"]
        ev_active_signal_error = obs_dict["charging_station0"]["grid_ev_active_reg_signal"] - obs_dict["charging_station0"]["cluster_ev_active_power"]
        reactive_signal_error = obs_dict["charging_station0"]["grid_ev_reactive_reg_signal"] - obs_dict["charging_station0"]["cluster_ev_reactive_power"]
        cumul_hvac_active_signal_offset += hvac_active_signal_error
        cumul_hvac_active_signal_error += np.abs(hvac_active_signal_error)
        cumul_ev_active_signal_offset += ev_active_signal_error
        cumul_ev_active_signal_error += np.abs(ev_active_signal_error)
        cumul_ev_reactive_signal_offset += reactive_signal_error
        cumul_ev_reactive_signal_error += np.abs(reactive_signal_error)
        
        
    elif any(isinstance(x, int) for x in obs_dict.keys()):  # 只有HVAC
        if i>= opt.start_stats_from:
            cumul_squared_max_error_temp += max_temp_error_houses**2
        cumul_OD_temp += obs_dict[0]["OD_temp"]
        cumul_hvac_active_signal += obs_dict[0]["grid_hvac_active_reg_signal"]
        cumul_ev_active_signal = 0
        cumul_reactive_signal = 0
        cumul_hvac_cons += obs_dict[0]["cluster_hvac_active_power"]
        cumul_ev_cons = 0
        cumul_cons += obs_dict[0]["cluster_hvac_active_power"]
        hvac_active_signal_error = obs_dict[0]["grid_hvac_active_reg_signal"] - obs_dict[0]["cluster_hvac_active_power"]
        ev_active_signal_error = 0
        reactive_signal_error = 0
        cumul_hvac_active_signal_offset += hvac_active_signal_error
        cumul_hvac_active_signal_error += np.abs(hvac_active_signal_error)
        cumul_ev_active_signal_offset = 0
        cumul_ev_active_signal_error = 0
        cumul_ev_reactive_signal_offset = 0
        cumul_ev_reactive_signal_error = 0

    elif any('charging_station' in str(x) for x in obs_dict.keys()):  # 只有EV
        cumul_OD_temp = 0
        cumul_hvac_active_signal = 0
        cumul_ev_active_signal += obs_dict["charging_station0"]["grid_ev_active_reg_signal"]
        cumul_reactive_signal += obs_dict["charging_station0"]["grid_ev_reactive_reg_signal"]
        cumul_hvac_cons = 0
        cumul_ev_cons += obs_dict["charging_station0"]["cluster_ev_active_power"]
        cumul_cons += obs_dict["charging_station0"]["cluster_ev_active_power"]
        hvac_active_signal_error = 0
        ev_active_signal_error = obs_dict["charging_station0"]["grid_ev_active_reg_signal"] - obs_dict["charging_station0"]["cluster_ev_active_power"]
        reactive_signal_error = obs_dict["charging_station0"]["grid_ev_reactive_reg_signal"] - obs_dict["charging_station0"]["cluster_ev_reactive_power"]
        cumul_hvac_active_signal_offset = 0
        cumul_hvac_active_signal_error = 0
        cumul_ev_active_signal_offset += ev_active_signal_error
        cumul_ev_active_signal_error += np.abs(ev_active_signal_error)
        cumul_ev_reactive_signal_offset += reactive_signal_error
        cumul_ev_reactive_signal_error += np.abs(reactive_signal_error)
    

    if i >= opt.start_stats_from:
        cumul_squared_error_hvac_active_sig += hvac_active_signal_error**2
        cumul_squared_error_ev_active_sig += ev_active_signal_error**2
        cumul_squared_error_reactive_sig += reactive_signal_error**2

    if i % time_steps_log == time_steps_log - 1:  # Log train statistics

        # print("Logging stats at time {}".format(t))

        #print("Average absolute noise: {} W".format(env.power_grid.cumulated_abs_noise / env.power_grid.nb_steps ))

        # Mean temperature offset(mean_temp_offset)在日志记录间隔内，房屋温度与目标温度差异的平均值。
        mean_temp_offset = cumul_temp_offset / time_steps_log
        # Mean temperature error(mean_temp_error)在日志记录间隔内，房屋温度与目标温度差异的绝对值的平均值。
        mean_temp_error = cumul_temp_error / time_steps_log
        # Mean outside temperature(mean_OD_temp)在日志记录间隔内，室外温度的平均值。
        mean_OD_temp = cumul_OD_temp / time_steps_log

        # Mean signal offset(mean_signal_offset)在日志记录间隔内，调节信号与实际集群HVAC功率之间差异的平均值。
        mean_ev_reactive_signal_offset = cumul_ev_reactive_signal_offset / time_steps_log
        mean_ev_reactive_signal_error = cumul_ev_reactive_signal_error / time_steps_log
        mean_reactive_signal = cumul_reactive_signal / time_steps_log
        mean_ev_active_signal_offset = cumul_ev_active_signal_offset / time_steps_log
        mean_ev_active_signal_error = cumul_ev_active_signal_error / time_steps_log
        mean_ev_active_signal = cumul_ev_active_signal / time_steps_log

        mean_hvac_active_signal_offset = cumul_hvac_active_signal_offset / time_steps_log
        # Mean signal error(mean_signal_error)在日志记录间隔内，调节信号与实际集群HVAC功率之间差异的绝对值的平均值。
        mean_hvac_active_signal_error = cumul_hvac_active_signal_error / time_steps_log
        # Mean signal(mean_signal)在日志记录间隔内，调节信号的平均值。
        mean_hvac_active_signal = cumul_hvac_active_signal / time_steps_log
        # Mean consumption(mean_consumption)在日志记录间隔内，集群HVAC功率消耗的平均值。
        mean_hvac_consumption = cumul_hvac_cons / time_steps_log
        mean_ev_consumption = cumul_ev_cons / time_steps_log
        mean_consumption = cumul_cons / time_steps_log

        # 循环中的逐步累积计算, 计算到目前为止的累积误差的均方根误差（RMSE）等
        if i >= opt.start_stats_from:
            if env.hvac_nb_agents == 0: # 避免除以 0
                rmse_temp = 0
                rms_max_error_temp = 0
                rmse_hvac_active_sig_per_ag = 0
            else:
                # RMSE temperature（rmse_temp）：计算所有代理的温度误差的均方根误差，再除以代理数量来获取平均值。这反映了智能体在维持目标温度方面的准确性。
                rmse_temp = np.sqrt(cumul_squared_error_temp/((i-opt.start_stats_from)*env.hvac_nb_agents))
                # RMS Max Error temperature(rms_max_error_temp)：在当前记录间隔内，房屋遇到的最大温度误差的均方根，这是对极端情况性能的量度。
                rms_max_error_temp = np.sqrt(cumul_squared_max_error_temp/(i-opt.start_stats_from))
                # RMSE signal per agent（rmse_active_sig_per_ag）：计算所有代理的信号误差的均方根误差，并除以代理数量来获取每个代理的平均值。这是评估信号跟踪性能的关键指标。    
                rmse_hvac_active_sig_per_ag = np.sqrt(cumul_squared_error_hvac_active_sig / (i - opt.start_stats_from)) / env.hvac_nb_agents

            if env.station_nb_agents == 0:
                rmse_ev_active_sig_per_ag = 0
                rmse_ev_reactive_sig_per_ag = 0
            else:
                rmse_ev_active_sig_per_ag = np.sqrt(cumul_squared_error_ev_active_sig / (i - opt.start_stats_from)) / env.station_nb_agents
                rmse_ev_reactive_sig_per_ag = np.sqrt(cumul_squared_error_reactive_sig / (i - opt.start_stats_from)) / env.station_nb_agents
        else:
            rmse_hvac_active_sig_per_ag = nan
            rmse_ev_active_sig_per_ag = nan
            rmse_ev_reactive_sig_per_ag = nan
            rmse_temp = nan 
            rms_max_error_temp = nan

        if log_wandb:
            wandb_run.log(
                {
                    "RMSE hvac active signal per agent": rmse_hvac_active_sig_per_ag,
                    "RMSE ev active signal per agent": rmse_ev_active_sig_per_ag,
                    "RMSE ev reactive signal per agent": rmse_ev_reactive_sig_per_ag,
                    "RMSE temperature": rmse_temp,
                    "RMS Max Error temperature": rms_max_error_temp,
                    "Mean temperature offset": mean_temp_offset,
                    "Mean temperature error": mean_temp_error,
                    "Max temperature error": max_temp_error,
                    "Mean hvac active signal offset": mean_hvac_active_signal_offset,
                    "Mean ev active signal offset": mean_ev_active_signal_offset,
                    "Mean ev reactive signal offset": mean_ev_reactive_signal_offset,
                    "Mean hvac active signal error": mean_hvac_active_signal_error,
                    "Mean ev active signal error": mean_ev_active_signal_error,
                    "Mean ev reactive signal error": mean_ev_reactive_signal_error,
                    "Mean outside temperature": mean_OD_temp,
                    "Mean hvac active signal" : mean_hvac_active_signal,
                    "Mean ev active signal" : mean_ev_active_signal,
                    "Mean ev reactive signal" : mean_reactive_signal,
                    "Mean hvac consumption": mean_hvac_consumption,
                    "Mean ev consumption": mean_ev_consumption,
                    "Mean consumption": mean_consumption,
                    # 当前时间步中的小时数，这可能用于分析一天中不同时间的性能。
                    # "Time (hour)": obs_dict[0]["datetime"].hour,
                    "Time (hour)": next(  # 处理有可能没有某种智能体的情况
                        (obs_dict[key]["datetime"].hour for key in obs_dict.keys() if isinstance(key, int) or 'charging_station' in str(key)), 
                        -1
                    ),
                    # 当前的时间步数或迭代数。
                    "Time step": i,
                }
            )

        cumul_temp_offset = 0
        cumul_temp_error = 0
        max_temp_error = 0
        cumul_hvac_active_signal_offset = 0
        cumul_ev_active_signal_offset = 0
        cumul_ev_reactive_signal_offset = 0
        cumul_hvac_active_signal_error = 0
        cumul_ev_active_signal_error = 0
        cumul_ev_reactive_signal_error = 0
        cumul_OD_temp = 0
        cumul_hvac_active_signal = 0
        cumul_ev_active_signal = 0
        cumul_reactive_signal = 0
        cumul_cons = 0
        cumul_hvac_cons = 0
        cumul_ev_cons = 0
        print("Time step: {}".format(i))
        t1_stop = time.process_time()
        print("Elapsed time for {}% of steps: {} seconds.".format(int(np.round(float(i)/nb_time_steps*100)), int(t1_stop - t1_start))) 
# 循环结束时的最终计算：在模拟结束后，代码再次计算RMSE，这次使用的是从 opt.start_stats_from 到最后一个时间步的所有数据。这为整个模拟提供了最终的性能评估。
if env.station_nb_agents != 0:
    rmse_ev_active_sig_per_ag = np.sqrt(cumul_squared_error_ev_active_sig/(nb_time_steps-opt.start_stats_from))/(env.station_nb_agents)
    rmse_ev_reactive_sig_per_ag = np.sqrt(cumul_squared_error_reactive_sig/(nb_time_steps-opt.start_stats_from))/(env.station_nb_agents)
    print("RMSE EV Reactive Signal Per Agent: {} W".format(int(rmse_ev_reactive_sig_per_ag)))
    print("RMSE EV Active Signal Per Agent: {} W".format(int(rmse_ev_active_sig_per_ag)))
if any(isinstance(x, int) for x in obs_dict.keys()):
    rmse_hvac_active_sig_per_ag = np.sqrt(cumul_squared_error_hvac_active_sig/(nb_time_steps-opt.start_stats_from))/(env.hvac_nb_agents+env.station_nb_agents)
    rmse_temp = np.sqrt(cumul_squared_error_temp/((nb_time_steps-opt.start_stats_from)*env.hvac_nb_agents))
    rms_max_error_temp = np.sqrt(cumul_squared_max_error_temp/(nb_time_steps-opt.start_stats_from))
    print("RMSE HVAC Active Signal Per Agent: {} W".format(int(rmse_hvac_active_sig_per_ag)))
    print("RMSE Temperature: {} C".format(rmse_temp))
    print("RMS Max Error Temperature: {} C".format(rms_max_error_temp))


#print("Average absolute noise: {} W".format(env.power_grid.cumulated_abs_noise / env.power_grid.nb_steps ))
if log_wandb:
    if any(isinstance(x, int) for x in obs_dict.keys()) and any('charging_station' in str(x) for x in obs_dict.keys()):
        wandb_run.log({
            "RMSE hvac active signal per agent": rmse_hvac_active_sig_per_ag,
            "RMSE ev active signal per agent": rmse_ev_active_sig_per_ag,
            "RMSE ev reactive signal per agent": rmse_ev_reactive_sig_per_ag,
            "RMSE temperature": rmse_temp,
            "RMS Max Error temperature": rms_max_error_temp,
            }
        )
    elif any(isinstance(x, int) for x in obs_dict.keys()):
        wandb_run.log({
            "RMSE hvac active signal per agent": rmse_hvac_active_sig_per_ag,
            # "RMSE ev reactive signal per agent": rmse_ev_reactive_sig_per_ag,
            "RMSE temperature": rmse_temp,
            "RMS Max Error temperature": rms_max_error_temp,
            }
        )
    elif any('charging_station' in str(x) for x in obs_dict.keys()):
        wandb_run.log({
            "RMSE ev active signal per agent": rmse_ev_active_sig_per_ag,
            "RMSE ev reactive signal per agent": rmse_ev_reactive_sig_per_ag,
            # "RMSE temperature": rmse_temp,
            # "RMS Max Error temperature": rms_max_error_temp,
            }
        )
if opt.log_metrics_path != "":
    # 获取当前时间
    current_time = config_dict["default_env_prop"]["start_real_date"]
    # 构建新的文件名，包含actor_name和当前时间
    filename = f"{opt.actor_name}-{current_time}.csv"
    # 生成新的完整路径
    new_file_path = opt.log_metrics_path.replace('log_metrics.csv', filename)
    df_metrics.to_csv(new_file_path)
    print(f"Metrics saved to {new_file_path}")