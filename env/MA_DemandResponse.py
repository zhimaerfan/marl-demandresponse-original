from distutils.command.config import config
from sympy import octave_code
import gym
import ray
import numpy as np
import warnings
import random
from copy import deepcopy
import json
import csv

from datetime import datetime, timedelta, time
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.typing import MultiAgentDict, AgentID
from typing import Tuple, Dict, List, Any
import sys
from utils import applyPropertyNoise, Perlin, deadbandL2
import time

# Efan
from cli import cli_train  # 只引入种子. 其他的参数不要使用,在main函数里已经调整过
import os
import contextlib
from elvis.simulate import simulate
from elvis.utility.elvis_general import create_time_steps, num_time_steps
from dateutil.relativedelta import relativedelta
from elvis.config import ScenarioConfig

# import noise
# import wandb


sys.path.append("..")
sys.path.append("./monteCarlo")
from utils import (
    applyPropertyNoise,
    clipInterpolationPoint,
    sortDictKeys,
    house_solar_gain,
)
from interpolation import PowerInterpolator


class MADemandResponseEnv(MultiAgentEnv):
    """
    用于模拟需求响应环境的基本框架，其中包括多个房屋、电网和需求响应代理之间的交互。
    Multi agent demand response environment

    Attributes:

    default_env_prop: dictionary, containing the default configuration properties of the environment
    default_house_prop: dictionary, containing the default configuration properties of houses
    noise_house_prop: dictionary, containing the noise properties of houses' properties
    default_hvac_prop: dictionary, containing the default configuration properties of HVACs
    noise_hvac_prop: dictionary, containing the noise properties of HVACs' properties
    env_properties: a dictionary, containing the current configuration properties of the environment.
    start_datetime: a datetime object, representing the date and time at which the simulation starts.
    datetime: a datetime object, representing the current date and time.
    time_step: a timedelta object, representing the time step for the simulation.
    hvac_agent_ids: a list, containing the ids of every agents of the environment.
    hvac_nb_agents: an int, with the number of agents
    cluster: a ClusterAgents object modeling all the houses and stations.
    power_grid: a PowerGrid object, modeling the power grid.

    Main functions:

    build_environment(self): Builds a new environment with noise on properties
    reset(self): Reset the environment
    step(self, action_dict): take a step in time for each TCL, given actions of TCL agents
    compute_rewards(self, temp_penalty_dict, cluster_hvac_active_power, power_grid_reg_signal): compute the reward of each TCL agent

    Helper functions:
    merge_cluster_powergrid_obs(self, cluster_obs_dict, power_grid_reg_signal, cluster_hvac_active_power): merge the cluster and powergrid observations for the TCL agents
    make_dones_dict(self): create the "done" signal for each TCL agent
    """

    # 这3行代码是 Python 类型注解（type annotation），用于指定类属性的类型。这有助于提高代码的可读性和可维护性
    start_datetime: datetime
    datetime: datetime
    time_step: timedelta

    def __init__(self, config, test=False):
        """
        环境的基本属性被初始化，如设置配置参数、日期时间属性、初始化时间步长、噪声属性、定义智能体的 ID 等。

        参数:
        config: 包含环境配置属性的字典。
        test: 是否为测试环境的布尔值。

        Initialize the environment

        Parameters:
        config: dictionary, containing the default configuration properties of the environment, house, hvac, and noise
        test: boolean, true it is a testing environment, false if it is for training

        """
        super(MADemandResponseEnv, self).__init__()
        self.test = test
        self.default_env_prop = config["default_env_prop"]
        self.default_ev_prop = config["default_ev_prop"]  # Efan 设置默认电动车属性
        self.default_house_prop = config["default_house_prop"]
        self.default_hvac_prop = config["default_hvac_prop"]
        if test:
            self.noise_house_prop = config["noise_house_prop_test"]
            self.noise_hvac_prop = config["noise_hvac_prop_test"]
        else:
            self.noise_house_prop = config["noise_house_prop"]
            self.noise_hvac_prop = config["noise_hvac_prop"]
        self.build_environment()


    def build_environment(self):
        """
        该方法用于构建环境，包括初始化或重置环境中的各个组件的状态，如房屋、电网等。
        """

        # 添加噪声,差异化房屋,都存在新的env_properties中, EV等智能体先不实例化,注意在这之前只有参数,没有对象
        self.env_properties = applyPropertyNoise(
            self.default_env_prop,
            self.default_house_prop,
            self.noise_house_prop,
            self.default_hvac_prop,
            self.noise_hvac_prop,
        )

        self.start_datetime = self.env_properties[
            "start_datetime"
        ]  # Start date and time
        self.datetime = self.start_datetime  # Current time

        self.time_step = timedelta(seconds=self.env_properties["time_step"])

        # hvac_agent_ids只包含HVAC智能体
        self.hvac_agent_ids = self.env_properties["hvac_agent_ids"]
        self.hvac_nb_agents = len(self.hvac_agent_ids)

        # 保存充电站和 EV 代理信息到环境属性中  目前,EV智能体就是充电桩智能体,后续可能会改进. ID stations_agent_ids':['charging_station0', ...]
        num_stations = self.default_ev_prop["num_stations"]  # 已经在adjust中更新过该参数,保持与参数station_nb_agents一致
        self.stations_agent_ids = ["charging_station" + str(i) for i in range(num_stations)]
        self.station_nb_agents = len(self.stations_agent_ids)

        # 更新处理完后的ev参数,其实应该在adjust中写这部分,懒的改了
        self.env_properties["cluster_prop"]["stations_agent_ids"] = self.stations_agent_ids
        self.env_properties["cluster_prop"]["default_ev_prop"] = self.default_ev_prop

        # 需要调整,实例化房屋集群和电网
        self.cluster = ClusterAgents(
            # cluster_prop包括HVAC和EV
            self.env_properties,
            self.hvac_agent_ids,
            self.stations_agent_ids,
            self.datetime,
            self.time_step,
            self.default_env_prop,
        )
        self.all_agent_ids = self.cluster.all_agent_ids
        self.all_nb_agents = self.cluster.all_nb_agents
        # Efan's 待办 还需要加上EV的最大功耗
        self.env_properties["power_grid_prop"]["max_hvac_power"] = self.cluster.max_hvac_power

        # 初始化 PowerGrid
        self.power_grid = PowerGrid(
            self.env_properties["power_grid_prop"],
            self.default_house_prop,
            self.env_properties["nb_hvac"],
            self.cluster,
            self.default_ev_prop  # Efan 添加 EV 相关配置
        )
        self.power_grid.step(self.start_datetime, self.time_step)

    def reset(self):
        """
        重置环境到初始状态，并生成每个TCL代理的观察结果。
            首先调用build_environment方法，初始化或重置环境中各个组件的状态。
            调用make_cluster_obs_dict方法，生成集群观察字典，包含每个代理的观察结果。
            获取电网的当前调节信号和集群的HVAC总功率。
            调用merge_cluster_powergrid_obs方法，将集群观察字典、电网调节信号和集群HVAC功率合并，生成最终的观察字典。
            返回观察字典，供代理在训练或决策时使用。        

        返回:
        obs_dict: 字典，包含每个TCL代理的观察结果。

        参数:
        self

        Reset the environment.

        Returns:
        obs_dict: a dictionary, contaning the observations for each TCL agent.

        Parameters:
        self
        """

        self.build_environment()

        cluster_obs_dict = self.cluster.make_cluster_obs_dict(self.datetime)
        power_grid_reg_signal = (self.power_grid.current_hvac_active_signal, self.power_grid.current_ev_active_signal, self.power_grid.current_ev_reactive_signal)
        cluster_hvac_active_power = self.cluster.cluster_hvac_active_power

        # reset计算EV相关状态，以供所有智能体使用
        cluster_ev_active_power = sum(station.current_ev_active_power for station in self.cluster.stations.values())
        cluster_ev_reactive_power = sum(station.current_ev_reactive_power for station in self.cluster.stations.values())
        ev_queue_count = len(self.cluster.stations_manager.pending_charging_events)  # 还在排队的ev

        # 19个集群观察结果+电网的调节信号+集群的HVAC功率=21
        obs_dict = self.merge_cluster_powergrid_obs(
            cluster_obs_dict, power_grid_reg_signal, cluster_hvac_active_power, cluster_ev_active_power, cluster_ev_reactive_power, ev_queue_count
        )

        return obs_dict

    def step(self, action_dict):
        """
        该方法用于使环境向前推进一步，根据需求响应代理采取的行动（actions）来更新环境状态。
        返回观察结果、奖励、完成标志和附加信息.
        
        Take a step in time for each TCL, given actions of TCL agents

        Returns:
        obs_dict: a dictionary, containing the observations for each TCL agent.
        rewards_dict: a dictionary, containing the rewards of each TCL agent.
        dones_dict: a dictionary, containing the "done" signal for each TCL agent.
        info_dict: a dictonary, containing additional information for each TCL agent.

        Parameters:
        self
        action_dict: a dictionary, containing the actions taken per each agent.
        """

        self.datetime += self.time_step
        # Cluster step
        cluster_obs_dict, cluster_hvac_active_power, cluster_ev_active_power, cluster_ev_reactive_power, ev_queue_count, _ = self.cluster.step(
            self.datetime, action_dict, self.time_step
        )

        # Compute reward with the old grid signal
        rewards_dict = self.compute_rewards(cluster_hvac_active_power, cluster_ev_active_power, cluster_ev_reactive_power)

        # Power grid step
        power_grid_reg_signal = self.power_grid.step(self.datetime, self.time_step)

        # Merge observations
        obs_dict = self.merge_cluster_powergrid_obs(
            cluster_obs_dict, power_grid_reg_signal, cluster_hvac_active_power, cluster_ev_active_power, cluster_ev_reactive_power, ev_queue_count
        )

        dones_dict = self.make_dones_dict()
        info_dict = {"cluster_hvac_active_power": cluster_hvac_active_power}
        # print("cluster_hvac_active_power: {}, power_grid_reg_signal: {}".format(cluster_hvac_active_power, power_grid_reg_signal))

        return obs_dict, rewards_dict, dones_dict, info_dict

    def merge_cluster_powergrid_obs(
        self, cluster_obs_dict, power_grid_reg_signal, cluster_hvac_active_power, cluster_ev_active_power, cluster_ev_reactive_power, ev_queue_count
    ) -> None:
        """
        该方法用于合并集群和电网观察结果，以创建最终的观察结果，供代理在训练或决策时使用。
        
        Merge the cluster and powergrid observations for the TCL agents

        Returns:
        obs_dict: a dictionary, containing the observations for each TCL agent.

        Parameters:
        cluster_obs_dict: a dictionary, containing the cluster observations for each TCL agent.
        power_grid_reg_signal: a float. Regulation signal, or target total power, in Watts.
        cluster_hvac_active_power: a float. Total power used by the TCLs, in Watts.
        """

        obs_dict = cluster_obs_dict


        # 给所有智能体，添加全局观察结果
        for hvac_agent_id in self.hvac_agent_ids:
            obs_dict[hvac_agent_id]["grid_hvac_active_reg_signal"] = power_grid_reg_signal[0]
            obs_dict[hvac_agent_id]["grid_ev_active_reg_signal"] = power_grid_reg_signal[1]
            obs_dict[hvac_agent_id]["grid_ev_reactive_reg_signal"] = power_grid_reg_signal[2]
            obs_dict[hvac_agent_id]["cluster_hvac_active_power"] = cluster_hvac_active_power
            obs_dict[hvac_agent_id]["cluster_ev_active_power"] = cluster_ev_active_power
            obs_dict[hvac_agent_id]["cluster_ev_reactive_power"] = cluster_ev_reactive_power
            obs_dict[hvac_agent_id]["ev_queue_count"] = ev_queue_count

        for station_agent_id in self.stations_agent_ids:
            obs_dict[station_agent_id]["grid_hvac_active_reg_signal"] = power_grid_reg_signal[0]
            obs_dict[station_agent_id]["grid_ev_active_reg_signal"] = power_grid_reg_signal[1]
            obs_dict[station_agent_id]["grid_ev_reactive_reg_signal"] = power_grid_reg_signal[2]
            obs_dict[station_agent_id]["cluster_hvac_active_power"] = cluster_hvac_active_power
            obs_dict[station_agent_id]["cluster_ev_active_power"] = cluster_ev_active_power
            obs_dict[station_agent_id]["cluster_ev_reactive_power"] = cluster_ev_reactive_power
            obs_dict[station_agent_id]["ev_queue_count"] = ev_queue_count

        return obs_dict

    def reg_signal_penalty(self, cluster_hvac_active_power, cluster_ev_active_power, cluster_ev_reactive_power):
        """
        该方法用于计算与电网调节信号和需求响应代理总功率之间的惩罚，以鼓励代理适应电网调节信号。
        
        Returns: a float, representing the positive penalty due to the distance between the regulation signal and the total power used by the TCLs.

        Parameters:
        - cluster_ev_active_power: Total active power used by EV charging stations.
        - cluster_ev_reactive_power: Total EV Reactive Power used by EV charging stations.
        cluster_hvac_active_power: a float. Total power used by the TCLs, in Watts.
        power_grid_reg_signal: a float. Regulation signal, or target total power, in Watts.
        """
        sig_penalty_mode = self.default_env_prop["reward_prop"]["sig_penalty_mode"]
        
        self.power_grid.total_max_station_power # Efan 未修改待办
        
        # Efan 惩罚还需要修改
        if sig_penalty_mode == "common_L2":
            if self.hvac_nb_agents != 0:
                hvac_active_signal_penalty = ((cluster_hvac_active_power - self.power_grid.current_hvac_active_signal) / (self.hvac_nb_agents)) ** 2
            else:
                hvac_active_signal_penalty = 0
            if self.station_nb_agents != 0:
                ev_active_signal_penalty = ((cluster_ev_active_power - self.power_grid.current_ev_active_signal) / (self.station_nb_agents)) ** 2
                ev_reactive_signal_penalty = ((cluster_ev_reactive_power - self.power_grid.current_ev_reactive_signal) / self.station_nb_agents) ** 2
            else:
                ev_reactive_signal_penalty = 0
                ev_active_signal_penalty = 0
        else:
            raise ValueError("Unknown signal penalty mode: {}".format(sig_penalty_mode))

        return hvac_active_signal_penalty, ev_active_signal_penalty, ev_reactive_signal_penalty

    def compute_ev_reward(self, agent_id, cluster_ev_active_power, cluster_ev_reactive_power):
        # 不在这里考虑有功和无功的跟踪，EV也一定会在最后离开之前充满，还有别的即时惩罚可行吗？
        # Efan 待完善. 基于EV充电桩的特定目标和约束来定义奖励计算逻辑 每个智能体有独立的reward
        # 示例：考虑充电效率、电池状态、电网负荷等
        # 这里需要具体定义计算奖励的逻辑
        # 如果达到了需要使用最大功率充电,则使用剩余时间的负数作为惩罚,可作为创新点
        # 获取当前充电桩的状态信息
        station = self.cluster.stations[agent_id]
        remaining_time = station.remaining_departure_time
        remaining_controllable_time = station.remaining_controllable_time
        if station.connected_ev == None:
            current_soc = 0
            target_soc = 0
        else:
            current_soc = station.connected_ev.soc
            target_soc = station.soc_target_energy / station.battery_capacity
            
        remaining_time_percentage = remaining_time / station.park_time if station.park_time != 0 else 0
        # if remaining_time > 0:
        #     i=1
        # 电动汽车充电的惩罚,正常是有平方的正值,越大越惩罚
        # 计算SoC惩罚，当达到目标SoC时惩罚为0
        soc_penalty = (target_soc - current_soc) ** 2 if current_soc < target_soc else 0

        time_penalty = (remaining_time - remaining_controllable_time) / self.default_ev_prop["mean_park"] / 3600  # 似乎是原始的更好?
        # time_penalty = ((remaining_time - remaining_controllable_time) / self.default_ev_prop["mean_park"] / 3600)**2  
        # time_penalty = min(9,((remaining_time - remaining_controllable_time) / (remaining_controllable_time + 0.0001))**2) if station.connected_ev != None else 0  # 随着时间增加,EV可控时间会减少,不可控时间即差值应减少(提高soc). 否则该比例会增大

        return soc_penalty, time_penalty, station.connected_ev

    def compute_temp_penalty(self, one_house_id):
        """
        该方法用于计算每个房屋的室内温度与目标温度之间的惩罚，以鼓励代理使室内温度接近目标温度。
        该函数计算了一个房屋与其目标温度之间的温度惩罚。这个惩罚是基于房屋的当前温度与其目标温度之间的差异来计算的。函数的主要目的是为了在强化学习中为智能体提供关于其行为好坏的反馈。

        Returns: a float, representing the positive penalty due to distance between the target (indoors) temperature and the indoors temperature in a house.

        Parameters:
        target_temp: a float. Target indoors air temperature, in Celsius.
        deadband: a float. Margin of tolerance for indoors air temperature difference, in Celsius.
        house_temp: a float. Current indoors air temperature, in Celsius
        """
        temp_penalty_mode = self.default_env_prop["reward_prop"]["temp_penalty_mode"]

        if temp_penalty_mode == "individual_L2":

            house = self.cluster.houses[one_house_id]
            temperature_penalty = deadbandL2(
                house.target_temp, house.deadband, house.current_temp
            )

            # temperature_penalty = np.clip(temperature_penalty, 0, 20)

        elif temp_penalty_mode == "common_L2":
            ## Mean of all houses L2
            temperature_penalty = 0
            for house_id in self.hvac_agent_ids:
                house = self.cluster.houses[house_id]
                house_temperature_penalty = deadbandL2(
                    house.target_temp, house.deadband, house.current_temp
                )
                temperature_penalty += house_temperature_penalty / self.hvac_nb_agents

        elif temp_penalty_mode == "common_max":
            temperature_penalty = 0
            for house_id in self.hvac_agent_ids:
                house = self.cluster.houses[house_id]
                house_temperature_penalty = deadbandL2(
                    house.target_temp, house.deadband, house.current_temp
                )
                if house_temperature_penalty > temperature_penalty:
                    temperature_penalty = house_temperature_penalty

        elif temp_penalty_mode == "mixture":
            temp_penalty_params = self.default_env_prop["reward_prop"][
                "temp_penalty_parameters"
            ][temp_penalty_mode]

            ## Common and max penalties
            common_L2 = 0
            common_max = 0
            for house_id in self.hvac_agent_ids:
                house = self.cluster.houses[house_id]
                house_temperature_penalty = deadbandL2(
                    house.target_temp, house.deadband, house.current_temp
                )
                if house_id == one_house_id:
                    ind_L2 = house_temperature_penalty
                common_L2 += house_temperature_penalty / self.hvac_nb_agents
                if house_temperature_penalty > common_max:
                    common_max = house_temperature_penalty

            ## Putting together
            alpha_ind_L2 = temp_penalty_params["alpha_ind_L2"]
            alpha_common_L2 = temp_penalty_params["alpha_common_L2"]
            alpha_common_max = temp_penalty_params["alpha_common_max"]
            temperature_penalty = (
                alpha_ind_L2 * ind_L2
                + alpha_common_L2 * common_L2
                + alpha_common_max * common_max
            ) / (alpha_ind_L2 + alpha_common_L2 + alpha_common_max)

        else:
            raise ValueError(
                "Unknown temperature penalty mode: {}".format(temp_penalty_mode)
            )

        return temperature_penalty

    def compute_rewards(self, cluster_hvac_active_power, cluster_ev_active_power, cluster_ev_reactive_power):
        """
        该方法用于计算每个需求响应代理的奖励，根据温度惩罚和电网调节信号惩罚的组合计算奖励。

        Compute the reward of each TCL agent

        Returns:
        rewards_dict: a dictionary, containing the rewards of each TCL agent.

        Parameters:
        temp_penalty_dict: a dictionary, containing the temperature penalty for each TCL agent
        cluster_hvac_active_power: a float. Total power used by the TCLs, in Watts.
        power_grid_reg_signal: a float. Regulation signal, or target total power, in Watts.
        """

        rewards_dict: dict[str, float] = {}
        hvac_active_signal_penalty, ev_active_signal_penalty, ev_reactive_signal_penalty = self.reg_signal_penalty(cluster_hvac_active_power, cluster_ev_active_power, cluster_ev_reactive_power)

        # 归一化惩罚的分母
        norm_temp_penalty = deadbandL2(
            self.default_house_prop["target_temp"],
            0,
            self.default_house_prop["target_temp"] + 1,
        )

        norm_hvac_active_sig_penalty = deadbandL2(
            self.default_env_prop["reward_prop"]["norm_active_reg_sig"][0],
            0,
            0.75 * self.default_env_prop["reward_prop"]["norm_active_reg_sig"][0],
        )

        norm_ev_active_sig_penalty = deadbandL2(
            self.default_env_prop["reward_prop"]["norm_active_reg_sig"][1],
            0,
            0.75 * self.default_env_prop["reward_prop"]["norm_active_reg_sig"][1],
        )

        # Efan 需要修改
        norm_ev_reactive_sig_penalty = deadbandL2(
            self.default_env_prop["reward_prop"]["norm_reactive_reg_sig"],
            0,
            0.75 * self.default_env_prop["reward_prop"]["norm_reactive_reg_sig"],
        )

        # temp_penalty_dict = {}
        # # Temperature penalties
        # for house_id in self.hvac_agent_ids:
        #     house = self.cluster.houses[house_id]
        #     temp_penalty_dict[house_id] = self.compute_temp_penalty(house_id)

        # for hvac_agent_id in self.hvac_agent_ids:
        #     rewards_dict[hvac_agent_id] = -1 * (
        #         self.env_properties["reward_prop"]["alpha_temp"]
        #         * temp_penalty_dict[hvac_agent_id]
        #         / norm_temp_penalty
        #         + self.env_properties["reward_prop"]["alpha_hvac_active_sig"]
        #         * hvac_active_signal_penalty
        #         / norm_hvac_active_sig_penalty
        #     )
            
        for agent_id in self.all_agent_ids:
            if agent_id in self.hvac_agent_ids:  # 对于HVAC智能体
                temp_penalty = self.compute_temp_penalty(agent_id)
                reward = -1 * (
                    self.env_properties["reward_prop"]["alpha_temp"]
                    * temp_penalty
                    / norm_temp_penalty
                    + self.env_properties["reward_prop"]["alpha_hvac_active_sig"]
                    * hvac_active_signal_penalty
                    / norm_hvac_active_sig_penalty
                )
            elif agent_id.startswith('charging_station'):  # 对于EV充电桩智能体
                soc_penalty, time_penalty , connected_ev = self.compute_ev_reward(agent_id, cluster_ev_active_power, cluster_ev_reactive_power)
                # 结合有功和无功调节的表现
                        # 结合电网信号调节的惩罚和电动汽车的充电目标
                ev_reward = (
                    - self.env_properties["reward_prop"]["alpha_ev_soc_penalty"] * soc_penalty
                    - self.env_properties["reward_prop"]["alpha_ev_time_penalty"] * time_penalty
                )  
                this_ev_reactive_signal_penalty = self.env_properties["reward_prop"]["alpha_ev_reactive_sig"] * ev_reactive_signal_penalty / norm_ev_reactive_sig_penalty
                this_ev_active_signal_penalty = self.env_properties["reward_prop"]["alpha_ev_active_sig"] * ev_active_signal_penalty / norm_ev_active_sig_penalty if connected_ev != None else 0
                
                reward = ev_reward - this_ev_reactive_signal_penalty - this_ev_active_signal_penalty
                # print(agent_id, "Penalty: SoC", soc_penalty, ", time", time_penalty,", P_sig", ev_active_signal_penalty, ", Q_sig",ev_reactive_signal_penalty)
            rewards_dict[agent_id] = reward
        return rewards_dict

    def make_dones_dict(self):
        """
        该方法用于创建每个需求响应代理的完成标志，指示代理是否已经完成任务。

        Create the "done" signal for each TCL agent

        Returns:
        done_dict: a dictionary, containing the done signal of each TCL agent.

        Parameters:
        self
        """
        dones_dict: dict[str, bool] = {}
        for hvac_agent_id in self.hvac_agent_ids:
            dones_dict[
                hvac_agent_id
            ] = False  # There is no state which terminates the environment.
        return dones_dict

class StationsManager:
    """
    只负责充电桩和EV的连接和断开。
    这与 HVAC 类的角色不完全相同，因为 HVAC 类专注于单个智能体（即单个HVAC系统）的行为和状态。
    """
    def __init__(self, stations, stations_properties, time_step):
        self.stations = stations
        self.stations_properties = stations_properties
        self.processed_events = set()  # 用于跟踪已处理的充电事件ID, 防止重复处理,现实世界中需要修改,这里的EVid均不相同
        # Efan 设置env_seed
        self.env_seed = cli_train().env_seed
        np.random.seed(self.env_seed)
        random.seed(self.env_seed)

    def update_stations(self, charging_events, date_time):
        # 获取当前时间应处理的充电事件, 不包括已处理的, pending_charging_events最后会变成除去充电正在排队的
        # self.pending_charging_events = [event for event in charging_events if event.arrival_time <= date_time < event.leaving_time and event.id not in self.processed_events]
        if self.stations_properties["process_existing_events"] == "Previously":
            # 包括在当前时间点之前到达的EV(这些EV可能需要最大功率充电,会使系统不稳定,需要定制其初始化时刻的soc)
            self.pending_charging_events = [
                event for event in charging_events
                if event.arrival_time <= date_time < event.leaving_time and event.id not in self.processed_events
            ]
        elif self.stations_properties["process_existing_events"] == "Newly":
            # 只处理从当前时间点之后到达的EV, 即排队超过2h的EV将被排除
            n_hours_ago = date_time - timedelta(hours=self.stations_properties["mean_park"]/6)
            self.pending_charging_events = [
                event for event in charging_events
                if n_hours_ago <= event.arrival_time < date_time < event.leaving_time and event.id not in self.processed_events
            ]
            
        # 收集所有当前空闲的充电站
        idle_stations = [station for station in self.stations.values() if not station.is_occupied]
        if len(self.pending_charging_events) > 0:
            print("Num pending charging events:", len(self.pending_charging_events),", Num processed_events:", len(self.processed_events), "idle_stations:", len(idle_stations), date_time)

        # Efan's 已修改为随机分配EV, 不然有的智能体一直闲置. 为每个充电桩分配EV（如果有可用的）
        if self.stations_properties["station_type"] == "private": 

            while self.pending_charging_events and idle_stations:
                # 随机选择一个空闲的充电站
                selected_station = random.choice(idle_stations)
                # 从待处理事件中取出一个事件（这里仍然按顺序取，因为随机性已经在选择充电站时体现）
                ev_event = self.pending_charging_events.pop(0)
                # 将EV连接到选中的充电站
                selected_station.connect_ev(ev_event, date_time)
                self.processed_events.add(ev_event.id)
                # 将这个充电站从空闲充电站列表中移除，因为它现在已被占用
                idle_stations.remove(selected_station)
                
            # 对于没有连接EV的充电站，确保它们的状态被重置. 这里不是必要的, 因为update_status中有disconnect_ev, 属于重复了
            for station in idle_stations:
                station.disconnect_ev()

            # 更新所有充电站的状态
            for station in self.stations.values():
                station.update_status(date_time)

        else: # "public"顺序模式, 按顺序安排EV充电. 如优先快充桩, 最后才慢充
            for station in self.stations.values():
                if not station.is_occupied:
                    if self.pending_charging_events:
                        ev_event = self.pending_charging_events.pop(0)
                        station.connect_ev(ev_event, date_time)  # 连接新到达的EV
                        self.processed_events.add(ev_event.id)
                    else:
                        station.disconnect_ev()  # 没有等待的EV时重置状态, 也不是必要的, 因为update_status中有disconnect_ev, 属于重复了
                station.update_status(date_time)  # 更新充电桩状态
        # print("仍排队的队列 Num pending charging events", len(self.pending_charging_events), date_time)

    def generate_ev_charging_events(self, env_properties):
        """
        使用 Elvis 生成电动车充电事件队列。

        参数:
        ev_config: 电动车配置参数。

        返回:
        charging_events: 更新后的充电事件列表。
        """
        # 配置 Elvis 环境
        start_datetime = env_properties["start_datetime"]
        start_date_adjusted = start_datetime - timedelta(days=1)
        start_date_adjusted_str = start_date_adjusted.strftime("%Y-%m-%dT%H:%M:%S")  # 使用调整后的起始时间，提前一天. 使用原始系统的,因为原系统的起始时间可能为random模式

        ev_config=env_properties["cluster_prop"]["default_ev_prop"]

        # 设置随机种子以生成事件
        np.random.seed(self.env_seed)
        random.seed(self.env_seed)

        print("生成一次EV序列generate_ev_charging_events")
        elvis_config = ScenarioConfig.from_dict(ev_config)
        elvis_realisation = elvis_config.create_realisation(
            start_date_adjusted_str,  # 原系统的起始时间, 而不是使用ev_config["start_date"], 因为这可能会造成实验起始时就有EV已经排队很久必须以最大功率充电.
            ev_config["end_date"], 
            ev_config["resolution"]
        )

        # 使用 Elvis 生成电动车序列
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            result = simulate(elvis_realisation)
        
        # 获取充电事件
        charging_events = elvis_realisation.charging_events

        # 更新充电事件中的参数
        for ev_data in charging_events:
            # 从充电事件中获取 EV 型号,唯一识别.
            ev_type = ev_data.vehicle_type.to_dict()
            brand = ev_type['brand']
            model = ev_type['model']

            # 在 config 中查找对应型号的最大视在功率,因为Elvis并不能满足所有需求,需往EV充电事件中添加其他参数
            for vehicle in ev_config["vehicle_types"]:
                if vehicle["brand"] == brand and vehicle["model"] == model:
                    ev_data.max_apparent_power = vehicle["battery"]["max_apparent_power"]
                    break

            # 修改目标 SoC
            ev_data.soc_target = random.normalvariate(ev_config["soc_target"], ev_config["std_soc_target"])
            ev_data.soc_target = max(min(ev_data.soc_target, 1), 0)  # 确保 SoC 在 [0, 1] 范围内

            # 根据 Previously 模式调整初始化时刻到达的 EV 的 SoC
            if ev_data.arrival_time <= start_datetime < ev_data.leaving_time:
                # 根据总停车时间和目标 SoC 计算当前 SoC
                total_parking_time = (ev_data.leaving_time - ev_data.arrival_time).total_seconds() / 3600.0  # 小时
                time_since_arrival = (start_datetime - ev_data.arrival_time).total_seconds() / 3600.0  # 小时
                ev_data.soc = ev_config["mean_soc"] + (ev_config["soc_target"] - ev_config["mean_soc"]) * (time_since_arrival / total_parking_time)
                ev_data.soc = max(min(ev_data.soc, 1), 0)  # 确保 SoC 在 [0, 1] 范围内

        return charging_events

class SingleStation(object):
    """针对单个充电站智能体的操作"""
    def __init__(self, station_id, env_properties, date_time, time_step):
        self.station_id = station_id
        self.env_properties = env_properties
        # 这是变压器,是所有station的和,暂时不用.
        self.stations_properties = env_properties["cluster_prop"]["default_ev_prop"]
        self.transformer_max_power = self.stations_properties["infrastructure"]["transformers"][0]["max_power"]  # 充电桩的最大功率
        self.station_max_power = self.stations_properties["infrastructure"]["transformers"][0]["charging_stations"][0]["max_power"]
        self.station_rated_power = self.stations_properties["infrastructure"]["transformers"][0]["charging_stations"][0]["rated_power"]
        self.is_occupied = False
        self.connected_ev = None  # 存储连接的EV
        self.time_step = time_step  # 时间步长
        self.disp_count = 0  # 初始化print计数器
        
        self.current_ev_reactive_power = 0  # 无功并不随着EV离开而置0
        self.reset_status()

    def connect_ev(self, ev_event, date_time):
        self.connected_ev = ev_event
        self.is_occupied = True
        self.set_initial_status(date_time)

    def disconnect_ev(self):
        self.connected_ev = None
        self.is_occupied = False
        self.reset_status()

    def set_initial_status(self, date_time):
        # 刚连接时初始化
        if self.connected_ev:
            self.battery_capacity = self.connected_ev.vehicle_type.battery.capacity
            self.soc_target_energy = self.connected_ev.soc_target * self.battery_capacity
            self.current_battery_energy = self.connected_ev.soc * self.battery_capacity
            self.max_ev_active_power = min(self.station_max_power, self.connected_ev.vehicle_type.battery.max_charge_power)
            self.park_time = int((self.connected_ev.leaving_time - date_time).total_seconds())
            self.remaining_departure_time = int((self.connected_ev.leaving_time - date_time).total_seconds())
            self.remaining_controllable_time = self.remaining_departure_time-max((self.soc_target_energy-self.current_battery_energy)/self.max_ev_active_power, 0)*3600
            self.max_schedulable_reactive_power = np.sqrt(self.station_max_power**2 - self.current_ev_active_power**2)
        # 具体充放动作不在这

    def reset_status(self):
        # 初始化充电桩的状态信息1-8
        self.battery_capacity = 0  # 电池总容量
        self.soc_target_energy = 0  # 目标SoC能量
        self.current_battery_energy = 0  # 当前电池电量
        self.max_ev_active_power = 0  # 最大充放电功率
        self.park_time = 0
        self.remaining_departure_time = 0  # 剩余离开时间
        self.remaining_controllable_time = 0  # 剩余可控时间, 当电量不足必须以最大功率充电时将不可控
        self.current_ev_active_power = 0  # 当前充放电功率
        self.max_schedulable_reactive_power = np.sqrt(self.station_max_power**2 - self.current_ev_active_power**2)  # 去掉有功后可调度的无功功率最大值
        # self.current_ev_reactive_power = 0  # 当前无功功率并不应该为0
        # 并不在这添加共有统计的状态9-14
        # self.system_active_power_target = 0  # 系统给定的有功目标值
        # self.hvac_total_active_power = 0      # HVAC总有功功率
        # self.system_reactive_power_target = 0 # 系统给定的无功目标值
        # self.ev_queue_count = 0               # EV队列等待个数
        # self.ev_total_active_power = 0        # EV总有功功率
        # self.ev_total_reactive_power = 0      # EV总无功功率

    def control_reactive_power(self, active_power_action, reactive_power_action, date_time, reactive_power_need_added):
        duration = self.time_step.total_seconds() / 3600  # 时间步长转换为小时
        # 计算当前动作执行后的电池能量变化
        p_power = 0
        max_q_power = np.sqrt(self.station_max_power**2 - p_power**2)

        if reactive_power_need_added is False:  # 不需要自适应无功,使用RL控制
            q_power = min(abs(reactive_power_action * self.station_max_power), max_q_power) * (1 if reactive_power_action >= 0 else -1)
        else:
            # 如果需要提供无功补偿
            if abs(reactive_power_need_added) <= max_q_power:
                q_power = reactive_power_need_added  # 如果能满足需求，直接使用需要的无功功率
                self.remaining_reactive_power = 0  # 计算还需要补偿的无功功率
            else:
                q_power = max_q_power * (1 if reactive_power_need_added >= 0 else -1)  # 使用最大无功功率，保持方向
                self.remaining_reactive_power = (abs(reactive_power_need_added) - max_q_power) * (1 if reactive_power_need_added >= 0 else -1)
        if q_power < -0.1 :
            print("q_power_", q_power)
        # 更新当前的有功和无功功率
        self.current_ev_reactive_power = q_power

    def control_active_reactive_power(self, active_power_action, reactive_power_action, date_time, reactive_power_need_added):
        duration = self.time_step.total_seconds() / 3600  # 时间步长转换为小时
        if self.is_occupied and self.connected_ev:
            # 计算当前动作执行后的电池能量变化 watt
            p_power_action = min(abs(active_power_action * self.max_ev_active_power), self.max_ev_active_power) * (1 if active_power_action >= 0 else -1)

            energy_change = p_power_action * duration * (self.connected_ev.vehicle_type.battery.efficiency if p_power_action >= 0 else 1 / self.connected_ev.vehicle_type.battery.efficiency)
            predicted_battery_energy = self.current_battery_energy + energy_change
            
            # 选择充电方式:精确充电到目标soc、不额外控制soc、允许在目标soc向下浮动百分之几
            charging_mode = self.env_properties["cluster_prop"]["charging_mode"]
            if charging_mode == "accurate":
                # 检查当前步骤执行后的电池电量是否足以使用最大功率充满
                energy_needed_to_full_charge = self.soc_target_energy - predicted_battery_energy
                time_needed_to_full_charge = energy_needed_to_full_charge / self.max_ev_active_power / (self.connected_ev.vehicle_type.battery.efficiency if energy_needed_to_full_charge >= 0 else 1 / self.connected_ev.vehicle_type.battery.efficiency) * 3600
                # 如果不足以充满电池，则全力充电
                if time_needed_to_full_charge > self.remaining_departure_time:
                    p_power_action = self.max_ev_active_power

            elif charging_mode == "uncontrol":
                pass
            elif isinstance(charging_mode, str) and charging_mode.endswith('%'):
                # 允许在目标SOC上下浮动百分之几
                soc_variation = float(charging_mode.strip('%')) / 100.0
                # 计算能量浮动范围内的充电时间
                additional_energy_needed = self.battery_capacity * soc_variation
                additional_time_needed = additional_energy_needed / self.max_ev_active_power / self.connected_ev.vehicle_type.battery.efficiency * 3600
                # 检查当前步骤执行后的电池电量是否足以使用最大功率充满
                energy_needed_to_full_charge = self.soc_target_energy - predicted_battery_energy
                time_needed_to_full_charge = energy_needed_to_full_charge / self.max_ev_active_power / (self.connected_ev.vehicle_type.battery.efficiency if energy_needed_to_full_charge >= 0 else 1 / self.connected_ev.vehicle_type.battery.efficiency) * 3600
                # 总时间包括浮动范围内的充电时间
                adjusted_time_needed = max(time_needed_to_full_charge - additional_time_needed, 0)
                if adjusted_time_needed > self.remaining_departure_time:
                    # 如果预测充不满，全力充电到(目标SOC-浮动百分)以上
                    p_power_action = self.max_ev_active_power

            energy_change = p_power_action * duration * (self.connected_ev.vehicle_type.battery.efficiency if p_power_action >= 0 else 1 / self.connected_ev.vehicle_type.battery.efficiency)
            predicted_battery_energy = self.current_battery_energy + energy_change

            if predicted_battery_energy > self.battery_capacity:
                # 如果使用最大充电功率会导致过充，则调整为所需的最大值以避免过充。并防止充电功率超标, 如某电车停车时间极短
                p_power_action = min((self.battery_capacity - self.current_battery_energy) / (duration * self.connected_ev.vehicle_type.battery.efficiency), self.max_ev_active_power)
            if predicted_battery_energy < 0:
                # 防过放, 同时防止计算出的放电功率超标
                p_power_action = max(-self.current_battery_energy / (duration / self.connected_ev.vehicle_type.battery.efficiency), -self.max_ev_active_power)


            # 确定最终的有功和无功功率，同时考虑功率因数限制。确保不超过变压器容量。
            p_power = p_power_action
            max_q_power = np.sqrt(self.station_max_power**2 - p_power**2)
            if reactive_power_need_added is False:  # 不需要自适应无功,使用RL控制  注意不能用==False判断, 因为False、0、[]（空列表）、{}（空字典）、""（空字符串）和 None 都被认为在布尔上下文中为 False
                q_power = min(abs(reactive_power_action * self.station_max_power), max_q_power) * (1 if reactive_power_action >= 0 else -1)
            else:
                # 如果需要提供无功补偿
                if abs(reactive_power_need_added) <= max_q_power:
                    q_power = reactive_power_need_added  # 如果能满足需求，直接使用需要的无功功率
                    self.remaining_reactive_power = 0  # 计算还需要补偿的无功功率
                else:
                    q_power = max_q_power * (1 if reactive_power_need_added >= 0 else -1)  # 使用最大无功功率，保持方向
                    self.remaining_reactive_power = (abs(reactive_power_need_added) - max_q_power) * (1 if reactive_power_need_added >= 0 else -1)

            # 更新电池SOC和当前电池能量，确保不超出电池容量范围（考虑效率）
            self.current_battery_energy += p_power * duration * (self.connected_ev.vehicle_type.battery.efficiency if p_power >= 0 else 1 / self.connected_ev.vehicle_type.battery.efficiency)
            if self.current_battery_energy > 100000.1 or self.current_battery_energy < -0.1:
                print("current_battery_energy:", self.current_battery_energy)
            
            self.current_battery_energy = min(max(self.current_battery_energy, 0), self.battery_capacity)
            self.connected_ev.soc = self.current_battery_energy / self.battery_capacity

            # 更新当前的有功和无功功率
            self.current_ev_active_power = p_power
            if q_power < -0.1 :
                print("q_power:",q_power)
            self.current_ev_reactive_power = q_power


    def update_status(self, date_time):
        if self.is_occupied and self.connected_ev:
            if date_time >= self.connected_ev.leaving_time:
                self.disconnect_ev()
            else:
                self.remaining_departure_time = int((self.connected_ev.leaving_time - date_time).total_seconds())
                self.remaining_controllable_time = self.remaining_departure_time-max((self.soc_target_energy-self.current_battery_energy)/self.max_ev_active_power, 0)*3600
                # 更新最大无功功率
                self.max_schedulable_reactive_power = np.sqrt(self.station_max_power**2 - self.current_ev_active_power**2)
        else:
            self.disconnect_ev()

    def step(self, active_power_action, reactive_power_action, date_time, reactive_power_need_added):
        """
        根据给定的动作更新充电桩的状态，包括有功和无功功率的控制。
        """
        if self.is_occupied and self.connected_ev:
            # 根据动作执行充电或放电操作
            self.control_active_reactive_power(active_power_action, reactive_power_action, date_time, reactive_power_need_added)
        else:  # 没EV无功也需要被控制
            self.control_reactive_power(active_power_action, reactive_power_action, date_time, reactive_power_need_added)
        # 更新充电桩的状态
        self.update_status(date_time)
        # 更新打印计数器并在达到阈值时打印状态信息
        self.disp_count += 1
        if self.disp_count >= 10000:  # 10000
            print(
                "ID: {} -- {}, Diff SOC: {:.2f}, Current P: {:.2f}, Current Q: {:.2f}, Control Δt%: {:.2f}, Departure Δt%: {:.2f}, Date: {}".format(
                    self.station_id,
                    self.connected_ev.id if self.connected_ev else None,
                    self.connected_ev.soc-self.connected_ev.soc_target if self.connected_ev else 0,
                    self.current_ev_active_power,
                    self.current_ev_reactive_power,
                    self.remaining_controllable_time /self.park_time if self.park_time > 0 else 0,
                    self.remaining_departure_time /self.park_time  if self.park_time > 0 else 0,
                    date_time,
                )
            )
            self.disp_count = 0

    def message(self, message_properties, empty=False):
        """
        这可以写全一点,后面归一化才决定哪个用及标为-1. 用于房屋向其他代理发送消息，包括当前温度差异、HVAC状态和热力学属性等信息
        Message sent by the house to other agents
        """
        if not empty and self.is_occupied:
            message = {
                "agent_type": "EV",
                "battery_capacity": self.battery_capacity,  # 布尔值区分是否有EV
                "soc_diff_energy": self.soc_target_energy - self.current_battery_energy,
                "soc_target_energy": self.soc_target_energy,
                "current_battery_energy": self.current_battery_energy,
                "max_ev_active_power": self.max_ev_active_power,
                "remaining_departure_time": self.remaining_departure_time,
                "remaining_controllable_time":self.remaining_controllable_time,
                "max_schedulable_reactive_power": self.max_schedulable_reactive_power,
                "current_ev_active_power": self.current_ev_active_power,
                "current_ev_reactive_power": self.current_ev_reactive_power
            }
        else:
            message = {
                "agent_type": "EV",
                "battery_capacity": 0,
                "soc_diff_energy": 0,
                "soc_target_energy": 0,
                "current_battery_energy": 0,
                "max_ev_active_power": 0,
                "remaining_departure_time": 0,
                "remaining_controllable_time":0,
                "max_schedulable_reactive_power": self.max_schedulable_reactive_power,
                "current_ev_active_power": 0,
                "current_ev_reactive_power": self.current_ev_reactive_power
            }
        return message

class HVAC(object):
    """
    这个类用于模拟HVAC系统的行为，包括其能量效率、制冷能力、开关控制等方面的特性。在模拟需求响应环境中，HVAC对象可以被用来模拟房屋内的空调系统，以便进一步评估需求响应策略的性能。
    
    id: HVAC对象的唯一标识符。
    hvac_properties: 包含HVAC配置属性的字典。
    COP: 系数性能（Coefficient of Performance），表示制冷容量与电力消耗之间的比率。
    cooling_capacity: 制冷容量，表示HVAC产生的负热传递速率，以瓦特（Watts）为单位。
    latent_cooling_fraction: 隐性制冷分数，介于0和1之间，表示感知制冷（温度）中的潜在制冷（湿度）的比例。
    lockout_duration: 锁定时长，表示HVAC在关闭后在再次打开之前的硬件约束时间，以秒（seconds）为单位。
    turned_on: 表示HVAC当前是否打开（True）或关闭（False）的布尔值。
    seconds_since_off: 距离HVAC上次关闭的秒数。
    time_step: 表示模拟中的时间步长的timedelta对象。
    
    Simulator of HVAC object (air conditioner)

    Attributes:

    id: string, unique identifier of the HVAC object.
    hvac_properties: dictionary, containing the configuration properties of the HVAC.
    COP: float, coefficient of performance (ratio between cooling capacity and electric power consumption)
    cooling_capacity: float, rate of "negative" heat transfer produced by the HVAC, in Watts
    latent_cooling_fraction: float between 0 and 1, fraction of sensible cooling (temperature) which is latent cooling (humidity)
    lockout_duration: int, duration of lockout (hardware constraint preventing to turn on the HVAC for some time after turning off), in seconds
    turned_on: bool, if the HVAC is currently ON (True) or OFF (False)
    seconds_since_off: int, number of seconds since the HVAC was last turned off
    time_step: a timedelta object, representing the time step for the simulation.


    Main functions:

    step(self, command): take a step in time for this TCL, given action of TCL agent
    get_Q(self): compute the rate of heat transfer produced by the HVAC
    power_consumption(self): compute the electric power consumption of the HVAC
    """

    def __init__(self, hvac_properties, time_step):
        """
        初始化HVAC对象,按个性化差异进行配置,根据传入的HVAC配置属性(最大功耗max_consumption)、锁定时长(是否有差异)、时间步长,检查错误.
        
        Initialize the HVAC

        Parameters:
        house_properties: dictionary, containing the configuration properties of the HVAC
        time_step: timedelta, time step of the simulation
        """
        self.id = hvac_properties["id"]
        self.hvac_properties = hvac_properties
        self.COP = hvac_properties["COP"]
        self.cooling_capacity = hvac_properties["cooling_capacity"]
        self.latent_cooling_fraction = hvac_properties["latent_cooling_fraction"]
        self.lockout_duration = hvac_properties["lockout_duration"] + random.randint(-hvac_properties["lockout_noise"],hvac_properties["lockout_noise"])
       
        self.turned_on = False
        self.lockout = False
        self.seconds_since_off = self.lockout_duration
        self.time_step = time_step
        self.max_consumption = self.cooling_capacity / self.COP

        if self.latent_cooling_fraction > 1 or self.latent_cooling_fraction < 0:
            raise ValueError(
                "HVAC id: {} - Latent cooling fraction must be between 0 and 1. Current value: {}.".format(
                    self.id, self.latent_cooling_fraction
                )
            )
        if self.lockout_duration < 0:
            raise ValueError(
                "HVAC id: {} - Lockout duration must be positive. Current value: {}.".format(
                    self.id, self.lockout_duration
                )
            )
        if self.cooling_capacity < 0:
            raise ValueError(
                "HVAC id: {} - Cooling capacity must be positive. Current value: {}.".format(
                    self.id, self.cooling_capacity
                )
            )
        if self.COP < 0:
            raise ValueError(
                "HVAC id: {} - Coefficient of performance (COP) must be positive. Current value: {}.".format(
                    self.id, self.COP
                )
            )

    def step(self, command):
        """
        用于在模拟中推进HVAC对象的状态，根据TCL代理的行动（command）来控制HVAC的打开或关闭。
        在一定的锁定时长内，HVAC无法立即重新打开，以模拟硬件约束。
        
        Take a step in time for this TCL, given action of the TCL agent.

        Return:
        -

        Parameters:
        self
        command: bool, action of the TCL agent (True: ON, False: OFF)
        """

        if self.turned_on == False:
            self.seconds_since_off += self.time_step.seconds

        if self.turned_on or self.seconds_since_off >= self.lockout_duration:
            self.lockout = False
        else:
            self.lockout = True

        if self.lockout:
            self.turned_on = False
        else:
            self.turned_on = command  # Efan's 此处可能有问题,因为数组有值则为True, 即使值为0. 经过测试:array([1])为True, array([0])为False, 即不需要使用command[0]
            if self.turned_on:
                self.seconds_since_off = 0
            elif (
                self.seconds_since_off + self.time_step.seconds < self.lockout_duration
            ):
                self.lockout = True

    def get_Q(self):
        """
        计算HVAC产生的热传递速率（热功率），以瓦特（Watts）为单位。
        如果HVAC打开，则热传递速率为负制冷容量。
        
        Compute the rate of heat transfer produced by the HVAC

        Return:
        q_hvac: float, heat of transfer produced by the HVAC, in Watts

        Parameters:
        self
        """
        if self.turned_on:
            q_hvac = -1 * self.cooling_capacity / (1 + self.latent_cooling_fraction)
        else:
            q_hvac = 0

        return q_hvac

    def power_consumption(self):
        """
        计算HVAC的电力消耗，以瓦特（Watts）为单位。
        如果HVAC打开，则电力消耗等于最大电力消耗，否则为0。
        
        Compute the electric power consumption of the HVAC

        Return:
        power_cons: float, electric power consumption of the HVAC, in Watts
        """
        if self.turned_on:
            power_cons = self.max_consumption
        else:
            power_cons = 0

        return power_cons

class SingleHouse(object):
    """
    这个类用于模拟单个房屋的温度变化，包括室内空气温度和质量温度，以及HVAC系统的行为。在需求响应环境中，多个单个房屋对象可以被创建和模拟，以评估不同的HVAC策略对室内温度的影响。

    house_properties：包含单个房屋配置属性的字典。
    id：房屋的唯一标识符。
    init_air_temp：房屋初始室内空气温度，以摄氏度为单位。
    current_temp：房屋当前室内空气温度，以摄氏度为单位。
    init_mass_temp：房屋初始室内质量温度，以摄氏度为单位。
    current_mass_temp：房屋当前质量温度，以摄氏度为单位。
    window_area：窗户总面积，以平方米为单位。
    shading_coeff：窗户太阳热增益系数（通过窗户传递的太阳增益比例），介于0和1之间。
    solar_gain_bool：是否考虑太阳热增益的布尔值。
    current_solar_gain：当前太阳热增益，以瓦特（Watts）为单位。
    target_temp：房屋目标室内空气温度，以摄氏度为单位。
    deadband：室内空气温度差异的容忍度，以摄氏度为单位。
    Ua：房屋传导率，以瓦特/开尔文（Watts/Kelvin）为单位。
    Ca：空气热质量，以焦耳/开尔文（Joules/Kelvin）或瓦特/开尔文/秒（Watts/Kelvin.second）为单位。
    Hm：房屋质量表面传导率，以瓦特/开尔文（Watts/Kelvin）为单位。
    Cm：房屋热质量，以焦耳/开尔文（Joules/Kelvin）或瓦特/开尔文/秒（Watts/Kelvin.second）为单位。
    hvac_properties：包含房屋HVAC属性的字典。
    hvac：房屋的HVAC对象，用于模拟HVAC系统的行为。
    disp_count：用于计算打印计数的迭代器。

    Single house simulator.
    **Attention** Although the infrastructure could support more, each house can currently only have one HVAC (several HVAC/house not implemented yet)

    Attributes:
    house_properties: dictionary, containing the configuration properties of the SingleHouse object
    id: string, unique identifier of he house.
    init_air_temp: float, initial indoors air temperature of the house, in Celsius
    init_mass_temp: float, initial indoors mass temperature of the house, in Celsius
    current_temp: float, current indoors air temperature of the house, in Celsius
    current_mass_temp: float, current house mass temperature, in Celsius
    window_area: float, gross window area, in m^2
    shading_coeff: float between 0 and 1, window solar heat gain coefficient (ratio of solar gain passing through the windows)
    target_temp: float, target indoors air temperature of the house, in Celsius
    deadband: float, margin of tolerance for indoors air temperature difference, in Celsius.
    Ua: float, House conductance in Watts/Kelvin
    Ca: float, Air thermal mass, in Joules/Kelvin (or Watts/Kelvin.second)
    Hm: float, House mass surface conductance, in Watts/Kelvin
    Cm: float, House thermal mass, in Joules/Kelvin (or Watts/Kelvin.second)
    hvac_properties: dictionary, containing the properties of the houses' hvacs
    hvac: hvac object for the house
    disp_count: int, iterator for printing count

    Functions:
    step(self, od_temp, time_step): Take a time step for the house
    update_temperature(self, od_temp, time_step): Compute the new temperatures depending on the state of the house's HVACs
    """

    def __init__(self, house_properties, time_step):
        """
        初始化房屋对象，根据传入的房屋配置属性和时间步长。
        Initialize the house

        Parameters:
        house_properties: dictionary, containing the configuration properties of the SingleHouse
        time_step: timedelta, time step of the simulation
        """

        self.house_properties = house_properties
        self.id = house_properties["id"]
        self.init_air_temp = house_properties["init_air_temp"]
        self.current_temp = self.init_air_temp
        self.init_mass_temp = house_properties["init_mass_temp"]
        self.current_mass_temp = self.init_mass_temp
        self.window_area = house_properties["window_area"]
        self.shading_coeff = house_properties["shading_coeff"]
        self.solar_gain_bool = house_properties["solar_gain_bool"]
        self.current_solar_gain = 0


        # Thermal constraints
        self.target_temp = house_properties["target_temp"]
        self.deadband = house_properties["deadband"]

        # Thermodynamic properties
        self.Ua = house_properties["Ua"]
        self.Ca = house_properties["Ca"]
        self.Hm = house_properties["Hm"]
        self.Cm = house_properties["Cm"]

        # HVACs
        # self.hvac_properties 提供了初始化 HVAC 对象所需的静态配置信息，仅仅是一个字典,包含了配置 HVAC 对象所需的参数，如 COP（性能系数）、cooling_capacity（制冷容量）等。
        # 而 self.hvac 是一个功能性对象，能够根据这些配置信息以及实时的模拟数据来动态地模拟 HVAC 系统的行为。还包含了 HVAC 行为的实现，例如能够计算其产生的热传递率、电力消耗等。还包含了与 HVAC 状态相关的附加属性，如 turned_on、lockout 和 seconds_since_off 等。
        self.hvac_properties = house_properties["hvac_properties"]
        self.hvac = HVAC(self.hvac_properties, time_step)

        self.disp_count = 0

    def step(self, od_temp, time_step, date_time):
        """
        用于在模拟中推进房屋对象的状态，根据室外温度和时间步长来更新房屋温度。
        该方法还包括打印房屋状态信息的功能。
        Take a time step for the house

        Return: -

        Parameters:
        self
        od_temp: float, current outdoors temperature in Celsius
        time_step: timedelta, time step duration
        date_time: datetime, current date and time
        """

        self.update_temperature(od_temp, time_step, date_time)

        # Printing
        self.disp_count += 1
        if self.disp_count >= 10000:  # Efan's 需要修改10000
            print(
                "House ID: {} -- OD_temp : {:f}, ID_temp: {:f}, target_temp: {:f}, diff: {:f}, HVAC on: {}, HVAC lockdown: {}, date: {}".format(
                    self.id,
                    od_temp,
                    self.current_temp,
                    self.target_temp,
                    self.current_temp - self.target_temp,
                    self.hvac.turned_on,
                    self.hvac.seconds_since_off,
                    date_time,
                )
            )
            self.disp_count = 0

    def message(self, message_properties, empty=False):
        """
        用于房屋向其他代理发送消息，包括当前温度差异、HVAC状态和热力学属性等信息
        Message sent by the house to other agents
        """
        if not empty:
            message = {
                "agent_type": "HVAC",
                "current_temp_diff_to_target": self.current_temp - self.target_temp,
                "hvac_seconds_since_off": self.hvac.seconds_since_off,
                "hvac_curr_consumption": self.hvac.power_consumption(),
                "hvac_max_consumption": self.hvac.max_consumption,
                "hvac_lockout_duration": self.hvac.lockout_duration
            }
            if message_properties["thermal"]:
                message["house_Ua"] = self.Ua
                message["house_Cm"] = self.Cm
                message["house_Ca"] = self.Ca
                message["house_Hm"] = self.Hm
            if message_properties["hvac"]:
                message["hvac_COP"] = self.hvac.COP
                message["hvac_cooling_capacity"] = self.hvac.cooling_capacity
                message["hvac_latent_cooling_fraction"] = self.hvac.latent_cooling_fraction
        else:
            message = {
                "agent_type": "HVAC",
                "current_temp_diff_to_target": 0,
                "hvac_seconds_since_off": 0,
                "hvac_curr_consumption": 0,
                "hvac_max_consumption": 0,
                "hvac_lockout_duration": 0
            }
            if message_properties["thermal"]:
                message["house_Ua"] = 0
                message["house_Cm"] = 0
                message["house_Ca"] = 0
                message["house_Hm"] = 0
            if message_properties["hvac"]:
                message["hvac_COP"] = 0
                message["hvac_cooling_capacity"] = 0
                message["hvac_latent_cooling_fraction"] = 0
        return message

    def update_temperature(self, od_temp, time_step, date_time):
        """
        用于更新房屋的温度，根据室外温度、时间步长和热力学参数计算新的温度。
        Update the temperature of the house

        Return: -

        Parameters:
        self
        od_temp: float, current outdoors temperature in Celsius
        time_step: timedelta, time step duration
        date_time: datetime, current date and time


        ---
        Model taken from http://gridlab-d.shoutwiki.com/wiki/Residential_module_user's_guide
        """

        time_step_sec = time_step.seconds
        Hm, Ca, Ua, Cm = self.Hm, self.Ca, self.Ua, self.Cm

        # Convert Celsius temperatures in Kelvin
        od_temp_K = od_temp + 273
        current_temp_K = self.current_temp + 273
        current_mass_temp_K = self.current_mass_temp + 273

        # Heat from hvacs (negative if it is AC)
        total_Qhvac = self.hvac.get_Q()

        # Total heat addition to air
        if self.solar_gain_bool:
            self.current_solar_gain = house_solar_gain(date_time, self.window_area, self.shading_coeff)
        else:
            self.current_solar_gain = 0

        other_Qa = self.current_solar_gain # windows, ...
        Qa = total_Qhvac + other_Qa
        # Heat from inside devices (oven, windows, etc)
        Qm = 0

        # Variables and time constants
        a = Cm * Ca / Hm
        b = Cm * (Ua + Hm) / Hm + Ca
        c = Ua
        d = Qm + Qa + Ua * od_temp_K
        g = Qm / Hm

        r1 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        r2 = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)

        dTA0dt = (
            Hm * current_mass_temp_K / Ca
            - (Ua + Hm) * current_temp_K / Ca
            + Ua * od_temp_K / Ca
            + Qa / Ca
        )

        A1 = (r2 * current_temp_K - dTA0dt - r2 * d / c) / (r2 - r1)
        A2 = current_temp_K - d / c - A1
        A3 = r1 * Ca / Hm + (Ua + Hm) / Hm
        A4 = r2 * Ca / Hm + (Ua + Hm) / Hm

        # Updating the temperature
        old_temp_K = current_temp_K
        new_current_temp_K = (
            A1 * np.exp(r1 * time_step_sec) + A2 * np.exp(r2 * time_step_sec) + d / c
        )
        new_current_mass_temp_K = (
            A1 * A3 * np.exp(r1 * time_step_sec)
            + A2 * A4 * np.exp(r2 * time_step_sec)
            + g
            + d / c
        )

        self.current_temp = new_current_temp_K - 273
        self.current_mass_temp = new_current_mass_temp_K - 273


class ClusterAgents(object):
    """
    包含多个房屋和多个充电桩(在这实例化)的集群，这些房屋具有相同的室外温度。集群行为，包括控制HVAC系统、计算室外温度和生成代理的观察信息。
    在需求响应环境中，多个集群可以用于研究不同的HVAC策略和代理通信模式对集群行为和电力消耗的影响。
    cluster_prop：包含集群配置属性的字典。
    houses：包含集群中所有房屋的字典。
    hvacs_id_registry：将每个HVAC与其所属房屋相对应的字典。
    day_temp：白天的最高温度，以摄氏度为单位。
    night_temp：夜晚的最低温度，以摄氏度为单位。
    temp_std：温度标准差，以摄氏度为单位。
    current_OD_temp：当前的室外温度，以摄氏度为单位。
    cluster_hvac_active_power：所有集群HVAC的当前累积电功率消耗，以瓦特（Watts）为单位。

    A cluster contains several houses, with the same outdoors temperature.

    Attributes:
    cluster_prop: dictionary, containing the configuration properties of the cluster
    houses: dictionary, containing all the houses in the Cluster
    hvacs_id_registry: dictionary, mapping each HVAC to its house
    day_temp: float, maximal temperature during the day, in Celsius
    night_temp: float, minimal temperature during the night, in Celsius
    temp_std: float, standard deviation of the temperature, in Celsius
    current_OD_temp: float, current outdoors temperature, in Celsius
    cluster_hvac_active_power: float, current cumulative electric power consumption of all cluster HVACs, in Watts

    Functions:
    make_cluster_obs_dict(self, date_time): generate the cluster observation dictionary for all agents
    step(self, date_time, actions_dict, time_step): take a step in time for all the houses in the cluster
    compute_OD_temp(self, date_time): models the outdoors temperature
    """

    def __init__(self, env_properties, hvac_agent_ids, stations_agent_ids, date_time, time_step, default_env_properties):

        """
        初始化集群对象，根据传入的集群配置属性、代理标识、日期时间、时间步长和默认环境属性。
        创建集群中的所有房屋，并初始化它们。
        设置集群的温度模式、温度参数和初始室外温度。
        计算初始集群HVAC功率消耗。
        
        Initialize the cluster of houses

        Parameters:
        cluster_prop: dictionary, containing the configuration properties of the cluster
        date_time: datetime, initial date and time
        time_step: timedelta, time step of the simulation
        """
        self.cluster_prop = env_properties["cluster_prop"]
        self.hvac_agent_ids = hvac_agent_ids
        self.hvac_nb_agents = len(hvac_agent_ids)
        # print("nb agents: {}".format(self.hvac_nb_agents))

        self.stations_agent_ids = stations_agent_ids
        self.station_nb_agents = len(stations_agent_ids)

        self.all_agent_ids = self.hvac_agent_ids + self.stations_agent_ids
        self.all_nb_agents = len(self.all_agent_ids)

        
        # 创建充电桩智能体管理器,这里逻辑不像HVAC和SingleHouse, 只需要生成一次charging_events,所以不必在SingleStation中每次都调用EV大类生成EV 
        # 先SingleStation初始化多个充电桩,再安排充电的EV和状态信息存在self.stations中
        self.stations = {}  # 目前是重复一样的, 没有差异化, 如charging_station0~4
        for station_agent_id in env_properties["cluster_prop"]["stations_agent_ids"]:
            station = SingleStation(station_agent_id, env_properties, date_time, time_step)
            self.stations[station.station_id] = station

        self.stations_manager = StationsManager(self.stations, env_properties["cluster_prop"]["default_ev_prop"], time_step)
        # Efan 生成 EV 充电事件
        self.ev_charging_events = self.stations_manager.generate_ev_charging_events(env_properties)
        self.stations_manager.update_stations(self.ev_charging_events, date_time)  # env_properties["start_datetime"]  env_properties["cluster_prop"]["default_ev_prop"]["start_date"]

        # 打印正在充电的 EV 信息
        for station_id, station in self.stations_manager.stations.items():
            if station.is_occupied:
                print(f"Station {station_id} has EV with ID: {station.connected_ev.id}")
        self.reactive_power_need_added = 0  # 若无功不由RL控制,则使用该参数,逐个station添加无功直到满足条件. 由PowerGrid逐步更新

        # Houses
        self.houses = {}
        for house_properties in env_properties["cluster_prop"]["houses_properties"]:
            house = SingleHouse(house_properties, time_step)
            self.houses[house.id] = house

        self.temp_mode = env_properties["cluster_prop"]["temp_mode"]
        self.temp_params = env_properties["cluster_prop"]["temp_parameters"][self.temp_mode]
        self.day_temp = self.temp_params["day_temp"]
        self.night_temp = self.temp_params["night_temp"]
        self.temp_std = self.temp_params["temp_std"]
        self.random_phase_offset = self.temp_params["random_phase_offset"]
        self.env_prop = default_env_properties
        # 温度偏移量（phase），它将影响随后的周期函数（如正弦波或余弦波）的计算
        if self.random_phase_offset:
            self.phase = random.random() * 24
        else:
            self.phase = 0
        self.current_OD_temp = self.compute_OD_temp(date_time)

        # Compute the Initial cluster_hvac_active_power
        self.cluster_hvac_active_power = 0
        self.max_hvac_power = 0
        for house_id in self.houses.keys():
            house = self.houses[house_id]
            hvac = house.hvac
            self.cluster_hvac_active_power += hvac.power_consumption()
            self.max_hvac_power += hvac.max_consumption

        self.build_agent_comm_links()

    def build_agent_comm_links(self):
        """
        建立代理之间的通信链接，根据集群配置中的通信模式和参数。
        邻居（neighbours）：此模式下，每个智能体的邻居是固定的，基于它们的ID或位置。例如，如果设置为与周围的邻居通信，那么每个智能体的邻居在整个过程中都是固定的。
        封闭组（closed_groups）：在这个模式下，智能体被分成固定的小组，并只与同组内的其他智能体通信。
        随机样本（random_sample）：在这种模式下，智能体在每个时间步随机选择一组新的邻居进行通信。这意味着通信的对象在每个步骤中都可能发生变化。
        随机固定（random_fixed）：智能体的邻居是随机选择的，但一旦确定，就在整个模拟过程中保持不变。
        邻居2D（neighbours_2D）：这个模式类似于标准的“邻居”模式，但适用于二维空间布局的智能体。这意味着智能体的邻居是基于二维空间位置确定的。
        无消息（no_message）：在这种模式下，智能体之间不进行任何通信。
        """
        self.agent_communicators = {}
        # 根据智能体类型（HVAC或EV）和数量来确定每个智能体的可通信最大数量
        # 如果两种类型的智能体都存在
        if self.cluster_prop["hvac_nb_agents"] > 0 and self.cluster_prop["station_nb_agents"] > 0:
            # 检查HVAC和EV智能体的可通信最大数量是否一致
            if self.cluster_prop["hvac_nb_agents_comm"] == self.cluster_prop["station_nb_agents_comm"]:
                nb_comm = self.cluster_prop["hvac_nb_agents_comm"]
            else:
                # 如果不一致，取两者中较小的值，并报错提示
                nb_comm = min(self.cluster_prop["hvac_nb_agents_comm"], self.cluster_prop["station_nb_agents_comm"])
                raise ValueError("现阶段HVAC和EV智能体的可通信最大数量应该一致The maximum number of communicable HVAC and EV intelligences at this stage should be the same")
        elif self.cluster_prop["hvac_nb_agents"] > 0:
            # 只有HVAC智能体
            nb_comm = self.cluster_prop["hvac_nb_agents_comm"]
        elif self.cluster_prop["station_nb_agents"] > 0:
            # 只有EV智能体
            nb_comm = self.cluster_prop["station_nb_agents_comm"]
        # 确保通信数量不超过智能体总数
        nb_comm = np.minimum(nb_comm, len(self.all_agent_ids) - 1)

        # 如有需要，打乱智能体ID的顺序
        if self.cluster_prop["shuffle_ids"]:
            random.shuffle(self.all_agent_ids)

            # This is to get the neighbours of each agent in a circular fashion,
            # if this_agent_id is 5, the half before will be [0, 1, 2, 3, 4] and half after will be [6, 7, 8, 9, 10]
            # if this_agent_id is 1, the half before will be [7, 8, 9, 10, 0] and half after will be [2, 3, 4, 5, 6]
        # 适用于所有智能体的通信模式
        if self.cluster_prop["agents_comm_mode"] == "neighbours":
            for this_agent_id in self.all_agent_ids:
                possible_ids = deepcopy(self.all_agent_ids)
                possible_ids.remove(this_agent_id)
                
                index = self.all_agent_ids.index(this_agent_id)
                ids_neighbours = []

                # 获取邻居，确保邻居总数为nb_comm
                start_index = index - nb_comm // 2
                end_index = start_index + nb_comm

                for i in range(start_index, end_index):
                    neighbor_index = i % len(possible_ids)
                    if possible_ids[neighbor_index] != this_agent_id:
                        ids_neighbours.append(possible_ids[neighbor_index])

                self.agent_communicators[this_agent_id] = ids_neighbours

        elif self.cluster_prop["agents_comm_mode"] == "closed_groups":
            for this_agent_id in self.all_agent_ids:
                possible_ids = deepcopy(self.all_agent_ids)
                possible_ids.remove(this_agent_id)            

                # 注意当nb_comm=10 > 智能体数=2时,原有代码会直接创建一个:
                # Agent Communicators:
                #   0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                #   1: [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                # 封闭组模式处理
                group_size = nb_comm + 1
                agent_index = self.all_agent_ids.index(this_agent_id)
                base = agent_index // group_size * group_size
                ids_group = []

                # 从列表末尾开始计算组成员
                if base + group_size > len(self.all_agent_ids):
                    start_index = len(self.all_agent_ids) - group_size
                    ids_group = [self.all_agent_ids[i] for i in range(start_index, len(self.all_agent_ids)) if self.all_agent_ids[i] != this_agent_id]
                else:
                    # 获取组成员
                    for i in range(base, base + group_size):
                        if self.all_agent_ids[i] != this_agent_id:
                            ids_group.append(self.all_agent_ids[i])

                self.agent_communicators[this_agent_id] = ids_group

        elif self.cluster_prop["agents_comm_mode"] == "random_sample":
            # 在别的地方有处理,这里处理不了
            pass

        elif self.cluster_prop["agents_comm_mode"] == "random_fixed":
            for this_agent_id in self.all_agent_ids:
                possible_ids = deepcopy(self.all_agent_ids)
                possible_ids.remove(this_agent_id)

                # 随机固定模式处理,上面已经deepcopy了
                if len(possible_ids) > nb_comm:
                    if this_agent_id not in self.agent_communicators:
                        self.agent_communicators[this_agent_id] = random.sample(possible_ids, k=nb_comm)
                else:
                    self.agent_communicators[this_agent_id] = possible_ids

        elif self.cluster_prop["agents_comm_mode"] == "neighbours_2D":
            # 邻居2D模式处理
            row_size = self.cluster_prop["agents_comm_parameters"]["neighbours_2D"]["row_size"]
            distance_comm = self.cluster_prop["agents_comm_parameters"]["neighbours_2D"]["distance_comm"]
            if len(self.all_agent_ids) % row_size != 0:
                raise ValueError("Neighbours 2D row_size must be a divisor of total number of agents")

            max_y = len(self.all_agent_ids) // row_size
            if distance_comm >= (row_size+1) // 2 or distance_comm >= (max_y+1) // 2:
                raise ValueError("Neighbours 2D distance_comm ({}) must be strictly smaller than (row_size+1) / 2 ({}) and (max_y+1) / 2 ({})".format(distance_comm, (row_size+1) // 2, (max_y+1) // 2))

            distance_pattern = []
            for x_diff in range(-1*distance_comm, distance_comm + 1):
                for y_diff in range(-1*distance_comm, distance_comm + 1):
                    if abs(x_diff) + abs(y_diff) <= distance_comm and (x_diff != 0 or y_diff != 0):
                        distance_pattern.append((x_diff, y_diff))

            # 为每个智能体分配邻居
            for this_agent_id in self.all_agent_ids:
                index = self.all_agent_ids.index(this_agent_id)
                x = index % row_size
                y = index // row_size
                ids_neighbours = []

                for pair_diff in distance_pattern:
                    x_new = (x + pair_diff[0] + row_size) % row_size
                    y_new = (y + pair_diff[1] + max_y) % max_y
                    neighbour_index = y_new * row_size + x_new
                    if neighbour_index < len(self.all_agent_ids) and self.all_agent_ids[neighbour_index] != this_agent_id:
                        ids_neighbours.append(self.all_agent_ids[neighbour_index])

                self.agent_communicators[this_agent_id] = ids_neighbours


        elif self.cluster_prop["agents_comm_mode"] == "no_message":
            for this_agent_id in self.all_agent_ids:
                # 无消息模式处理
                self.agent_communicators[this_agent_id] = []

        else:
            # 未知通信模式的错误处理
            raise ValueError(
                "Cluster property: unknown agents_comm_mode '{}'.".format(
                    self.cluster_prop["agents_comm_mode"]
                )
            )

        # 打印或返回结果，用于验证
        print("Agent Communicators:")
        for this_agent_id, communicators in self.agent_communicators.items():
            print(f"  {this_agent_id}: {communicators}")

    def make_cluster_obs_dict(self, date_time):
        """
        生成所有代理的集群观察字典，包括动态和静态观察值以及其他代理的消息。
        用于为每个代理提供其需要的环境信息和其他代理的信息，以便代理能够根据这些信息做出决策。
        
        集群观察字典包含了每个TCL（Thermostatically Controlled Load，恒温控制负载）代理的观察结果，用于训练和决策。
            方法首先初始化一个空字典cluster_obs_dict，用于存储所有代理的观察结果。
            遍历所有房屋（代理），为每个房屋（代理）生成观察结果，并存储到字典中。
            从集群、房屋和HVAC（Heating, Ventilation, and Air Conditioning，暖通空调）中获取动态和静态的观察值。
            如果代理间通信是随机样本模式，则从其他代理中随机选择一些代理，获取它们的消息。
            将所有观察结果和消息存储到字典中，并返回这个字典。
        
        为所有代理生成集群观察字典。 
        返回:
        cluster_obs_dict: 字典，包含每个TCL代理的集群观察结果。
        参数:
        self
        date_time: datetime, 当前日期和时间
        
        Generate the cluster observation dictionary for all agents.

        Return:
        cluster_obs_dict: dictionary, containing the cluster observations for every TCL agent.

        Parameters:
        self
        date_time: datetime, current date and time
        """
        # 初始化集群观察字典
        cluster_obs_dict = {}
        for agent_id in self.all_agent_ids:
            cluster_obs_dict[agent_id] = {}

            # 根据智能体类型获取相应的对象
            if isinstance(agent_id, int):  # HVAC智能体
                # 获取房屋和HVAC对象
                # Getting the house and the HVAC
                house = self.houses[agent_id]
                hvac = house.hvac

                # 从集群中获取动态值
                # Dynamic values from cluster
                cluster_obs_dict[agent_id]["OD_temp"] = self.current_OD_temp
                cluster_obs_dict[agent_id]["datetime"] = date_time

                # Dynamic values from house
                cluster_obs_dict[agent_id]["house_temp"] = house.current_temp
                cluster_obs_dict[agent_id]["house_mass_temp"] = house.current_mass_temp

                # Dynamic values from HVAC
                cluster_obs_dict[agent_id]["hvac_turned_on"] = hvac.turned_on
                cluster_obs_dict[agent_id][
                    "hvac_seconds_since_off"
                ] = hvac.seconds_since_off
                cluster_obs_dict[agent_id]["hvac_lockout"] = hvac.lockout

                # 从房屋中获取常量值（以后可能会更改）
                # Supposedly constant values from house (may be changed later)
                cluster_obs_dict[agent_id]["house_target_temp"] = house.target_temp
                cluster_obs_dict[agent_id]["house_deadband"] = house.deadband
                cluster_obs_dict[agent_id]["house_Ua"] = house.Ua
                cluster_obs_dict[agent_id]["house_Cm"] = house.Cm
                cluster_obs_dict[agent_id]["house_Ca"] = house.Ca
                cluster_obs_dict[agent_id]["house_Hm"] = house.Hm
                cluster_obs_dict[agent_id]["house_solar_gain"] = house.current_solar_gain

                # Supposedly constant values from hvac
                cluster_obs_dict[agent_id]["hvac_COP"] = hvac.COP
                cluster_obs_dict[agent_id]["hvac_cooling_capacity"] = hvac.cooling_capacity
                cluster_obs_dict[agent_id][
                    "hvac_latent_cooling_fraction"
                ] = hvac.latent_cooling_fraction
                cluster_obs_dict[agent_id]["hvac_lockout_duration"] = hvac.lockout_duration

            elif isinstance(agent_id, str) and agent_id.startswith('charging_station'):  # 充电桩智能体
                # 获取充电桩对象
                station = self.stations[agent_id]
                cluster_obs_dict[agent_id]["datetime"] = date_time
                # 尽量写全,归一化再决定用哪个. 生成充电桩的动态和静态值
                cluster_obs_dict[agent_id]["battery_capacity"] = station.battery_capacity  # 需要一个布尔值的标志位确定是否有ev  
                cluster_obs_dict[agent_id]["soc_diff_energy"] = station.soc_target_energy - station.current_battery_energy
                cluster_obs_dict[agent_id]["soc_target_energy"] = station.soc_target_energy
                cluster_obs_dict[agent_id]["current_battery_energy"] = station.current_battery_energy
                cluster_obs_dict[agent_id]["max_ev_active_power"] = station.max_ev_active_power
                cluster_obs_dict[agent_id]["remaining_departure_time"] = station.remaining_departure_time
                cluster_obs_dict[agent_id]["remaining_controllable_time"] = station.remaining_controllable_time
                cluster_obs_dict[agent_id]["max_schedulable_reactive_power"] = station.max_schedulable_reactive_power
                cluster_obs_dict[agent_id]["current_ev_active_power"] = station.current_ev_active_power
                cluster_obs_dict[agent_id]["current_ev_reactive_power"] = station.current_ev_reactive_power


            # 处理通信消息 为啥把random_sample写到这?因为每次初始化或reset调用build_agent_comm_links才会更新交流模式,而随机采样需要每步都更新
            if self.cluster_prop["agents_comm_mode"] == "random_sample":
                # 重复一遍,计算nb_comm
                if self.cluster_prop["hvac_nb_agents"] > 0 and self.cluster_prop["station_nb_agents"] > 0:
                    # 检查HVAC和EV智能体的可通信最大数量是否一致
                    if self.cluster_prop["hvac_nb_agents_comm"] == self.cluster_prop["station_nb_agents_comm"]:
                        nb_comm = self.cluster_prop["hvac_nb_agents_comm"]
                    else:
                        # 如果不一致，取两者中较小的值，并报错提示
                        nb_comm = min(self.cluster_prop["hvac_nb_agents_comm"], self.cluster_prop["station_nb_agents_comm"])
                        raise ValueError("现阶段HVAC和EV智能体的可通信最大数量应该一致The maximum number of communicable HVAC and EV intelligences at this stage should be the same")
                elif self.cluster_prop["hvac_nb_agents"] > 0:
                    # 只有HVAC智能体
                    nb_comm = self.cluster_prop["hvac_nb_agents_comm"]
                elif self.cluster_prop["station_nb_agents"] > 0:
                    # 只有EV智能体
                    nb_comm = self.cluster_prop["station_nb_agents_comm"]
                # 确保通信数量不超过智能体总数
                nb_comm = np.minimum(nb_comm, len(self.all_agent_ids) - 1)
                
                possible_ids = deepcopy(self.all_agent_ids)
                possible_ids.remove(agent_id)
                ids_agents_messages = random.sample(possible_ids, k=nb_comm)
            else:
                ids_agents_messages = self.agent_communicators[agent_id]

            # 初始化消息列表
            cluster_obs_dict[agent_id]["message"] = []
            for id_agent_message in ids_agents_messages:
                if np.random.rand() > self.cluster_prop["comm_defect_prob"]:
                    # 判断消息来源的智能体类型并添加消息
                    if isinstance(id_agent_message, int):  # 来自HVAC智能体的消息
                        cluster_obs_dict[agent_id]["message"].append(
                            self.houses[id_agent_message].message(self.env_prop["message_properties"])
                        )
                    elif isinstance(id_agent_message, str) and id_agent_message.startswith('charging_station'):  # 来自充电桩的消息
                        cluster_obs_dict[agent_id]["message"].append(
                            self.stations[id_agent_message].message(self.env_prop["message_properties"])
                        )
                else:
                    # 添加空消息
                    cluster_obs_dict[agent_id]["message"].append(None)

        return cluster_obs_dict

    def step(self, date_time, actions_dict, time_step):
        """
        推进集群中所有房屋的状态，包括HVAC的控制。
        更新室外温度，生成集群观察字典，计算温度惩罚，并返回相关信息。
        
        Take a step in time for all the houses in the cluster

        Returns:
        cluster_obs_dict: dictionary, containing the cluster observations for every TCL agent.
        temp_penalty_dict: dictionary, containing the temperature penalty for each TCL agent
        cluster_hvac_active_power: float, total power used by the TCLs, in Watts.
        info_dict: dictonary, containing additional information for each TCL agent.

        Parameters:
        date_time: datetime, current date and time
        actions_dict: dictionary, containing the actions of each TCL agent.
        time_step: timedelta, time step of the simulation
        """

        # 用于计算总的EV有功和无功功率
        cluster_ev_active_power = 0
        cluster_ev_reactive_power = 0
        # self.reactive_power_need_added = 0

        # Send command to the hvacs
        for house_id in self.houses.keys():
            # Getting the house and the HVAC
            house = self.houses[house_id]
            hvac = house.hvac
            if house_id in actions_dict.keys():  # 2个房子{0: True, 1: True}
                command = actions_dict[house_id]
            else:
                warnings.warn(
                    "HVAC in house {} did not receive any command.".format(house_id)
                )
                command = False
            hvac.step(command)
            house.step(self.current_OD_temp, time_step, date_time)

        # 处理EV充电桩智能体的动作
        station_disconnected = False
        # 处理EV充电桩智能体的动作
        for station_id in self.stations.keys():
            station = self.stations[station_id]
            station.remaining_reactive_power = 0  # 初始化, 防止上一轮遗留值, 后面step计算会更新
            if station_id in actions_dict:
                # Efan 功无功分别控制, [0]是否应放在这
                active_power_action, reactive_power_action = actions_dict[station_id][0], actions_dict[station_id][1]
                # Efan 需检查
                if self.cluster_prop["default_ev_prop"]["reactive_control_mode"] == "RL":
                    station.step(active_power_action, reactive_power_action, date_time, reactive_power_need_added = False)
                elif self.cluster_prop["default_ev_prop"]["reactive_control_mode"] == "Adaptive":
                    station.step(active_power_action, reactive_power_action, date_time, reactive_power_need_added = self.reactive_power_need_added)
                    self.reactive_power_need_added = station.remaining_reactive_power  # 更新还需自适应补偿的无功
                if not station.is_occupied:
                    station_disconnected = True

        # 如果有充电桩中有闲置，则更新充电桩状态和EV队列
        if station_disconnected:
            self.stations_manager.update_stations(self.ev_charging_events, date_time)

        # 计算总的EV有功和无功功率
        cluster_ev_active_power = sum(station.current_ev_active_power for station in self.stations.values())
        cluster_ev_reactive_power = sum(station.current_ev_reactive_power for station in self.stations.values())
        ev_queue_count = len(self.stations_manager.pending_charging_events)


        # Update outdoors temperature
        self.current_OD_temp = self.compute_OD_temp(date_time)
        ## Observations
        cluster_obs_dict = self.make_cluster_obs_dict(date_time)

        ## Temperature penalties and total cluster power consumption
        self.cluster_hvac_active_power = 0

        for house_id in self.houses.keys():
            # Getting the house and the HVAC
            house = self.houses[house_id]
            hvac = house.hvac

            # Cluster hvac power consumption
            self.cluster_hvac_active_power += hvac.power_consumption()

        # Info
        info_dict = {}  # Not necessary for the moment

        return cluster_obs_dict, self.cluster_hvac_active_power, cluster_ev_active_power, cluster_ev_reactive_power, ev_queue_count, info_dict

    def compute_OD_temp(self, date_time) -> float:
        """
        基于时间计算室外温度的模型，可以根据集群配置中的参数生成室外温度。
        模拟室外温度的变化，包括日夜温度变化和噪声。
        
        Compute the outdoors temperature based on the time, according to a model

        Returns:
        temperature: float, outdoors temperature, in Celsius.

        Parameters:
        self
        date_time: datetime, current date and time.

        """

        # Sinusoidal model
        amplitude = (self.day_temp - self.night_temp) / 2
        bias = (self.day_temp + self.night_temp) / 2
        delay = -6 + self.phase  # Temperature is coldest at 6am
        time_day = date_time.hour + date_time.minute / 60.0

        temperature = amplitude * np.sin(2 * np.pi * (time_day + delay) / 24) + bias

        # Adding noise
        temperature += random.gauss(0, self.temp_std)

        return temperature


class PowerGrid(object):
    """
    模拟电力网络行为，用于计算输出调节信号，这个信号可以被恒温控制负载（TCL）代理用来调整其功率消耗，以响应电力需求。不同的信号模式和参数可以用于模拟不同的电力网络行为。
    Efan's 在多数情况下，由于电力系统主要是感性负载，因此通常需要用户提供感性正无功以帮助提升系统的电压或维持电压稳定。但在特定条件下，如电压过高，可能需要用户吸收感性无功或提供容性负无功。

    avg_power_per_hvac：每个HVAC设备的平均功率，以瓦特（Watts）为单位。
    signal_mode：信号变化的模式（可以是"none"或"sinusoidal"）。
    signal_params：信号变化的参数字典。
    nb_hvacs：集群中HVAC设备的数量。
    init_signal：每个HVAC的初始信号值，以瓦特（Watts）为单位。
    current_hvac_active_signal：当前的信号值，以瓦特（Watts）为单位。

    Simulated power grid outputting the regulation signal.

    Attributes:
    avg_power_per_hvac: float, average power to be given per HVAC, in Watts
    signal_mode: string, mode of variation in the signal (can be none or sinusoidal)
    signal_params: dictionary, parameters of the variation of the signal
    nb_hvac: int, number of HVACs in the cluster
    init_signal: float, initial signal value per HVAC, in Watts
    current_hvac_active_signal: float, current signal in Watts

    Functions:baharerajabi2015@gmail.combaharerajabi2015@gmail.com
    step(self, date_time): Computes the regulation signal at given date and time
    """

    def __init__(self, power_grid_prop, default_house_prop, nb_hvacs, cluster_agents=None, default_ev_prop=None):
 
        """
        Efan 待办 加入EV的每周到达数量、EV的到达型号和概率、智能体的数量(目前是充电桩),计算EV的基础功率(先算一个固定值). 
        根据传入的电力网配置属性、默认房屋属性和HVAC数量, 设置基础功率模式、初始信号值、人工干扰比例等属性。
        根据配置文件中的不同模式，初始化不同的电力基础功率。
        
        基础功率是指在电力系统或电网中，特定设备或系统在正常操作条件下所需的最低稳定功率。在这，基础功率通常是指电网为了满足所有连接设备（如HVAC系统和EV充电桩）的基本需求而必须提供的最低总功率。这个功率会根据不同设备的需求和外部条件（如天气、时间等）动态变化。在进行电力管理和优化时，准确估计和调节基础功率非常重要。
        Initialize PowerGrid.

        Returns: -

        Parameters:
        power_grid_prop: dictionary, containing the configuration properties of the power grid
        nb_hvacs: int, number of HVACs in the cluster
        """

        # Efan 计算基础功率:ideal_avg_power_per_station, 添加EV相关的初始化逻辑, 能否根据是否有EV来控制其功率?添加EV能否被控的变量.
        self.ev_base_power_mode = power_grid_prop["ev_base_power_mode"]
        if default_ev_prop["num_stations"]>0:
            self.num_stations = default_ev_prop["num_stations"]
            self.max_station_power = default_ev_prop["infrastructure"]["transformers"][0]["charging_stations"][0]["max_power"]
            self.total_max_station_power = self.num_stations * self.max_station_power  # Efan's 待改进支持变压器约束

            self.mean_park = default_ev_prop["mean_park"]
            self.num_charging_events = default_ev_prop["num_charging_events"]  # 每7天来多少辆
            self.battery_capacity = default_ev_prop["vehicle_types"][1]["battery"]["capacity"]
            self.efficiency = default_ev_prop["vehicle_types"][1]["battery"]["efficiency"]
            self.mean_soc = default_ev_prop["mean_soc"]
            self.soc_target = default_ev_prop["soc_target"]

            # EV基础功率计算思路, 未计算EV之间互传的能量需求:
            # 每辆车占用23.99h, 每辆需要的电量 = (80% - 20%) × 100,000 / 0.9 Wh = 66667 Wh, 若1个充电桩每天能为1辆车充电, 每个充电桩的平均功率 = 66667 Wh / 24 h ≈ 2,778W。
            # 若7天来14辆, 即每天来2辆, 3个充电桩, 每天每个充电桩平均充电车辆数=2/3辆, 每天每个充电桩的充电需求 = 66667 * 2/3 = 44444Wh , 每个充电桩的平均功率为 = 44444/24= 1,852W
            # 若7天来28辆, 即每天来4辆, 3个充电桩, 每天每个桩最多服务n_max=min(4/3, 24/23.99)辆车, 每天每个充电桩的充电需求66667 * n_max, 再/24h
            
            # 每辆车占用11.99, 每辆需要的电量 = (80% - 20%) × 100,000 / 0.9 Wh = 66667 Wh, 1个充电桩最多每天能为24/11.99=2辆车充电, 每个充电桩的平均功率 = 66667 Wh *2 / 24 h ≈ 2,778 * 2W。
            # 若7天来140辆 , 即每天来20辆, 5个充电桩, 每天每个充电桩平均充电车辆数=20/5=4辆, 每天每个充电桩的充电需求 = 66667Wh * min(2, 4) = 66667Wh * 2 , 每个充电桩的平均功率为 = 66667Wh * 2 / 24h
            

            # 每辆车需要的电量
            required_energy = (self.soc_target - self.mean_soc) * self.battery_capacity / self.efficiency
            # 每天每个充电桩最多服务的车辆数
            daily_events_per_station = min(self.num_charging_events / 7 / self.num_stations, 24 / self.mean_park)
            
            # 每个充电桩每天的总电量需求
            daily_energy_per_station = required_energy * daily_events_per_station  # 瓦时
            
            # 计算每个充电桩的平均功率、平均无功 待办:在这之前计算的每个充电站的功率没有考虑EV到达的约束,相同station数时event70和140给的信号一样的
            self.ideal_avg_power_per_station = daily_energy_per_station / 24  # 一天24小时 瓦
            self.max_q_power_per_station = np.sqrt(self.max_station_power**2 - self.ideal_avg_power_per_station**2)
            self.max_reactive_power = self.max_q_power_per_station * self.num_stations

        else:
            self.num_stations = 0
            self.total_max_station_power = 0
            self.ideal_avg_power_per_station = 0
            self.max_q_power_per_station = 0
            self.max_reactive_power = 0

        # Base power
        self.hvac_base_power_mode = power_grid_prop["hvac_base_power_mode"]
        self.init_signal_per_hvac = power_grid_prop["base_power_parameters"]["constant"]["init_signal_per_hvac"]
        self.active_artificial_ratio = power_grid_prop["active_artificial_ratio"] * power_grid_prop["artificial_active_signal_ratio_range"]**(random.random()*2 - 1)      # Base ratio, randomly multiplying by a number between 1/artificial_active_signal_ratio_range and artificial_active_signal_ratio_range, scaled on a logarithmic scale.
        # Efan's
        self.ev_base_power_mode = power_grid_prop["ev_base_power_mode"]
        self.ev_q_base_power_mode = power_grid_prop["ev_q_base_power_mode"]
        self.init_signal_per_ev = power_grid_prop["base_power_parameters"]["constant"]["init_signal_per_ev"]
        self.init_signal_q_per_ev = power_grid_prop["base_power_parameters"]["constant"]["init_signal_q_per_ev"]
        self.reactive_artificial_ratio = power_grid_prop["reactive_artificial_ratio"] * power_grid_prop["artificial_reactive_signal_ratio_range"]**(random.random()*2 - 1)
        self.cumulated_abs_noise = 0
        self.cumulated_abs_ev_noise = 0
        self.cumulated_abs_reactive_noise = 0
        self.nb_steps = 0

        ## Constant base power
        if self.hvac_base_power_mode == "constant":
            self.avg_power_per_hvac = power_grid_prop["base_power_parameters"][
                "constant"
            ]["avg_power_per_hvac"]
            self.init_signal_per_hvac = power_grid_prop["base_power_parameters"][
                "constant"
            ]["init_signal_per_hvac"]

        ## Interpolated base power
        elif self.hvac_base_power_mode == "interpolation":
            interp_data_path = power_grid_prop["base_power_parameters"][
                "interpolation"
            ]["path_datafile"]
            with open(
                power_grid_prop["base_power_parameters"]["interpolation"][
                    "path_parameter_dict"
                ]
            ) as json_file:
                self.interp_parameters_dict = json.load(json_file)
            with open(
                power_grid_prop["base_power_parameters"]["interpolation"][
                    "path_dict_keys"
                ]
            ) as f:
                reader = csv.reader(f)
                self.interp_dict_keys = list(reader)[0]

            self.power_interpolator = PowerInterpolator(
                interp_data_path, self.interp_parameters_dict, self.interp_dict_keys
            )

            self.interp_update_period = power_grid_prop["base_power_parameters"][
                "interpolation"
            ]["interp_update_period"]
            self.time_since_last_interp = self.interp_update_period + 1
            self.interp_hvac_nb_agents = power_grid_prop["base_power_parameters"][
                "interpolation"
            ]["interp_hvac_nb_agents"]

        ## Error

        else:
            raise ValueError(
                "The hvac_base_power_mode parameter in the config file can only be 'constant' or 'interpolation'. It is currently: {}".format(
                    self.hvac_base_power_mode
                )
            )

        if cluster_agents:
            self.cluster_agents = cluster_agents
        else:
            raise ValueError(
                "The PowerGrid object in interpolation mode needs a ClusterAgents object as a cluster_all argument."
            )

        self.max_hvac_power = power_grid_prop["max_hvac_power"]

        # Efan's ev模仿上面hvac, ev还未实现基础功率为插值
        ## Constant base power
        if self.ev_base_power_mode == "constant":
            self.avg_power_per_ev = power_grid_prop["base_power_parameters"]["constant"]["avg_power_per_ev"]
            # self.init_signal_per_ev = power_grid_prop["base_power_parameters"]["constant"]["init_signal_per_ev"]
        ## Interpolated base power
        elif self.ev_base_power_mode == "interpolation":
            raise ValueError(
                "The ev_base_power_mode parameter in the config file can only be 'constant' or 'ideal'. It is currently: {}".format(
                    self.ev_base_power_mode
                )
            )
        elif self.ev_base_power_mode == "ideal":
            pass
        else:
            raise ValueError(
                "The ev_base_power_mode parameter in the config file can only be 'constant' or 'ideal'. It is currently: {}".format(
                    self.ev_base_power_mode
                )
            )
        
        # EV无功
        if self.ev_q_base_power_mode == "constant":
            self.avg_q_power_per_ev = power_grid_prop["base_power_parameters"]["constant"]["avg_q_power_per_ev"]
            # self.init_signal_per_ev = power_grid_prop["base_power_parameters"]["constant"]["init_signal_per_ev"]
        ## Interpolated base power
        elif self.ev_q_base_power_mode == "interpolation":
            raise ValueError(
                "The ev_q_base_power_mode parameter in the config file can only be 'constant' or 'ideal'. It is currently: {}".format(
                    self.ev_q_base_power_mode
                )
            )
        elif self.ev_q_base_power_mode == "ideal":
            pass
        else:
            raise ValueError(
                "The ev_q_base_power_mode parameter in the config file can only be 'constant' or 'ideal'. It is currently: {}".format(
                    self.ev_q_base_power_mode
                )
            )

        if "perlin" in power_grid_prop["signal_mode"]:
            self.signal_params = power_grid_prop["signal_parameters"][power_grid_prop["signal_mode"]]
            nb_octaves = self.signal_params["nb_octaves"]
            octaves_step = self.signal_params["octaves_step"]
            period = self.signal_params["period"]
            self.perlin = Perlin(
                1, nb_octaves, octaves_step, period, random.random()
            )  # Random seed (will be the same given a seeded random function)

        self.signal_mode = power_grid_prop["signal_mode"]
        self.signal_params = power_grid_prop["signal_parameters"][self.signal_mode]
        self.nb_hvacs = nb_hvacs
        self.default_house_prop = default_house_prop
        self.base_power = 0
        
        self.ev_base_power = 0
        self.ev_q_base_power = 0




    def interpolatePower(self, date_time):
        """
        根据房屋的多个属性来计算基础功率的插值。
        考虑了房屋的各种属性，包括日期、时间、Ua、Cm、Ca、Hm、空气温度、质量温度、室外温度和HVAC功率等。
        """
        # 设置初始功率为零，并根据是否有太阳能增益确定要使用的点数据结构。
        base_power = 0

        if self.default_house_prop["solar_gain_bool"]:
            point = {
                "date": date_time.timetuple().tm_yday,
                "hour": (date_time - date_time.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
            }
        else:       # No solar gain - make it think it is midnight
            point = {
                "date": 0.0,
                "hour": 0.0,
            }            

        # 择房屋进行插值：从所有房屋中选择一定数量（由 self.interp_hvac_nb_agents 决定,即运行插值的最大代理数）进行插值计算。如果总房屋数量少于这个值，就包括所有房屋,否则随机取最大值self.interp_hvac_nb_agents=100个房屋。
        all_ids = list(self.cluster_agents.houses.keys())
        # 调整总功率：如果选中的房屋数量少于总数，按比例调整计算得出的基础功率。
        if len(all_ids) <= self.interp_hvac_nb_agents:
            interp_house_ids = all_ids
            multi_factor = 1
        else:
            interp_house_ids = random.choices(all_ids, k=self.interp_hvac_nb_agents)
            multi_factor = float(len(all_ids)) / self.interp_hvac_nb_agents

        # 设置插值点：根据当前日期和时间以及房屋的特性（如Ua、Cm、Ca等比率，室内外温差，HVAC功率等）,对于每个选中的房屋设定一个插值点。
        # Adding the interpolated power for each house
        for house_id in interp_house_ids:
            house = self.cluster_agents.houses[house_id]
            point["Ua_ratio"] = (
                house.Ua / self.default_house_prop["Ua"]
            )  # TODO: This is ugly as in the Monte Carlo, we compute the ratio based on the Ua in config. We should change the dict for absolute numbers.
            point["Cm_ratio"] = house.Cm / self.default_house_prop["Cm"]
            point["Ca_ratio"] = house.Ca / self.default_house_prop["Ca"]
            point["Hm_ratio"] = house.Hm / self.default_house_prop["Hm"]
            point["air_temp"] = house.current_temp - house.target_temp
            point["mass_temp"] = house.current_mass_temp - house.target_temp
            point["OD_temp"] = self.cluster_agents.current_OD_temp - house.target_temp
            point["HVAC_power"] = house.hvac.cooling_capacity
            # 使用 clipInterpolationPoint 函数确保插值点中的每个数值都在合理的范围内，并用 sortDictKeys 函数确保插值点的键按照特定顺序排列。
            point = clipInterpolationPoint(point, self.interp_parameters_dict)
            point = sortDictKeys(point, self.interp_dict_keys)
            # 对于选定的一组房屋（根据 interp_hvac_nb_agents 参数决定数量），使用插值模型（PowerInterpolator）计算每个房屋的基础功率，然后将这些功率加总，得到总的基础功率。
            base_power += self.power_interpolator.interpolateGridFast(point)[0][0]
        base_power *= multi_factor
        return base_power

    def step(self, date_time, time_step) -> float:
        """
        计算给定日期和时间的调节信号，用于模拟电力网络的输出。
        根据不同的信号模式（如平坦、正弦波、脉冲宽度调制、Perlin噪声等）计算信号值。
        考虑了不同的信号参数和基础功率模式。
        限制信号值不超过最大功率限制。
        
        Compute the regulation signal at given date and time

        Returns:
        current_hvac_active_signal: Current regulation signal in Watts

        Parameters:
        self
        date_time: datetime, current date and time
        """

        # 先计算各模式HVAC和EV的基础功率, 再进行各种花活
        if self.hvac_base_power_mode == "constant":
            self.base_power = self.avg_power_per_hvac * self.nb_hvacs
        elif self.hvac_base_power_mode == "interpolation":
            self.time_since_last_interp += time_step.seconds

            if self.time_since_last_interp >= self.interp_update_period:
                self.base_power = self.interpolatePower(date_time)
                self.time_since_last_interp = 0

        # Efan 待办 
        if self.ev_base_power_mode == "constant":
            self.ev_base_power = self.avg_power_per_ev * self.num_stations
        elif self.ev_base_power_mode == "ideal":  # 计算的理想功率
            self.ev_base_power = self.ideal_avg_power_per_station * self.num_stations

        # Efan 待办 
        if self.ev_q_base_power_mode == "constant":
            self.ev_q_base_power = self.avg_q_power_per_ev * self.num_stations
        elif self.ev_q_base_power_mode == "ideal":  # 计算的理想最大功率
            self.ev_q_base_power = self.max_q_power_per_station * self.num_stations

        # 功率信号的各种花活
        if self.signal_mode == "flat":
            self.current_hvac_active_signal = self.base_power
            self.current_ev_active_signal = self.ev_base_power
            self.current_ev_reactive_signal = self.ev_q_base_power

        elif self.signal_mode == "sinusoidals":
            """Compute the outdoors temperature based on the time, being the sum of several sinusoidal signals"""
            amplitudes = [
                self.base_power * ratio
                for ratio in self.signal_params["amplitude_ratios"]
            ]
            periods = self.signal_params["periods"]
            if len(periods) != len(amplitudes):
                raise ValueError(
                    "Power grid signal parameters: periods and amplitude_ratios lists should have the same length. Change it in the config.py file. len(periods): {}, leng(amplitude_ratios): {}.".format(
                        len(periods), len(amplitudes)
                    )
                )

            time_sec = date_time.hour * 3600 + date_time.minute * 60 + date_time.second

            signal = self.base_power
            for i in range(len(periods)):
                signal += amplitudes[i] * np.sin(2 * np.pi * time_sec / periods[i])
            self.current_hvac_active_signal = signal
            
            # ev
            reactive_amplitudes = [
                self.ev_base_power * ratio
                for ratio in self.signal_params["amplitude_ratios"]
            ]
            periods = self.signal_params["periods"]
            if len(periods) != len(reactive_amplitudes):
                raise ValueError(
                    "Power grid signal parameters: periods and amplitude_ratios lists should have the same length. Change it in the config.py file. len(periods): {}, leng(amplitude_ratios): {}.".format(
                        len(periods), len(reactive_amplitudes)
                    )
                )

            signal = self.ev_base_power
            for i in range(len(periods)):
                signal += reactive_amplitudes[i] * np.sin(2 * np.pi * time_sec / periods[i])
            self.current_ev_active_signal = signal

            # q
            reactive_amplitudes = [
                self.ev_q_base_power * ratio
                for ratio in self.signal_params["amplitude_ratios"]
            ]
            periods = self.signal_params["periods"]
            if len(periods) != len(reactive_amplitudes):
                raise ValueError(
                    "Power grid signal parameters: periods and amplitude_ratios lists should have the same length. Change it in the config.py file. len(periods): {}, leng(amplitude_ratios): {}.".format(
                        len(periods), len(reactive_amplitudes)
                    )
                )

            signal = self.ev_q_base_power
            for i in range(len(periods)):
                signal += reactive_amplitudes[i] * np.sin(2 * np.pi * time_sec / periods[i])
            self.current_ev_reactive_signal = signal

        elif self.signal_mode == "regular_steps":
            """Compute the outdoors temperature based on the time using pulse width modulation"""
            amplitude = self.signal_params["amplitude_per_hvac"] * self.nb_hvacs
            ratio = self.base_power / amplitude

            period = self.signal_params["period"]

            signal = 0
            time_sec = date_time.hour * 3600 + date_time.minute * 60 + date_time.second

            signal = amplitude * np.heaviside(
                (time_sec % period) - (1 - ratio) * period, 1
            )
            self.current_hvac_active_signal = signal
            
            # EV
            ev_amplitude = self.signal_params["amplitude_p_per_ev"] * self.num_stations
            ratio = self.ev_base_power / ev_amplitude

            period = self.signal_params["period"]

            signal = 0

            signal = ev_amplitude * np.heaviside(
                (time_sec % period) - (1 - ratio) * period, 1
            )
            self.current_ev_active_signal = signal    
            
            # EV无功
            reactive_amplitude = self.signal_params["amplitude_q_per_ev"] * self.num_stations
            ratio = self.ev_q_base_power / reactive_amplitude

            period = self.signal_params["period"]

            signal = 0

            signal = reactive_amplitude * np.heaviside(
                (time_sec % period) - (1 - ratio) * period, 1
            )
            self.current_ev_reactive_signal = signal    

        elif "perlin" in self.signal_mode :
            amplitude = self.signal_params["amplitude_ratios"]
            unix_time_stamp = time.mktime(date_time.timetuple()) % 86400  # Normalize to seconds in a day
            signal = self.base_power
            perlin = self.perlin.calculate_noise(unix_time_stamp)

            self.cumulated_abs_noise += np.abs(signal * amplitude * perlin)
            self.nb_steps += 1

            self.current_hvac_active_signal = np.maximum(0, signal + (signal * amplitude * perlin))


            # EV
            ev_amplitude = self.signal_params["amplitude_ratios"]
            # unix_time_stamp = time.mktime(date_time.timetuple()) % 86400
            signal = self.ev_base_power
            # perlin = self.perlin.calculate_noise(unix_time_stamp)

            self.cumulated_abs_ev_noise += np.abs(signal * ev_amplitude * perlin)

            self.current_ev_active_signal = np.maximum(0, signal + (signal * ev_amplitude * perlin))
            
            # q
            reactive_amplitude = self.signal_params["amplitude_ratios"]
            # unix_time_stamp = time.mktime(date_time.timetuple()) % 86400
            signal = self.ev_q_base_power
            # perlin = self.perlin.calculate_noise(unix_time_stamp)

            self.cumulated_abs_reactive_noise += np.abs(signal * reactive_amplitude * perlin)

            self.current_ev_reactive_signal = np.maximum(0, signal + (signal * reactive_amplitude * perlin))
            
        else:
            raise ValueError(
                "Invalid power grid signal mode: {}. Change value in the config file.".format(
                    self.signal_mode
                )
            )

        self.current_hvac_active_signal = self.current_hvac_active_signal * self.active_artificial_ratio    #Artificial_ration should be 1. Only change for experimental purposes.
        self.current_hvac_active_signal = np.minimum(self.current_hvac_active_signal, self.max_hvac_power)

        # Efan's 上面所有模式只针对HVAC的有功, 还需再加上EV的有功. 将电动汽车充电桩的总功率添加到基础功率中. 加的不是总station功率, 而是汽车的来的时间占比并平分给各充电站. 
        self.current_ev_active_signal = self.current_ev_active_signal * self.active_artificial_ratio
        self.current_ev_active_signal = np.minimum(self.current_ev_active_signal, self.total_max_station_power)
        
        # 更新总的有功信号
        # self.current_all_active_signal += self.current_ev_active_signal + self.current_hvac_active_signal

        # self.ev_q_base_power = np.sqrt(self.total_max_station_power**2 - self.current_ev_active_signal**2)
        self.current_ev_reactive_signal = self.current_ev_reactive_signal * self.reactive_artificial_ratio  # 乘比例来模拟为控制信号, 尽量为1
        self.current_ev_reactive_signal = np.minimum(self.current_ev_reactive_signal, self.max_reactive_power)
        
        self.cluster_agents.reactive_power_need_added = self.current_ev_reactive_signal  # Efan's 传回给station,可选择使用自适应无功而不是RL控制无功

        # 返回合计HVAC和EV的有功及无功
        return (self.current_hvac_active_signal, self.current_ev_active_signal, self.current_ev_reactive_signal)


# Efan 示例调用
from config import config_dict
if __name__ == "__main__":
    # 创建环境实例
    env = MADemandResponseEnv(config_dict)
    env.reset()
    # 使用环境中已生成的充电事件
    charging_events = env.ev_charging_events

    # 检查充电事件是否正确生成
    if charging_events:
        # 输出前五个充电事件的信息进行调试
        print("First 5 Charging Events:")
        for ev_data in charging_events[:5]:
            print(f"EV ID: {ev_data.id}, Arrival Time: {ev_data.arrival_time}, Departure Time: {ev_data.leaving_time}, Max Apparent Power: {ev_data.max_apparent_power} kVA, SoC Target: {ev_data.soc_target}")

        # 如果有超过五个事件，还需要输出最后五个事件
        if len(charging_events) > 5:
            print("...")
            for ev_data in charging_events[-5:]:
                print(f"EV ID: {ev_data.id}, Arrival Time: {ev_data.arrival_time}, Departure Time: {ev_data.leaving_time}, Max Apparent Power: {ev_data.max_apparent_power} kVA, SoC Target: {ev_data.soc_target}")
    else:
        print("No EV charging events are created.")

