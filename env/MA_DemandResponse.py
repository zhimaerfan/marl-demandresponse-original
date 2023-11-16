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
from datetime import datetime

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
    agent_ids: a list, containing the ids of every agents of the environment.
    nb_agents: an int, with the number of agents
    cluster: a ClusterHouses object modeling all the houses.
    power_grid: a PowerGrid object, modeling the power grid.

    Main functions:

    build_environment(self): Builds a new environment with noise on properties
    reset(self): Reset the environment
    step(self, action_dict): take a step in time for each TCL, given actions of TCL agents
    compute_rewards(self, temp_penalty_dict, cluster_hvac_power, power_grid_reg_signal): compute the reward of each TCL agent

    Helper functions:
    merge_cluster_powergrid_obs(self, cluster_obs_dict, power_grid_reg_signal, cluster_hvac_power): merge the cluster and powergrid observations for the TCL agents
    make_dones_dict(self): create the "done" signal for each TCL agent
    """

    start_datetime: datetime
    datetime: datetime
    time_step: timedelta

    def __init__(self, config, test=False):
        """
        类属性包括一些默认配置属性、噪声属性、日期时间属性、时间步长等。
        初始化函数__init__接受一个配置字典和一个布尔值test作为参数，用于初始化环境。

        Initialize the environment

        Parameters:
        config: dictionary, containing the default configuration properties of the environment, house, hvac, and noise
        test: boolean, true it is a testing environment, false if it is for training

        """
        super(MADemandResponseEnv, self).__init__()

        self.test = test

        self.default_env_prop = config["default_env_prop"]
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

        self.agent_ids = self.env_properties["agent_ids"]
        self.nb_agents = len(self.agent_ids)

        self.cluster = ClusterHouses(
            self.env_properties["cluster_prop"],
            self.agent_ids,
            self.datetime,
            self.time_step,
            self.default_env_prop
        )

        self.env_properties["power_grid_prop"]["max_power"] = self.cluster.max_power

        self.power_grid = PowerGrid(
            self.env_properties["power_grid_prop"],
            self.default_house_prop,
            self.env_properties["nb_hvac"],
            self.cluster,
        )
        self.power_grid.step(self.start_datetime, self.time_step)

    def reset(self):
        """
        这段代码定义了reset方法，用于重置环境到初始状态，并生成每个TCL代理的观察结果。
            首先调用build_environment方法，初始化或重置环境中各个组件的状态。
            调用make_cluster_obs_dict方法，生成集群观察字典，包含每个代理的观察结果。
            获取电网的当前调节信号和集群的HVAC总功率。
            调用merge_cluster_powergrid_obs方法，将集群观察字典、电网调节信号和集群HVAC功率合并，生成最终的观察字典。
            返回观察字典，供代理在训练或决策时使用。        

        重置环境。

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
        power_grid_reg_signal = self.power_grid.current_signal
        cluster_hvac_power = self.cluster.cluster_hvac_power

        # 19个集群观察结果+电网的调节信号+集群的HVAC功率=21
        obs_dict = self.merge_cluster_powergrid_obs(
            cluster_obs_dict, power_grid_reg_signal, cluster_hvac_power
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
        cluster_obs_dict, cluster_hvac_power, _ = self.cluster.step(
            self.datetime, action_dict, self.time_step
        )

        # Compute reward with the old grid signal
        rewards_dict = self.compute_rewards(cluster_hvac_power)

        # Power grid step
        power_grid_reg_signal = self.power_grid.step(self.datetime, self.time_step)

        # Merge observations
        obs_dict = self.merge_cluster_powergrid_obs(
            cluster_obs_dict, power_grid_reg_signal, cluster_hvac_power
        )

        dones_dict = self.make_dones_dict()
        info_dict = {"cluster_hvac_power": cluster_hvac_power}
        # print("cluster_hvac_power: {}, power_grid_reg_signal: {}".format(cluster_hvac_power, power_grid_reg_signal))

        return obs_dict, rewards_dict, dones_dict, info_dict

    def merge_cluster_powergrid_obs(
        self, cluster_obs_dict, power_grid_reg_signal, cluster_hvac_power
    ) -> None:
        """
        该方法用于合并集群和电网观察结果，以创建最终的观察结果，供代理在训练或决策时使用。
        
        Merge the cluster and powergrid observations for the TCL agents

        Returns:
        obs_dict: a dictionary, containing the observations for each TCL agent.

        Parameters:
        cluster_obs_dict: a dictionary, containing the cluster observations for each TCL agent.
        power_grid_reg_signal: a float. Regulation signal, or target total power, in Watts.
        cluster_hvac_power: a float. Total power used by the TCLs, in Watts.
        """

        obs_dict = cluster_obs_dict
        for agent_id in self.agent_ids:
            obs_dict[agent_id]["reg_signal"] = power_grid_reg_signal
            obs_dict[agent_id]["cluster_hvac_power"] = cluster_hvac_power

        return obs_dict

    def reg_signal_penalty(self, cluster_hvac_power):
        """
        该方法用于计算与电网调节信号和需求响应代理总功率之间的惩罚，以鼓励代理适应电网调节信号。
        
        Returns: a float, representing the positive penalty due to the distance between the regulation signal and the total power used by the TCLs.

        Parameters:
        cluster_hvac_power: a float. Total power used by the TCLs, in Watts.
        power_grid_reg_signal: a float. Regulation signal, or target total power, in Watts.
        """
        sig_penalty_mode = self.default_env_prop["reward_prop"]["sig_penalty_mode"]

        if sig_penalty_mode == "common_L2":
            penalty = (
                (cluster_hvac_power - self.power_grid.current_signal) / self.nb_agents
            ) ** 2
        else:
            raise ValueError("Unknown signal penalty mode: {}".format(sig_penalty_mode))

        return penalty

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
            for house_id in self.agent_ids:
                house = self.cluster.houses[house_id]
                house_temperature_penalty = deadbandL2(
                    house.target_temp, house.deadband, house.current_temp
                )
                temperature_penalty += house_temperature_penalty / self.nb_agents

        elif temp_penalty_mode == "common_max":
            temperature_penalty = 0
            for house_id in self.agent_ids:
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
            for house_id in self.agent_ids:
                house = self.cluster.houses[house_id]
                house_temperature_penalty = deadbandL2(
                    house.target_temp, house.deadband, house.current_temp
                )
                if house_id == one_house_id:
                    ind_L2 = house_temperature_penalty
                common_L2 += house_temperature_penalty / self.nb_agents
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

    def compute_rewards(self, cluster_hvac_power):
        """
        该方法用于计算每个需求响应代理的奖励，根据温度惩罚和电网调节信号惩罚的组合计算奖励。

        Compute the reward of each TCL agent

        Returns:
        rewards_dict: a dictionary, containing the rewards of each TCL agent.

        Parameters:
        temp_penalty_dict: a dictionary, containing the temperature penalty for each TCL agent
        cluster_hvac_power: a float. Total power used by the TCLs, in Watts.
        power_grid_reg_signal: a float. Regulation signal, or target total power, in Watts.
        """

        rewards_dict: dict[str, float] = {}
        signal_penalty = self.reg_signal_penalty(cluster_hvac_power)

        norm_temp_penalty = deadbandL2(
            self.default_house_prop["target_temp"],
            0,
            self.default_house_prop["target_temp"] + 1,
        )

        norm_sig_penalty = deadbandL2(
            self.default_env_prop["reward_prop"]["norm_reg_sig"],
            0,
            0.75 * self.default_env_prop["reward_prop"]["norm_reg_sig"],
        )

        temp_penalty_dict = {}
        # Temperature penalties
        for house_id in self.agent_ids:
            house = self.cluster.houses[house_id]
            temp_penalty_dict[house_id] = self.compute_temp_penalty(house_id)

        for agent_id in self.agent_ids:
            rewards_dict[agent_id] = -1 * (
                self.env_properties["reward_prop"]["alpha_temp"]
                * temp_penalty_dict[agent_id]
                / norm_temp_penalty
                + self.env_properties["reward_prop"]["alpha_sig"]
                * signal_penalty
                / norm_sig_penalty
            )
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
        for agent_id in self.agent_ids:
            dones_dict[
                agent_id
            ] = False  # There is no state which terminates the environment.
        return dones_dict


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
        初始化HVAC对象，根据传入的HVAC配置属性和时间步长。
        
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
            self.turned_on = command
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
        if self.disp_count >= 10000:
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


class ClusterHouses(object):
    """
    一个包含多个房屋的集群，这些房屋具有相同的室外温度。集群行为，包括控制HVAC系统、计算室外温度和生成代理的观察信息。在需求响应环境中，多个集群可以用于研究不同的HVAC策略和代理通信模式对集群行为和电力消耗的影响。
    cluster_prop：包含集群配置属性的字典。
    houses：包含集群中所有房屋的字典。
    hvacs_id_registry：将每个HVAC与其所属房屋相对应的字典。
    day_temp：白天的最高温度，以摄氏度为单位。
    night_temp：夜晚的最低温度，以摄氏度为单位。
    temp_std：温度标准差，以摄氏度为单位。
    current_OD_temp：当前的室外温度，以摄氏度为单位。
    cluster_hvac_power：所有集群HVAC的当前累积电功率消耗，以瓦特（Watts）为单位。

    A cluster contains several houses, with the same outdoors temperature.

    Attributes:
    cluster_prop: dictionary, containing the configuration properties of the cluster
    houses: dictionary, containing all the houses in the Cluster
    hvacs_id_registry: dictionary, mapping each HVAC to its house
    day_temp: float, maximal temperature during the day, in Celsius
    night_temp: float, minimal temperature during the night, in Celsius
    temp_std: float, standard deviation of the temperature, in Celsius
    current_OD_temp: float, current outdoors temperature, in Celsius
    cluster_hvac_power: float, current cumulative electric power consumption of all cluster HVACs, in Watts

    Functions:
    make_cluster_obs_dict(self, date_time): generate the cluster observation dictionary for all agents
    step(self, date_time, actions_dict, time_step): take a step in time for all the houses in the cluster
    compute_OD_temp(self, date_time): models the outdoors temperature
    """

    def __init__(self, cluster_prop, agent_ids, date_time, time_step, default_env_properties):
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
      
        self.cluster_prop = cluster_prop
        self.agent_ids = agent_ids
        self.nb_agents = len(agent_ids)
        # print("nb agents: {}".format(self.nb_agents))

        # Houses
        self.houses = {}
        for house_properties in cluster_prop["houses_properties"]:
            house = SingleHouse(house_properties, time_step)
            self.houses[house.id] = house

        self.temp_mode = cluster_prop["temp_mode"]
        self.temp_params = cluster_prop["temp_parameters"][self.temp_mode]
        self.day_temp = self.temp_params["day_temp"]
        self.night_temp = self.temp_params["night_temp"]
        self.temp_std = self.temp_params["temp_std"]
        self.random_phase_offset = self.temp_params["random_phase_offset"]
        self.env_prop = default_env_properties
        if self.random_phase_offset:
            self.phase = random.random() * 24
        else:
            self.phase = 0
        self.current_OD_temp = self.compute_OD_temp(date_time)

        # Compute the Initial cluster_hvac_power
        self.cluster_hvac_power = 0
        self.max_power = 0
        for house_id in self.houses.keys():
            house = self.houses[house_id]
            hvac = house.hvac
            self.cluster_hvac_power += hvac.power_consumption()
            self.max_power += hvac.max_consumption

        self.build_agent_comm_links()

    def build_agent_comm_links(self):
        """
        建立代理之间的通信链接，根据集群配置中的通信模式和参数。
        可以根据不同的通信模式选择代理之间的通信策略，如邻居、封闭组、随机样本等。
        """
        self.agent_communicators = {}
        nb_comm = np.minimum(
            self.cluster_prop["nb_agents_comm"], self.cluster_prop["nb_agents"] - 1
        )

        if self.cluster_prop["agents_comm_mode"] == "neighbours":
            # This is to get the neighbours of each agent in a circular fashion,
            # if agent_id is 5, the half before will be [0, 1, 2, 3, 4] and half after will be [6, 7, 8, 9, 10]
            # if agent_id is 1, the half before will be [7, 8, 9, 10, 0] and half after will be [2, 3, 4, 5, 6]
            for agent_id in self.agent_ids:
                possible_ids = deepcopy(self.agent_ids)
                # Give neighbours (in a circular manner when reaching extremes of the .
                half_before = [
                    (agent_id - int(np.floor(nb_comm / 2)) + i) % len(possible_ids)
                    for i in range(int(np.floor(nb_comm / 2)))
                ]
                half_after = [
                    (agent_id + 1 + i) % len(possible_ids)
                    for i in range(int(np.ceil(nb_comm / 2)))
                ]
                ids_houses_messages = half_before + half_after
                self.agent_communicators[agent_id] = ids_houses_messages

        elif self.cluster_prop["agents_comm_mode"] == "closed_groups":
            for agent_id in self.agent_ids:
                possible_ids = deepcopy(self.agent_ids)
                base = agent_id - (agent_id % (nb_comm + 1))
                if base + nb_comm <= self.cluster_prop["nb_agents"]:
                    ids_houses_messages = [
                        base + i for i in range(self.cluster_prop["nb_agents_comm"] + 1)
                    ]
                else:
                    ids_houses_messages = [
                        self.cluster_prop["nb_agents"] - nb_comm - 1 + i
                        for i in range(nb_comm + 1)
                    ]
                ids_houses_messages.remove(agent_id)
                self.agent_communicators[agent_id] = ids_houses_messages

        elif self.cluster_prop["agents_comm_mode"] == "random_sample":
            pass

        elif self.cluster_prop["agents_comm_mode"] == "random_fixed":
            for agent_id in self.agent_ids:
                possible_ids = deepcopy(self.agent_ids)
                possible_ids.remove(agent_id)
                ids_houses_messages = random.sample(possible_ids, k=nb_comm)
                self.agent_communicators[agent_id] = ids_houses_messages

        elif self.cluster_prop["agents_comm_mode"] == "neighbours_2D":
            row_size = self.cluster_prop["agents_comm_parameters"]["neighbours_2D"]["row_size"]
            distance_comm = self.cluster_prop["agents_comm_parameters"]["neighbours_2D"]["distance_comm"]
            if self.nb_agents % row_size != 0:
                raise ValueError("Neighbours 2D row_size must be a divisor of nb_agents")

            max_y = self.nb_agents // row_size
            if distance_comm >= (row_size+1) // 2 or distance_comm >= (max_y+1) // 2:
                raise ValueError("Neighbours 2D distance_comm ({}) must be strictly smaller than (row_size+1) / 2 ({}) and (max_y+1) / 2 ({})".format(distance_comm, (row_size+1) // 2, (max_y+1) // 2))

            distance_pattern = []
            for x_diff in range(-1*distance_comm, distance_comm + 1):
                for y_diff in range(-1*distance_comm, distance_comm + 1):
                    if abs(x_diff) + abs(y_diff) <= distance_comm and (x_diff != 0 or y_diff != 0):
                        distance_pattern.append((x_diff, y_diff))


            for agent_id in self.agent_ids:
                x = agent_id % row_size
                y = agent_id // row_size
                ids_houses_messages = []
                for pair_diff in distance_pattern:
                    x_new = x + pair_diff[0]
                    y_new = y + pair_diff[1]
                    if x_new < 0:
                        x_new += row_size
                    if x_new >= row_size:
                        x_new -= row_size
                    if y_new < 0:
                        y_new += max_y
                    if y_new >= max_y:
                        y_new -= max_y
                    agent_id_new = y_new*row_size + x_new
                    ids_houses_messages.append(agent_id_new)
                self.agent_communicators[agent_id] = ids_houses_messages
            print("self.agent_communicators: {}".format(self.agent_communicators))
        
        elif self.cluster_prop["agents_comm_mode"] == "no_message":
            for agent_id in self.agent_ids:
                self.agent_communicators[agent_id] = []

        else:
            raise ValueError(
                "Cluster property: unknown agents_comm_mode '{}'.".format(
                    self.cluster_prop["agents_comm_mode"]
                )
            )

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
        for house_id in self.houses.keys():
            cluster_obs_dict[house_id] = {}

            # 获取房屋和HVAC对象
            # Getting the house and the HVAC
            house = self.houses[house_id]
            hvac = house.hvac

            # 从集群中获取动态值
            # Dynamic values from cluster
            cluster_obs_dict[house_id]["OD_temp"] = self.current_OD_temp
            cluster_obs_dict[house_id]["datetime"] = date_time

            # Dynamic values from house
            cluster_obs_dict[house_id]["house_temp"] = house.current_temp
            cluster_obs_dict[house_id]["house_mass_temp"] = house.current_mass_temp

            # Dynamic values from HVAC
            cluster_obs_dict[house_id]["hvac_turned_on"] = hvac.turned_on
            cluster_obs_dict[house_id][
                "hvac_seconds_since_off"
            ] = hvac.seconds_since_off
            cluster_obs_dict[house_id]["hvac_lockout"] = hvac.lockout

            # 从房屋中获取常量值（以后可能会更改）
            # Supposedly constant values from house (may be changed later)
            cluster_obs_dict[house_id]["house_target_temp"] = house.target_temp
            cluster_obs_dict[house_id]["house_deadband"] = house.deadband
            cluster_obs_dict[house_id]["house_Ua"] = house.Ua
            cluster_obs_dict[house_id]["house_Cm"] = house.Cm
            cluster_obs_dict[house_id]["house_Ca"] = house.Ca
            cluster_obs_dict[house_id]["house_Hm"] = house.Hm
            cluster_obs_dict[house_id]["house_solar_gain"] = house.current_solar_gain

            # Supposedly constant values from hvac
            cluster_obs_dict[house_id]["hvac_COP"] = hvac.COP
            cluster_obs_dict[house_id]["hvac_cooling_capacity"] = hvac.cooling_capacity
            cluster_obs_dict[house_id][
                "hvac_latent_cooling_fraction"
            ] = hvac.latent_cooling_fraction
            cluster_obs_dict[house_id]["hvac_lockout_duration"] = hvac.lockout_duration
            
            # Messages from the other agents

            if self.cluster_prop["agents_comm_mode"] == "random_sample":
                possible_ids = deepcopy(self.agent_ids)
                nb_comm = np.minimum(
                    self.cluster_prop["nb_agents_comm"],
                    self.cluster_prop["nb_agents"] - 1,
                )
                possible_ids.remove(house_id)
                ids_houses_messages = random.sample(possible_ids, k=nb_comm)

            else:
                ids_houses_messages = self.agent_communicators[house_id]


            # 初始化消息列表
            cluster_obs_dict[house_id]["message"] = []
            for id_house_message in ids_houses_messages:
                if np.random.rand() > self.cluster_prop["comm_defect_prob"]:
                    # 添加正常消息
                    cluster_obs_dict[house_id]["message"].append(
                        self.houses[id_house_message].message(self.env_prop["message_properties"])
                    )
                else:
                    
                    # 添加空消息
                    cluster_obs_dict[house_id]["message"].append(
                        self.houses[id_house_message].message(self.env_prop["message_properties"], empty=True)
                    )
        return cluster_obs_dict

    def step(self, date_time, actions_dict, time_step):
        """
        推进集群中所有房屋的状态，包括HVAC的控制。
        更新室外温度，生成集群观察字典，计算温度惩罚，并返回相关信息。
        
        Take a step in time for all the houses in the cluster

        Returns:
        cluster_obs_dict: dictionary, containing the cluster observations for every TCL agent.
        temp_penalty_dict: dictionary, containing the temperature penalty for each TCL agent
        cluster_hvac_power: float, total power used by the TCLs, in Watts.
        info_dict: dictonary, containing additional information for each TCL agent.

        Parameters:
        date_time: datetime, current date and time
        actions_dict: dictionary, containing the actions of each TCL agent.
        time_step: timedelta, time step of the simulation
        """

        # Send command to the hvacs
        for house_id in self.houses.keys():
            # Getting the house and the HVAC
            house = self.houses[house_id]
            hvac = house.hvac
            if house_id in actions_dict.keys():
                command = actions_dict[house_id]
            else:
                warnings.warn(
                    "HVAC in house {} did not receive any command.".format(house_id)
                )
                command = False
            hvac.step(command)
            house.step(self.current_OD_temp, time_step, date_time)

        # Update outdoors temperature
        self.current_OD_temp = self.compute_OD_temp(date_time)
        ## Observations
        cluster_obs_dict = self.make_cluster_obs_dict(date_time)

        ## Temperature penalties and total cluster power consumption
        self.cluster_hvac_power = 0

        for house_id in self.houses.keys():
            # Getting the house and the HVAC
            house = self.houses[house_id]
            hvac = house.hvac

            # Cluster hvac power consumption
            self.cluster_hvac_power += hvac.power_consumption()

        # Info
        info_dict = {}  # Not necessary for the moment

        return cluster_obs_dict, self.cluster_hvac_power, info_dict

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

    avg_power_per_hvac：每个HVAC设备的平均功率，以瓦特（Watts）为单位。
    signal_mode：信号变化的模式（可以是"none"或"sinusoidal"）。
    signal_params：信号变化的参数字典。
    nb_hvacs：集群中HVAC设备的数量。
    init_signal：每个HVAC的初始信号值，以瓦特（Watts）为单位。
    current_signal：当前的信号值，以瓦特（Watts）为单位。

    Simulated power grid outputting the regulation signal.

    Attributes:
    avg_power_per_hvac: float, average power to be given per HVAC, in Watts
    signal_mode: string, mode of variation in the signal (can be none or sinusoidal)
    signal_params: dictionary, parameters of the variation of the signal
    nb_hvac: int, number of HVACs in the cluster
    init_signal: float, initial signal value per HVAC, in Watts
    current_signal: float, current signal in Watts

    Functions:baharerajabi2015@gmail.combaharerajabi2015@gmail.com
    step(self, date_time): Computes the regulation signal at given date and time
    """

    def __init__(
        self, power_grid_prop, default_house_prop, nb_hvacs, cluster_houses=None
    ):
        """
        初始化PowerGrid对象，根据传入的电力网配置属性、默认房屋属性和HVAC数量。
        设置基础功率模式、初始信号值、人工干扰比例等属性。
        根据配置文件中的不同模式，初始化不同的电力基础功率。
        
        Initialize PowerGrid.

        Returns: -

        Parameters:
        power_grid_prop: dictionary, containing the configuration properties of the power grid
        nb_hvacs: int, number of HVACs in the cluster
        """

        # Base power
        self.base_power_mode = power_grid_prop["base_power_mode"]
        self.init_signal_per_hvac = power_grid_prop["base_power_parameters"]["constant"]["init_signal_per_hvac"]
        self.artificial_ratio = power_grid_prop["artificial_ratio"] * power_grid_prop["artificial_signal_ratio_range"]**(random.random()*2 - 1)      # Base ratio, randomly multiplying by a number between 1/artificial_signal_ratio_range and artificial_signal_ratio_range, scaled on a logarithmic scale.
        self.cumulated_abs_noise = 0
        self.nb_steps = 0

        ## Constant base power
        if self.base_power_mode == "constant":
            self.avg_power_per_hvac = power_grid_prop["base_power_parameters"][
                "constant"
            ]["avg_power_per_hvac"]
            self.init_signal_per_hvac = power_grid_prop["base_power_parameters"][
                "constant"
            ]["init_signal_per_hvac"]

        ## Interpolated base power
        elif self.base_power_mode == "interpolation":
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
            self.interp_nb_agents = power_grid_prop["base_power_parameters"][
                "interpolation"
            ]["interp_nb_agents"]

            if cluster_houses:
                self.cluster_houses = cluster_houses
            else:
                raise ValueError(
                    "The PowerGrid object in interpolation mode needs a ClusterHouses object as a cluster_houses argument."
                )
        ## Error

        else:
            raise ValueError(
                "The base_power_mode parameter in the config file can only be 'constant' or 'interpolation'. It is currently: {}".format(
                    self.base_power_mode
                )
            )

        self.max_power = power_grid_prop["max_power"]

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




    def interpolatePower(self, date_time):
        """
        计算插值基础功率值，用于基于集群房屋特性的插值计算。
        考虑了房屋的各种属性，包括日期、时间、Ua、Cm、Ca、Hm、空气温度、质量温度、室外温度和HVAC功率等。
        可以根据不同的房屋特性进行插值计算，以模拟实际基础功率的变化。
        """
        
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

        all_ids = list(self.cluster_houses.houses.keys())
        if len(all_ids) <= self.interp_nb_agents:
            interp_house_ids = all_ids
            multi_factor = 1
        else:
            interp_house_ids = random.choices(all_ids, k=self.interp_nb_agents)
            multi_factor = float(len(all_ids)) / self.interp_nb_agents

        # Adding the interpolated power for each house
        for house_id in interp_house_ids:
            house = self.cluster_houses.houses[house_id]
            point["Ua_ratio"] = (
                house.Ua / self.default_house_prop["Ua"]
            )  # TODO: This is ugly as in the Monte Carlo, we compute the ratio based on the Ua in config. We should change the dict for absolute numbers.
            point["Cm_ratio"] = house.Cm / self.default_house_prop["Cm"]
            point["Ca_ratio"] = house.Ca / self.default_house_prop["Ca"]
            point["Hm_ratio"] = house.Hm / self.default_house_prop["Hm"]
            point["air_temp"] = house.current_temp - house.target_temp
            point["mass_temp"] = house.current_mass_temp - house.target_temp
            point["OD_temp"] = self.cluster_houses.current_OD_temp - house.target_temp
            point["HVAC_power"] = house.hvac.cooling_capacity
            point = clipInterpolationPoint(point, self.interp_parameters_dict)
            point = sortDictKeys(point, self.interp_dict_keys)
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
        current_signal: Current regulation signal in Watts

        Parameters:
        self
        date_time: datetime, current date and time
        """

        if self.base_power_mode == "constant":
            self.base_power = self.avg_power_per_hvac * self.nb_hvacs
        elif self.base_power_mode == "interpolation":
            self.time_since_last_interp += time_step.seconds

            if self.time_since_last_interp >= self.interp_update_period:
                self.base_power = self.interpolatePower(date_time)
                self.time_since_last_interp = 0

        if self.signal_mode == "flat":
            self.current_signal = self.base_power

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
            self.current_signal = signal

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
            self.current_signal = signal
        elif "perlin" in self.signal_mode :
            amplitude = self.signal_params["amplitude_ratios"]
            unix_time_stamp = time.mktime(date_time.timetuple()) % 86400
            signal = self.base_power
            perlin = self.perlin.calculate_noise(unix_time_stamp)

            self.cumulated_abs_noise += np.abs(signal * amplitude * perlin)
            self.nb_steps += 1

            self.current_signal = np.maximum(0, signal + (signal * amplitude * perlin))
        else:
            raise ValueError(
                "Invalid power grid signal mode: {}. Change value in the config file.".format(
                    self.signal_mode
                )
            )

        self.current_signal = self.current_signal * self.artificial_ratio    #Artificial_ration should be 1. Only change for experimental purposes.

        self.current_signal = np.minimum(self.current_signal, self.max_power)

        return self.current_signal
