import gym
import ray
import numpy as np
import warnings
import random
from copy import deepcopy


from datetime import datetime, timedelta, time
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.typing import MultiAgentDict, AgentID
from typing import Tuple, Dict, List, Any
import sys

sys.path.append("..")
from utils import applyPropertyNoise


def reg_signal_penalty(cluster_hvac_power, power_grid_reg_signal, nb_agents):
    """
    Returns: a float, representing the positive penalty due to the distance between the regulation signal and the total power used by the TCLs.

    Parameters:
    cluster_hvac_power: a float. Total power used by the TCLs, in Watts.
    power_grid_reg_signal: a float. Regulation signal, or target total power, in Watts.
    """

    penalty = ((cluster_hvac_power - power_grid_reg_signal) / nb_agents) ** 2

    return penalty


def compute_temp_penalty(target_temp, deadband, house_temp):
    """
    Returns: a float, representing the positive penalty due to distance between the target (indoors) temperature and the indoors temperature in a house.

    Parameters:
    target_temp: a float. Target indoors air temperature, in Celsius.
    deadband: a float. Margin of tolerance for indoors air temperature difference, in Celsius.
    house_temp: a float. Current indoors air temperature, in Celsius
    """

    if target_temp + deadband / 2 < house_temp:
        temperature_penalty = (house_temp - (target_temp + deadband / 2)) ** 2
    elif target_temp - deadband / 2 > house_temp:
        temperature_penalty = ((target_temp - deadband / 2) - house_temp) ** 2
    else:
        temperature_penalty = 0.0
    temperature_penalty = np.clip(temperature_penalty, 0, 20)
    return temperature_penalty


class MADemandResponseEnv(MultiAgentEnv):
    """
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
            self.env_properties["cluster_prop"], self.datetime, self.time_step
        )
        self.power_grid = PowerGrid(
            self.env_properties["power_grid_prop"], self.env_properties["nb_hvac"]
        )

    def reset(self):
        """
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

        obs_dict = self.merge_cluster_powergrid_obs(
            cluster_obs_dict, power_grid_reg_signal, cluster_hvac_power
        )

        return obs_dict

    def step(self, action_dict):
        """
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
        cluster_obs_dict, temp_penalty_dict, cluster_hvac_power, _ = self.cluster.step(
            self.datetime, action_dict, self.time_step
        )
        # Power grid step
        power_grid_reg_signal = self.power_grid.step(self.datetime)

        # Merge observations
        obs_dict = self.merge_cluster_powergrid_obs(
            cluster_obs_dict, power_grid_reg_signal, cluster_hvac_power
        )

        # Compute reward
        rewards_dict = self.compute_rewards(
            temp_penalty_dict, cluster_hvac_power, power_grid_reg_signal
        )
        dones_dict = self.make_dones_dict()
        info_dict = {"cluster_hvac_power": cluster_hvac_power}
        # print("cluster_hvac_power: {}, power_grid_reg_signal: {}".format(cluster_hvac_power, power_grid_reg_signal))

        return obs_dict, rewards_dict, dones_dict, info_dict

    def merge_cluster_powergrid_obs(
        self, cluster_obs_dict, power_grid_reg_signal, cluster_hvac_power
    ) -> None:
        """
        Merge the cluster and powergrid observations for the TCL agents

        Returns:
        obs_dict: a dictionary, containing the observations for each TCL agent.

        Parameters:
        cluster_obs_dict: a dictionary, containing the cluster observations for each TCL agent.
        power_grid_reg_signal: a float. Regulation signal, or target total power, in Watts.
        cluster_hvac_power: a float. Total power used by the TCLs, in Watts.
        """

        obs_dict = deepcopy(cluster_obs_dict)
        for agent_id in self.agent_ids:
            obs_dict[agent_id]["reg_signal"] = power_grid_reg_signal
            obs_dict[agent_id]["cluster_hvac_power"] = cluster_hvac_power

        return obs_dict

    def compute_rewards(
        self, temp_penalty_dict, cluster_hvac_power, power_grid_reg_signal
    ):
        """
        Compute the reward of each TCL agent

        Returns:
        rewards_dict: a dictionary, containing the rewards of each TCL agent.

        Parameters:
        temp_penalty_dict: a dictionary, containing the temperature penalty for each TCL agent
        cluster_hvac_power: a float. Total power used by the TCLs, in Watts.
        power_grid_reg_signal: a float. Regulation signal, or target total power, in Watts.
        """

        rewards_dict: dict[str, float] = {}
        signal_penalty = reg_signal_penalty(
            cluster_hvac_power, power_grid_reg_signal, self.nb_agents
        )

        norm_temp_penalty = compute_temp_penalty(
            self.default_house_prop["target_temp"],
            2,
            self.default_house_prop["target_temp"] + 3,
        )

        norm_sig_penalty = reg_signal_penalty(
            self.default_env_prop["power_grid_prop"]["avg_power_per_hvac"],
            0.75 * self.default_env_prop["power_grid_prop"]["avg_power_per_hvac"],
            1,
        )

        for agent_id in self.agent_ids:
            rewards_dict[agent_id] = -1 * (
                self.env_properties["alpha_temp"]
                * temp_penalty_dict[agent_id]
                / norm_temp_penalty
                + self.env_properties["alpha_sig"] * signal_penalty / norm_sig_penalty
            )
        return rewards_dict

    def make_dones_dict(self):
        """
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
        self.lockout_duration = hvac_properties["lockout_duration"]
        self.turned_on = False
        self.lockout = False
        self.seconds_since_off = self.lockout_duration
        self.time_step = time_step

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
        Compute the electric power consumption of the HVAC

        Return:
        power_cons: float, electric power consumption of the HVAC, in Watts
        """
        if self.turned_on:
            power_cons = self.cooling_capacity / self.COP
        else:
            power_cons = 0

        return power_cons


class SingleHouse(object):
    """
    Single house simulator.
    **Attention** Although the infrastructure could support more, each house can currently only have one HVAC (several HVAC/house not implemented yet)

    Attributes:
    house_properties: dictionary, containing the configuration properties of the SingleHouse object
    id: string, unique identifier of he house.
    init_temp: float, initial indoors air temperature of the house, in Celsius
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
    hvacs: dictionary, containing the hvacs of the house
    hvacs_ids: list, containing the ids of the hvacs of the house
    disp_count: int, iterator for printing count

    Functions:
    step(self, od_temp, time_step): Take a time step for the house
    update_temperature(self, od_temp, time_step): Compute the new temperatures depending on the state of the house's HVACs
    """

    def __init__(self, house_properties, time_step):
        """
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
        self.hvacs = {}
        self.hvacs_ids = []

        for hvac_prop in house_properties["hvac_properties"]:
            hvac = HVAC(hvac_prop, time_step)
            self.hvacs[hvac.id] = hvac
            self.hvacs_ids.append(hvac.id)

        if len(self.hvacs_ids) == 0:
            warnings.warn("House {} has no HVAC".format(self.id))

        if len(self.hvacs_ids) >= 2:
            raise NotImplementedError(
                "House {} has {} HVACs, which is more than one. The current simulator does not support several HVACs per house.".format(
                    self.id, len(self.hvacs_ids)
                )
            )

        self.disp_count = 0

    def step(self, od_temp, time_step, date_time):
        """
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
                    self.hvacs[self.id + "_1"].turned_on,
                    self.hvacs[self.id + "_1"].seconds_since_off,
                    date_time,
                )
            )
            self.disp_count = 0

    def update_temperature(self, od_temp, time_step, date_time):
        """
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
        total_Qhvac = 0
        for hvac_id in self.hvacs_ids:
            hvac = self.hvacs[hvac_id]
            total_Qhvac += hvac.get_Q()

        # Total heat addition to air
        other_Qa = self.house_solar_gain(date_time)  # windows, ...
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

    def house_solar_gain(self, date_time):
        """
        Computes the solar gain, i.e. the heat transfer received from the sun through the windows.

        Return:
        solar_gain: float, direct solar radiation passing through the windows at a given moment in Watts

        Parameters
        date_time: datetime, current date and time

        ---
        Source and assumptions:
        CIBSE. (2015). Environmental Design - CIBSE Guide A (8th Edition) - 5.9.7 Solar Cooling Load Tables. CIBSE.
        Retrieved from https://app.knovel.com/hotlink/pdf/id:kt0114THK9/environmental-design/solar-cooling-load-tables
        Table available: https://www.cibse.org/Knowledge/Guide-A-2015-Supplementary-Files/Chapter-5

        Coefficient obtained by performing a polynomial regression on the table "solar cooling load at stated sun time at latitude 30".

        Based on the following assumptions.
        - Latitude is 30. (The latitude of Austin in Texas is 30.266666)
        - The SCL before 7:30 and after 17:30 is negligible for latitude 30.
        - The windows are distributed perfectly evenly around the building.
        - There are no horizontal windows, for example on the roof.
        """

        x = date_time.hour + date_time.minute / 60 - 7.5
        if x < 0 or x > 10:
            solar_cooling_load = 0
        else:
            y = date_time.month + date_time.day / 30
            coeff = [
                4.36579418e01,
                1.58055357e02,
                8.76635241e01,
                -4.55944821e01,
                3.24275366e00,
                -4.56096472e-01,
                -1.47795612e01,
                4.68950855e00,
                -3.73313090e01,
                5.78827663e00,
                1.04354810e00,
                2.12969604e-02,
                2.58881400e-03,
                -5.11397219e-04,
                1.56398008e-02,
                -1.18302764e-01,
                -2.71446436e-01,
                -3.97855577e-02,
            ]

            solar_cooling_load = (
                coeff[0]
                + x * coeff[1]
                + y * coeff[2]
                + x**2 * coeff[3]
                + x**2 * y * coeff[4]
                + x**2 * y**2 * coeff[5]
                + y**2 * coeff[6]
                + x * y**2 * coeff[7]
                + x * y * coeff[8]
                + x**3 * coeff[9]
                + y**3 * coeff[10]
                + x**3 * y * coeff[11]
                + x**3 * y**2 * coeff[12]
                + x**3 * y**3 * coeff[13]
                + x**2 * y**3 * coeff[14]
                + x * y**3 * coeff[15]
                + x**4 * coeff[16]
                + y**4 * coeff[17]
            )

        solar_gain = self.window_area * self.shading_coeff * solar_cooling_load
        return solar_gain


class ClusterHouses(object):
    """
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

    def __init__(self, cluster_prop, date_time, time_step):
        """
        Initialize the cluster of houses

        Parameters:
        cluster_prop: dictionary, containing the configuration properties of the cluster
        date_time: datetime, initial date and time
        time_step: timedelta, time step of the simulation
        """
        self.cluster_prop = cluster_prop

        # Houses
        self.houses = {}
        self.hvacs_id_registry = {}
        for house_properties in cluster_prop["houses_properties"]:
            house = SingleHouse(house_properties, time_step)
            self.houses[house.id] = house
            for hvac_id in house.hvacs_ids:
                self.hvacs_id_registry[hvac_id] = house.id

        self.temp_mode = cluster_prop["temp_mode"]
        self.temp_params = cluster_prop["temp_parameters"][self.temp_mode]
        self.day_temp = self.temp_params["day_temp"]
        self.night_temp = self.temp_params["night_temp"]
        self.temp_std = self.temp_params["temp_std"]
        self.current_OD_temp = self.compute_OD_temp(date_time)

        # Compute the Initial cluster_hvac_power
        self.cluster_hvac_power = 0
        for hvac_id in self.hvacs_id_registry.keys():
            # Getting the house and the HVAC
            house_id = self.hvacs_id_registry[hvac_id]
            house = self.houses[house_id]
            hvac = house.hvacs[hvac_id]
            self.cluster_hvac_power += hvac.power_consumption()

    def make_cluster_obs_dict(self, date_time):
        """
        Generate the cluster observation dictionary for all agents.

        Return:
        cluster_obs_dict: dictionary, containing the cluster observations for every TCL agent.

        Parameters:
        self
        date_time: datetime, current date and time
        """
        cluster_obs_dict = {}
        for hvac_id in self.hvacs_id_registry.keys():
            cluster_obs_dict[hvac_id] = {}

            # Getting the house and the HVAC
            house_id = self.hvacs_id_registry[hvac_id]
            house = self.houses[house_id]
            hvac = house.hvacs[hvac_id]

            # Dynamic values from cluster
            cluster_obs_dict[hvac_id]["OD_temp"] = self.current_OD_temp
            cluster_obs_dict[hvac_id]["datetime"] = date_time

            # Dynamic values from house
            cluster_obs_dict[hvac_id]["house_temp"] = house.current_temp
            cluster_obs_dict[hvac_id]["house_mass_temp"] = house.current_mass_temp

            # Dynamic values from HVAC
            cluster_obs_dict[hvac_id]["hvac_turned_on"] = hvac.turned_on
            cluster_obs_dict[hvac_id]["hvac_seconds_since_off"] = hvac.seconds_since_off
            cluster_obs_dict[hvac_id]["hvac_lockout"] = hvac.lockout

            # Supposedly constant values from house (may be changed later)
            cluster_obs_dict[hvac_id]["house_target_temp"] = house.target_temp
            cluster_obs_dict[hvac_id]["house_deadband"] = house.deadband
            cluster_obs_dict[hvac_id]["house_Ua"] = house.Ua
            cluster_obs_dict[hvac_id]["house_Cm"] = house.Cm
            cluster_obs_dict[hvac_id]["house_Ca"] = house.Ca
            cluster_obs_dict[hvac_id]["house_Hm"] = house.Hm

            # Supposedly constant values from hvac
            cluster_obs_dict[hvac_id]["hvac_COP"] = hvac.COP
            cluster_obs_dict[hvac_id]["hvac_cooling_capacity"] = hvac.cooling_capacity
            cluster_obs_dict[hvac_id][
                "hvac_latent_cooling_fraction"
            ] = hvac.latent_cooling_fraction
            cluster_obs_dict[hvac_id]["hvac_lockout_duration"] = hvac.lockout_duration

        return cluster_obs_dict

    def step(self, date_time, actions_dict, time_step):
        """
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
        for hvac_id in self.hvacs_id_registry.keys():
            # Getting the house and the HVAC
            house_id = self.hvacs_id_registry[hvac_id]
            house = self.houses[house_id]
            hvac = house.hvacs[hvac_id]
            if hvac_id in actions_dict.keys():
                command = actions_dict[hvac_id]
            else:
                warnings.warn(
                    "HVAC {} in house {} did not receive any command.".format(
                        hvac_id, house_id
                    )
                )
                command = False
            hvac.step(command)

        # Update houses' temperatures
        for house_id in self.houses.keys():
            house = self.houses[house_id]
            house.step(self.current_OD_temp, time_step, date_time)

        # Update outdoors temperature
        self.current_OD_temp = self.compute_OD_temp(date_time)
        ## Observations
        cluster_obs_dict = self.make_cluster_obs_dict(date_time)

        ## Temperature penalties and total cluster power consumption
        temp_penalty_dict = {}
        self.cluster_hvac_power = 0

        for hvac_id in self.hvacs_id_registry.keys():
            # Getting the house and the HVAC
            house_id = self.hvacs_id_registry[hvac_id]
            house = self.houses[house_id]
            hvac = house.hvacs[hvac_id]

            # Temperature penalties
            temp_penalty_dict[hvac.id] = compute_temp_penalty(
                house.target_temp, house.deadband, house.current_temp
            )

            # Cluster hvac power consumption
            self.cluster_hvac_power += hvac.power_consumption()

        # Info
        info_dict = {}  # Not necessary for the moment

        return cluster_obs_dict, temp_penalty_dict, self.cluster_hvac_power, info_dict

    def compute_OD_temp(self, date_time) -> float:
        """
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
        delay = -6  # Temperature is coldest at 6am
        time_day = date_time.hour + date_time.minute / 60.0

        temperature = amplitude * np.sin(2 * np.pi * (time_day + delay) / 24) + bias

        # Adding noise
        temperature += random.gauss(0, self.temp_std)

        return temperature


class PowerGrid(object):
    """
    Simulated power grid outputting the regulation signal.

    Attributes:
    avg_power_per_hvac: float, average power to be given per HVAC, in Watts
    signal_mode: string, mode of variation in the signal (can be none or sinusoidal)
    signal_params: dictionary, parameters of the variation of the signal
    nb_hvac: int, number of HVACs in the cluster
    init_signal: float, initial signal value per HVAC, in Watts
    current_signal: float, current signal in Watts

    Functions:
    step(self, date_time): Computes the regulation signal at given date and time
    """

    def __init__(self, power_grid_prop, nb_hvacs):
        """
        Initialize PowerGrid.

        Returns: -

        Parameters:
        power_grid_prop: dictionary, containing the configuration properties of the power grid
        nb_hvacs: int, number of HVACs in the cluster
        """
        self.avg_power_per_hvac = power_grid_prop["avg_power_per_hvac"]
        self.signal_mode = power_grid_prop["signal_mode"]
        self.signal_params = power_grid_prop["signal_parameters"][self.signal_mode]
        self.nb_hvac = nb_hvacs
        self.init_signal_per_hvac = power_grid_prop["init_signal_per_hvac"]
        self.current_signal = self.init_signal_per_hvac * self.nb_hvac

    def step(self, date_time) -> float:
        """
        Compute the regulation signal at given date and time

        Returns:
        current_signal: Current regulation signal in Watts

        Parameters:
        self
        date_time: datetime, current date and time
        """
        if self.signal_mode == "flat":
            self.current_signal = self.init_signal_per_hvac * self.nb_hvac

        elif self.signal_mode == "sinusoidals":
            """Compute the outdoors temperature based on the time, being the sum of several sinusoidal signals"""
            amplitudes = [
                self.nb_hvac * self.avg_power_per_hvac * ratio
                for ratio in self.signal_params["amplitude_ratios"]
            ]
            periods = self.signal_params["periods"]
            if len(periods) != len(amplitudes):
                raise ValueError(
                    "Power grid signal parameters: periods and amplitude_ratios lists should have the same length. Change it in the config.py file. len(periods): {}, leng(amplitude_ratios): {}.".format(
                        len(periods), len(amplitude_ratios)
                    )
                )

            bias = self.avg_power_per_hvac * self.nb_hvac

            time_sec = date_time.hour * 3600 + date_time.minute * 60 + date_time.second

            signal = bias
            for i in range(len(periods)):
                signal += amplitudes[i] * np.sin(2 * np.pi * time_sec / periods[i])
            self.current_signal = signal

        elif self.signal_mode == "regular_steps":
            """Compute the outdoors temperature based on the time, being the sum of several rectangular signals"""
            ratios = self.signal_params["ratios"]
            amplitudes = [
                self.avg_power_per_hvac / len(ratios) * self.nb_hvac / ratio
                for ratio in ratios
            ]
            periods = self.signal_params["periods"]

            if len(periods) != len(ratios):
                raise ValueError(
                    "Power grid signal parameters: periods and ratios lists should have the same length. Change it in the config.py file. len(periods): {}, leng(ratios): {}.".format(
                        len(periods), len(ratios)
                    )
                )

            signal = 0
            time_sec = date_time.hour * 3600 + date_time.minute * 60 + date_time.second

            for i in range(len(periods)):
                signal += amplitudes[i] * np.heaviside(
                    (time_sec % periods[i]) - (1 - ratios[i]) * periods[i], 1
                )
            self.current_signal = signal

        else:
            raise ValueError(
                "Invalid power grid signal mode: {}. Change value in the config file.".format(
                    self.signal_mode
                )
            )
        return self.current_signal
