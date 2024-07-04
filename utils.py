#%% Imports

import numpy as np
import os
import random
import torch
import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise

from copy import deepcopy
from datetime import datetime, timedelta, time

from wandb_setup import wandb_setup

#%% Functions


def render_and_wandb_init(opt, config_dict):
    render = opt.render
    log_wandb = not opt.no_wandb
    wandb_run = None
    if log_wandb:
        wandb_run = wandb_setup(opt, config_dict)
    return render, log_wandb, wandb_run


def adjust_config_train(opt, config_dict):
    """Changes configuration of config_dict based on args."""

    print("Configuration elements changed by the CLI:")
### Environment
    print(" -- General environment properties --")
    if opt.hvac_nb_agents != -1:
        config_dict["default_env_prop"]["cluster_prop"]["hvac_nb_agents"] = opt.hvac_nb_agents
        print("Setting hvac_nb_agents to {}".format(opt.hvac_nb_agents))
    if opt.station_nb_agents != -1:
        config_dict["default_env_prop"]["cluster_prop"]["station_nb_agents"] = opt.station_nb_agents
        config_dict["default_ev_prop"]["num_stations"] = opt.station_nb_agents  # 以cli中的为准,覆盖掉充电桩数,现阶段充电桩数就是智能体数
        print("Setting station_nb_agents to {}".format(opt.station_nb_agents))
    else:
        config_dict["default_ev_prop"]["num_stations"] = config_dict["default_env_prop"]["cluster_prop"]["station_nb_agents"]
    if opt.time_step != -1:
        config_dict["default_env_prop"]["time_step"] = opt.time_step
        print("Setting time_step to {}".format(opt.time_step))
    if config_dict["default_ev_prop"]["num_charging_events"] == -1:
        config_dict["default_ev_prop"]["num_charging_events"] = int(config_dict["default_env_prop"]["cluster_prop"]["station_nb_agents"] * 7 * 24 / config_dict["default_ev_prop"]["mean_park"] * config_dict["default_ev_prop"]["alpha_num_events"])
    if config_dict["default_env_prop"]["start_real_date"] == -1:
        config_dict["default_env_prop"]["start_real_date"] = datetime.now().strftime("%Y%m%d-%H:%M:%S")
    # Efan 更新变压器的最大功率和额定功率,现在还没用上
    num_stations = config_dict["default_ev_prop"]["num_stations"]  # 已经在adjust中更新过该参数,保持与参数station_nb_agents一致
    transformer = config_dict["default_ev_prop"]["infrastructure"]["transformers"][0]
    charging_stations = transformer["charging_stations"]
    if charging_stations[0]["rated_power"] == -1:
        charging_stations[0]["rated_power"] = charging_stations[0]["max_power"] * config_dict["default_ev_prop"]["ratio_p"]
    transformer_max_power = charging_stations[0]["max_power"] * num_stations  # 目前充电桩都一样
    transformer_rated_power = charging_stations[0]["rated_power"] * num_stations
    # 将更新的变压器参数保存到 env_properties 中
    config_dict["default_ev_prop"]["infrastructure"]["transformers"][0]["max_power"] = transformer_max_power
    config_dict["default_ev_prop"]["infrastructure"]["transformers"][0]["rated_power"] = transformer_rated_power


## Reward
    print(" -- Reward properties --")
    if opt.alpha_temp != -1:
        print("Setting alpha_temp to {}".format(opt.alpha_temp))
        config_dict["default_env_prop"]["reward_prop"]["alpha_temp"] = opt.alpha_temp
    if opt.alpha_hvac_active_sig != -1:
        print("Setting alpha_hvac_active_sig to {}".format(opt.alpha_hvac_active_sig))
        config_dict["default_env_prop"]["reward_prop"]["alpha_hvac_active_sig"] = opt.alpha_hvac_active_sig
    if opt.alpha_ev_reactive_sig != -1:
        print("Setting alpha_ev_reactive_sig to {}".format(opt.alpha_ev_reactive_sig))
        config_dict["default_env_prop"]["reward_prop"]["alpha_ev_reactive_sig"] = opt.alpha_ev_reactive_sig
    if opt.temp_penalty_mode != "config":
        print("Setting temp_penalty_mode to {}".format(opt.temp_penalty_mode))
        config_dict["default_env_prop"]["reward_prop"]["temp_penalty_mode"] = opt.temp_penalty_mode
    if opt.alpha_ind_L2 != -1:
        print("Setting alpha_ind_L2 to {}".format(opt.alpha_ind_L2))
        config_dict["default_env_prop"]["reward_prop"]["temp_penalty_parameters"]["mixture"]["alpha_ind_L2"] = opt.alpha_ind_L2
    if opt.alpha_common_L2 != -1:
        print("Setting alpha_common_L2 to {}".format(opt.alpha_common_L2))
        config_dict["default_env_prop"]["reward_prop"]["temp_penalty_parameters"]["mixture"]["alpha_common_L2"] = opt.alpha_common_L2
    if opt.alpha_common_max != -1:
        print("Setting alpha_common_max to {}".format(opt.alpha_common_max))
        config_dict["default_env_prop"]["reward_prop"]["temp_penalty_parameters"]["mixture"]["alpha_common_max"] = opt.alpha_common_max

## Simulator
# Outdoors
    print("-- Outdoors environment --")
    if opt.OD_temp_mode != "config":
        print("Setting OD_temp_mode to {}".format(opt.OD_temp_mode))
        config_dict["default_env_prop"]["cluster_prop"]["temp_mode"] = opt.OD_temp_mode
    config_dict["default_house_prop"]["solar_gain_bool"] = not opt.no_solar_gain
    print("Setting solar_gain_bool to {}".format(not opt.no_solar_gain))
# House and HVAC
    print("-- HVAC properties --")
    if opt.cooling_capacity != -1:
        print("Setting cooling_capacity to {}".format(opt.cooling_capacity))
        config_dict["default_hvac_prop"]["cooling_capacity"] = opt.cooling_capacity
    if opt.lockout_duration != -1:
        print("Setting lockout_duration to {}".format(opt.lockout_duration))
        config_dict["default_hvac_prop"]["lockout_duration"] = opt.lockout_duration
# Noise
    print("-- Noise properties --")
    if opt.house_noise_mode != "config":
        print("Setting house_noise_mode to {}".format(opt.house_noise_mode))
        config_dict["noise_house_prop"]["noise_mode"] = opt.house_noise_mode
    if opt.house_noise_mode_test == "train":
        print("Setting house_noise_mode_test to {}".format(config_dict["noise_house_prop"]["noise_mode"]))
        config_dict["noise_house_prop_test"]["noise_mode"] = config_dict["noise_house_prop"]["noise_mode"]
    else:
        print("Setting house_noise_mode_test to {}".format(opt.house_noise_mode_test))
        config_dict["noise_house_prop_test"]["noise_mode"] = opt.house_noise_mode_test
    if opt.hvac_noise_mode != "config":
        print("Setting hvac_noise_mode to {}".format(opt.hvac_noise_mode))
        config_dict["noise_hvac_prop"]["noise_mode"] = opt.hvac_noise_mode
    if opt.hvac_lockout_noise != -1:
  
        config_dict["default_hvac_prop"]["lockout_noise"] = opt.hvac_lockout_noise 
    if opt.hvac_noise_mode_test == "train":
        print("Setting hvac_noise_mode_test to {}".format(config_dict["noise_hvac_prop_test"]["noise_mode"]))
        config_dict["noise_hvac_prop_test"]["noise_mode"] = config_dict["noise_hvac_prop_test"]["noise_mode"]
    else:
        print("Setting hvac_noise_mode_test to {}".format(opt.hvac_noise_mode_test))
        config_dict["noise_hvac_prop_test"]["noise_mode"] = opt.hvac_noise_mode_test

## Signal
    print("-- Signal --")
    if opt.signal_mode != "config":
        print("Setting signal_mode to {}".format(opt.signal_mode))
        config_dict["default_env_prop"]["power_grid_prop"]["signal_mode"] = opt.signal_mode
    if opt.hvac_base_power_mode != "config":
        print("Setting hvac_base_power_mode to {}".format(opt.hvac_base_power_mode))
        config_dict["default_env_prop"]["power_grid_prop"]["hvac_base_power_mode"] = opt.hvac_base_power_mode
    if opt.active_artificial_ratio != -1:
        config_dict["default_env_prop"]["power_grid_prop"]["active_artificial_ratio"] = opt.active_artificial_ratio
        print("Setting active_artificial_ratio to {}".format(opt.active_artificial_ratio))
    if opt.reactive_artificial_ratio != -1:
        config_dict["default_env_prop"]["power_grid_prop"]["reactive_artificial_ratio"] = opt.reactive_artificial_ratio
        print("Setting reactive_artificial_ratio to {}".format(opt.reactive_artificial_ratio))
    if opt.artificial_active_signal_ratio_range != -1:
        print("Setting artificial_active_signal_ratio_range to {}".format(opt.artificial_active_signal_ratio_range))
        config_dict["default_env_prop"]["power_grid_prop"]["artificial_active_signal_ratio_range"] = opt.artificial_active_signal_ratio_range

    if opt.artificial_reactive_signal_ratio_range != -1:
        print("Setting artificial_reactive_signal_ratio_range to {}".format(opt.artificial_reactive_signal_ratio_range))
        config_dict["default_env_prop"]["power_grid_prop"]["artificial_reactive_signal_ratio_range"] = opt.artificial_reactive_signal_ratio_range

    ## State
    if opt.state_solar_gain != "config":
        print("Setting state solar gain to {}".format(opt.state_solar_gain))
        if opt.state_solar_gain == "True":
            config_dict["default_env_prop"]["state_properties"]["solar_gain"] = True
        elif opt.state_solar_gain == "False":
            config_dict["default_env_prop"]["state_properties"]["solar_gain"] = False
        else:
            raise ValueError("Invalid value for state solar gain")

    if opt.state_hour != "config":
        print("Setting state hour to {}".format(opt.state_hour))
        if opt.state_hour == "True":
            config_dict["default_env_prop"]["state_properties"]["hour"] = True
        elif opt.state_hour == "False":
            config_dict["default_env_prop"]["state_properties"]["hour"] = False
        else:
            raise ValueError("Invalid value for state_hour")


    if opt.state_day != "config":
        print("Setting state day to {}".format(opt.state_day))
        if opt.state_day == "True":
            config_dict["default_env_prop"]["state_properties"]["day"] = True
        elif opt.state_day == "False":
            config_dict["default_env_prop"]["state_properties"]["day"] = False
        else:
            raise ValueError("Invalid value for state_day")

    if opt.state_thermal != "config":
        if opt.state_thermal == "True":
            config_dict["default_env_prop"]["state_properties"]["thermal"] = True
        elif opt.state_thermal == "False":
            config_dict["default_env_prop"]["state_properties"]["thermal"] = False
        else:
            raise ValueError("Invalid value for state_thermal")
    if opt.state_hvac != "config":
        if opt.state_hvac == "True":
            config_dict["default_env_prop"]["state_properties"]["hvac"] = True
        elif opt.state_hvac == "False":
            config_dict["default_env_prop"]["state_properties"]["hvac"] = False
        else:
            raise ValueError("Invalid value for state_hvac")

    if opt.message_thermal != "config":
        if opt.message_thermal == "True":
            config_dict["default_env_prop"]["message_properties"]["thermal"] = True
        elif opt.message_thermal == "False":
            config_dict["default_env_prop"]["message_properties"]["thermal"] = False
        else:
            raise ValueError("Invalid value for message_thermal")
    if opt.message_hvac != "config":
        if opt.message_hvac == "True":
            config_dict["default_env_prop"]["message_properties"]["hvac"] = True
        elif opt.state_hvac == "False":
            config_dict["default_env_prop"]["message_properties"]["hvac"] = False
        else:
            raise ValueError("Invalid value for message_hvac")


    ### Agent

    ## Agent communication constraints
    print("-- Agent communication constraints --")
    if opt.hvac_nb_agents_comm != -1:
        print("Setting hvac_nb_agents_comm to {}".format(opt.hvac_nb_agents_comm))
        config_dict["default_env_prop"]["cluster_prop"]["hvac_nb_agents_comm"] = opt.hvac_nb_agents_comm
    if opt.station_nb_agents_comm != -1:
        print("Setting station_nb_agents_comm to {}".format(opt.station_nb_agents_comm))
        config_dict["default_env_prop"]["cluster_prop"]["station_nb_agents_comm"] = opt.station_nb_agents_comm
    if opt.agents_comm_mode != "config":
        print("Setting agents_comm_mode to {}".format(opt.agents_comm_mode))
        config_dict["default_env_prop"]["cluster_prop"]["agents_comm_mode"] = opt.agents_comm_mode
    if opt.comm_defect_prob != -1:
        print("Setting comm_defect_prob to {}".format(opt.comm_defect_prob))
        config_dict["default_env_prop"]["cluster_prop"]["comm_defect_prob"] = opt.comm_defect_prob
    
    agent = opt.agent_type
    if agent == "ppo":
        print("-- PPO agent --")
    ## PPO agent
    # NN architecture
        if opt.layers_actor != "config":
            print("Setting PPO layers_actor to {}".format(opt.layers_actor))
            config_dict["PPO_prop"]["actor_layers"] = opt.layers_actor
        if opt.layers_critic != "config":
            print("Setting PPO layers_critic to {}".format(opt.layers_critic))
            config_dict["PPO_prop"]["critic_layers"] = opt.layers_critic
        if opt.layers_both != "config":
            print("Setting PPO layers_both to {}".format(opt.layers_both))
            config_dict["PPO_prop"]["actor_layers"] = opt.layers_both
            config_dict["PPO_prop"]["critic_layers"] = opt.layers_both
    # NN optimization
        if opt.batch_size != -1:
            print("Setting PPO batch_size to {}".format(opt.batch_size))
            config_dict["PPO_prop"]["batch_size"] = opt.batch_size
        if opt.lr_critic != -1:
            print("Setting PPO lr_critic to {}".format(opt.lr_critic))
            config_dict["PPO_prop"]["lr_critic"] = opt.lr_critic
        if opt.lr_hvac_actor != -1:
            print("Setting PPO lr_hvac_actor to {}".format(opt.lr_hvac_actor))
            config_dict["PPO_prop"]["lr_hvac_actor"] = opt.lr_hvac_actor
        if opt.lr_both != -1:
            print("Setting PPO lr_both to {}".format(opt.lr_both))
            config_dict["PPO_prop"]["lr_critic"] = opt.lr_both
            config_dict["PPO_prop"]["lr_hvac_actor"] = opt.lr_both
            if opt.lr_hvac_actor != -1 or opt.lr_critic != -1:
                raise ValueError("Potential conflict: both lr_both and lr_hvac_actor or lr_critic were set in the CLI")
    # RL optimization
        if opt.gamma != -1:
            print("Setting PPO gamma to {}".format(opt.gamma))
            config_dict["PPO_prop"]["gamma"] = opt.gamma
        if opt.clip_param != -1:
            print("Setting PPO clip_param to {}".format(opt.clip_param))
            config_dict["PPO_prop"]["clip_param"] = opt.clip_param
        if opt.max_grad_norm != -1:
            print("Setting PPO max_grad_norm to {}".format(opt.max_grad_norm))
            config_dict["PPO_prop"]["max_grad_norm"] = opt.max_grad_norm
        if opt.ppo_update_time != -1:
            print("Setting PPO ppo_update_time to {}".format(opt.ppo_update_time))
            config_dict["PPO_prop"]["ppo_update_time"] = opt.ppo_update_time

    elif agent == "dqn":
        print("-- DQN agent --")
    ## DQN agent
    # NN architecture
        if opt.DQNnetwork_layers != "config":
            print("Setting DQNnetwork_layers to {}".format(opt.DQNnetwork_layers))
            config_dict["DQN_prop"]["network_layers"] = opt.DQNnetwork_layers

    # NN optimization
        if opt.batch_size != -1:
            print("Setting DQN batch_size to {}".format(opt.batch_size))
            config_dict["DQN_prop"]["batch_size"] = opt.batch_size
        if opt.lr != -1:
            print("Setting DQN_lr to {}".format(opt.lr))
            config_dict["DQN_prop"]["lr"] = opt.lr       

    # RL optimization
        if opt.gamma != -1:
            print("Setting DQN gamma to {}".format(opt.gamma))
            config_dict["DQN_prop"]["gamma"] = opt.gamma
        if opt.tau != -1:
            print("Setting DQN tau to {}".format(opt.tau))
            config_dict["DQN_prop"]["tau"] = opt.tau
        if opt.buffer_capacity != -1:
            print("Setting DQN buffer_capacity to {}".format(opt.buffer_capacity))
            config_dict["DQN_prop"]["buffer_capacity"] = opt.buffer_capacity    
        if opt.epsilon_decay != -1:
            print("Setting DQN epsilon_decay to {}".format(opt.epsilon_decay))
            config_dict["DQN_prop"]["epsilon_decay"] = opt.epsilon_decay    
        if opt.min_epsilon != -1:
            print("Setting DQN min_epsilon to {}".format(opt.min_epsilon))
            config_dict["DQN_prop"]["min_epsilon"] = opt.min_epsilon    

    elif agent == "tarmac":
        print("-- TarMAC agent --")
    ## TarMAC agent
        if opt.recurrent_policy == "False":
            print("Setting TarMAC recurrent_policy to False")
            config_dict["TarMAC_prop"]["recurrent_policy"] = False
        if opt.state_size != -1:
            print("Setting TarMAC state_size to {}".format(opt.state_size))
            config_dict["TarMAC_prop"]["state_size"] = opt.state_size
        if opt.communication_size != -1:
            print("Setting TarMAC communication_size to {}".format(opt.communication_size))
            config_dict["TarMAC_prop"]["communication_size"] = opt.communication_size
        if opt.tarmac_communication_mode != "config":
            print("Setting tarmac_communication_mode to {}".format(opt.tarmac_communication_mode))
            config_dict["TarMAC_prop"]["tarmac_communication_mode"] = opt.tarmac_communication_mode
        if opt.comm_num_hops != -1:
            print("Setting TarMAC comm_num_hops to {}".format(opt.comm_num_hops))
            config_dict["TarMAC_prop"]["comm_num_hops"] = opt.comm_num_hops
        if opt.value_loss_coef != -1:
            print("Setting TarMAC value_loss_coef to {}".format(opt.value_loss_coef))
            config_dict["TarMAC_prop"]["value_loss_coef"] = opt.value_loss_coef
        if opt.entropy_coef != -1:
            print("Setting TarMAC entropy_coef to {}".format(opt.entropy_coef))
            config_dict["TarMAC_prop"]["entropy_coef"] = opt.entropy_coef
        if opt.max_grad_norm != -1:
            print("Setting TarMAC max_grad_norm to {}".format(opt.max_grad_norm))
            config_dict["TarMAC_prop"]["tarmac_max_grad_norm"] = opt.max_grad_norm
        if opt.lr != -1:
            print("Setting TarMAC lr to {}".format(opt.lr))
            config_dict["TarMAC_prop"]["lr"] = opt.lr
        if opt.eps != -1:
            print("Setting TarMAC eps to {}".format(opt.eps))
            config_dict["TarMAC_prop"]["tarmac_eps"] = opt.eps
        if opt.gamma != -1:
            print("Setting TarMAC gamma to {}".format(opt.gamma))
            config_dict["TarMAC_prop"]["tarmac_gamma"] = opt.gamma
        if opt.alpha != -1:
            print("Setting TarMAC alpha to {}".format(opt.alpha))
            config_dict["TarMAC_prop"]["tarmac_alpha"] = opt.alpha
        if opt.nb_tarmac_updates != -1:
            print("Setting TarMAC nb_tarmac_updates to {}".format(opt.nb_tarmac_updates))
            config_dict["TarMAC_prop"]["nb_tarmac_updates"] = opt.nb_tarmac_updates
        if opt.batch_size != -1:
            print("Setting TarMAC batch_size to {}".format(opt.batch_size))
            config_dict["TarMAC_prop"]["tarmac_batch_size"] = opt.batch_size

    elif agent == "tarmac_ppo":
        print("-- TarMAC PPO agent --")
    ## TarMAC PPO agent
        if opt.actor_hidden_state_size != -1:
            print("Setting TarMAC actor_hidden_state_size to {}".format(opt.actor_hidden_state_size))
            config_dict["TarMAC_PPO_prop"]["actor_hidden_state_size"] = opt.actor_hidden_state_size
        if opt.communication_size != -1:
            print("Setting TarMAC communication_size to {}".format(opt.communication_size))
            config_dict["TarMAC_PPO_prop"]["communication_size"] = opt.communication_size
        if opt.key_size != -1:
            print("Setting TarMAC key_size to {}".format(opt.key_size))
            config_dict["TarMAC_PPO_prop"]["key_size"] = opt.key_size
        if opt.comm_num_hops != -1:
            print("Setting TarMAC comm_num_hops to {}".format(opt.comm_num_hops))
            config_dict["TarMAC_PPO_prop"]["comm_num_hops"] = opt.comm_num_hops
        if opt.number_agents_comm_tarmac != -1:
            print("Setting TarMAC number_agents_comm_tarmac to {}".format(opt.number_agents_comm_tarmac))
            config_dict["TarMAC_PPO_prop"]["number_agents_comm_tarmac"] = opt.number_agents_comm_tarmac
        if opt.tarmac_comm_mode != "config":
            print("Setting tarmac_comm_mode to {}".format(opt.tarmac_comm_mode))
            config_dict["TarMAC_PPO_prop"]["tarmac_comm_mode"] = opt.tarmac_comm_mode
        if opt.tarmac_comm_mode != "config":
            print("Setting tarmac_comm_mode to {}".format(opt.tarmac_comm_mode))
            config_dict["TarMAC_PPO_prop"]["tarmac_comm_mode"] = opt.tarmac_comm_mode
        if opt.tarmac_comm_defect_prob != -1:
            print("Setting tarmac_comm_defect_prob to {}".format(opt.tarmac_comm_defect_prob))
            config_dict["TarMAC_PPO_prop"]["tarmac_comm_defect_prob"] = opt.tarmac_comm_defect_prob
        if opt.lr_critic != -1:
            print("Setting TarMAC lr_critic to {}".format(opt.lr_critic))
            config_dict["TarMAC_PPO_prop"]["lr_critic"] = opt.lr_critic
        if opt.lr_hvac_actor != -1:
            print("Setting TarMAC lr_hvac_actor to {}".format(opt.lr_hvac_actor))
            config_dict["TarMAC_PPO_prop"]["lr_hvac_actor"] = opt.lr_hvac_actor
        if opt.lr_both != -1:
            print("Setting PPO lr_both to {}".format(opt.lr_both))
            config_dict["TarMAC_PPO_prop"]["lr_critic"] = opt.lr_both
            config_dict["TarMAC_PPO_prop"]["lr_hvac_actor"] = opt.lr_both
            if opt.lr_hvac_actor != -1 or opt.lr_critic != -1:
                raise ValueError("Potential conflict: both lr_both and lr_hvac_actor or lr_critic were set in the CLI")
        if opt.eps != -1:
            print("Setting TarMAC eps to {}".format(opt.eps))
            config_dict["TarMAC_PPO_prop"]["eps"] = opt.eps
        if opt.gamma != -1:
            print("Setting TarMAC gamma to {}".format(opt.gamma))
            config_dict["TarMAC_PPO_prop"]["gamma"] = opt.gamma
        if opt.max_grad_norm != -1:
            print("Setting TarMAC max_grad_norm to {}".format(opt.max_grad_norm))
            config_dict["TarMAC_PPO_prop"]["max_grad_norm"] = opt.max_grad_norm
        if opt.clip_param != -1:
            print("Setting TarMAC clip_param to {}".format(opt.clip_param))
            config_dict["TarMAC_PPO_prop"]["clip_param"] = opt.clip_param
        if opt.ppo_update_time != -1:
            print("Setting TarMAC ppo_update_time to {}".format(opt.ppo_update_time))
            config_dict["TarMAC_PPO_prop"]["ppo_update_time"] = opt.ppo_update_time
        if opt.batch_size != -1:
            print("Setting TarMAC batch_size to {}".format(opt.batch_size))
            config_dict["TarMAC_PPO_prop"]["batch_size"] = opt.batch_size
        if opt.critic_hidden_layer_size != -1:
            print("Setting TarMAC critic_hidden_layer_size to {}".format(opt.critic_hidden_layer_size))
            config_dict["TarMAC_PPO_prop"]["critic_hidden_layer_size"] = opt.critic_hidden_layer_size
        if opt.with_gru != 'config':
            print("Setting TarMAC with_gru to {}".format(opt.with_gru))
            if opt.with_gru == "True":
                config_dict["TarMAC_PPO_prop"]["with_gru"] = True
            else:
                config_dict["TarMAC_PPO_prop"]["with_gru"] = False
        if opt.with_comm != 'config':
            print("Setting TarMAC with_comm to {}".format(opt.with_comm))
            if opt.with_comm == "True":
                config_dict["TarMAC_PPO_prop"]["with_comm"] = True
            else:
                config_dict["TarMAC_PPO_prop"]["with_comm"] = False        

        # 通常用于通信的智能体数量应该小于或等于集群中的总智能体数量。with_comm: 这个条件是一个布尔值，用于确定是否在 TarMAC PPO策略中启用了通信功能。
       ### TEST to avoid running for no reason
        total_agents = config_dict["default_env_prop"]["cluster_prop"]["hvac_nb_agents"] + config_dict["default_env_prop"]["cluster_prop"]["station_nb_agents"]
        if config_dict["TarMAC_PPO_prop"]["number_agents_comm_tarmac"] >= total_agents and config_dict["TarMAC_PPO_prop"]["with_comm"]:
            raise ValueError("number_agents_comm_tarmac {} is greater than or equal to hvac_nb_agents {}".format(config_dict["TarMAC_PPO_prop"]["number_agents_comm_tarmac"], config_dict["default_env_prop"]["cluster_prop"]["hvac_nb_agents"]))

    ### Training process
    if opt.nb_inter_saving_actor != -1:
        print("Setting nb_inter_saving_actor to {}".format(opt.nb_inter_saving_actor))
        config_dict["training_prop"]["nb_inter_saving_actor"] = opt.nb_inter_saving_actor
    if opt.nb_test_logs != -1:
        print("Setting nb_test_logs to {}".format(opt.nb_test_logs))
        config_dict["training_prop"]["nb_test_logs"] = opt.nb_test_logs
    if opt.nb_time_steps_test != -1:
        print("Setting nb_time_steps_test to {}".format(opt.nb_time_steps_test))
        config_dict["training_prop"]["nb_time_steps_test"] = opt.nb_time_steps_test
    if opt.nb_tr_episodes != -1:
        print("Setting nb_tr_episodes to {}".format(opt.nb_tr_episodes))
        config_dict["training_prop"]["nb_tr_episodes"] = opt.nb_tr_episodes
    if opt.nb_tr_epochs != -1:
        print("Setting nb_tr_epochs to {}".format(opt.nb_tr_epochs))
        config_dict["training_prop"]["nb_tr_epochs"] = opt.nb_tr_epochs
    if opt.nb_tr_logs != -1:
        print("Setting nb_tr_logs to {}".format(opt.nb_tr_logs))
        config_dict["training_prop"]["nb_tr_logs"] = opt.nb_tr_logs
    if opt.nb_time_steps != -1:
        print("Setting nb_time_steps to {}".format(opt.nb_time_steps))
        config_dict["training_prop"]["nb_time_steps"] = opt.nb_time_steps

def adjust_config_deploy(opt, config_dict):
    if opt.hvac_nb_agents != -1:
        config_dict["default_env_prop"]["cluster_prop"]["hvac_nb_agents"] = opt.hvac_nb_agents
    if opt.station_nb_agents != -1:
        config_dict["default_env_prop"]["cluster_prop"]["station_nb_agents"] = opt.station_nb_agents
        config_dict["default_ev_prop"]["num_stations"] = opt.station_nb_agents
    else:
        config_dict["default_ev_prop"]["num_stations"] = config_dict["default_env_prop"]["cluster_prop"]["station_nb_agents"]
    if opt.time_step != -1:
        config_dict["default_env_prop"]["time_step"] = opt.time_step
    if config_dict["default_ev_prop"]["num_charging_events"] == -1:
        config_dict["default_ev_prop"]["num_charging_events"] = int(config_dict["default_env_prop"]["cluster_prop"]["station_nb_agents"] * 7 * 24 / config_dict["default_ev_prop"]["mean_park"] * config_dict["default_ev_prop"]["alpha_num_events"])
    if config_dict["default_env_prop"]["start_real_date"] == -1:
        config_dict["default_env_prop"]["start_real_date"] = datetime.now().strftime("%Y%m%d-%H:%M:%S")
    # Efan 更新变压器的最大功率和额定功率,现在还没用上
    num_stations = config_dict["default_ev_prop"]["num_stations"]  # 已经在adjust中更新过该参数,保持与参数station_nb_agents一致
    transformer = config_dict["default_ev_prop"]["infrastructure"]["transformers"][0]
    charging_stations = transformer["charging_stations"]
    if charging_stations[0]["rated_power"] == -1:
        charging_stations[0]["rated_power"] = charging_stations[0]["max_power"] * config_dict["default_ev_prop"]["ratio_p"]
    transformer_max_power = charging_stations[0]["max_power"] * num_stations  # 目前充电桩都一样
    transformer_rated_power = charging_stations[0]["rated_power"] * num_stations
    # 将更新的变压器参数保存到 env_properties 中
    config_dict["default_ev_prop"]["infrastructure"]["transformers"][0]["max_power"] = transformer_max_power
    config_dict["default_ev_prop"]["infrastructure"]["transformers"][0]["rated_power"] = transformer_rated_power


    if opt.cooling_capacity != -1:
        config_dict["default_hvac_prop"]["cooling_capacity"] = opt.cooling_capacity
    if opt.lockout_duration != -1:
        config_dict["default_hvac_prop"]["lockout_duration"] = opt.lockout_duration
    if opt.MPC_rolling_horizon != -1:
        config_dict["MPC_prop"]["rolling_horizon"] = opt.MPC_rolling_horizon
    if opt.signal_mode != "config":
        config_dict["default_env_prop"]["power_grid_prop"][
            "signal_mode"
        ] = opt.signal_mode
    if opt.house_noise_mode != "config":
        config_dict["noise_house_prop"]["noise_mode"] = opt.house_noise_mode
    if opt.hvac_noise_mode != "config":
        config_dict["noise_hvac_prop"]["noise_mode"] = opt.hvac_noise_mode
    if opt.hvac_lockout_noise != -1:
        config_dict["default_hvac_prop"]["lockout_noise"] = opt.hvac_lockout_noise

    if opt.OD_temp_mode != "config":
        config_dict["default_env_prop"]["cluster_prop"]["temp_mode"] = opt.OD_temp_mode
    config_dict["default_house_prop"]["solar_gain_bool"] = not opt.no_solar_gain
    if opt.hvac_base_power_mode != "config":
        config_dict["default_env_prop"]["power_grid_prop"][
            "hvac_base_power_mode"
        ] = opt.hvac_base_power_mode
    if opt.hvac_nb_agents_comm != -1:
        config_dict["default_env_prop"]["cluster_prop"][
            "hvac_nb_agents_comm"
        ] = opt.hvac_nb_agents_comm
    if opt.station_nb_agents_comm != -1:
        config_dict["default_env_prop"]["cluster_prop"][
            "station_nb_agents_comm"
        ] = opt.station_nb_agents_comm
    if opt.agents_comm_mode != "config":
        config_dict["default_env_prop"]["cluster_prop"][
            "agents_comm_mode"
        ] = opt.agents_comm_mode
    if opt.comm_defect_prob != -1:
        print("Setting comm_defect_prob to {}".format(opt.comm_defect_prob))
        config_dict["default_env_prop"]["cluster_prop"]["comm_defect_prob"] = opt.comm_defect_prob
    if opt.layers_actor != "config":
        config_dict["PPO_prop"]["actor_layers"] = opt.layers_actor
    if opt.layers_critic != "config":
        config_dict["PPO_prop"]["critic_layers"] = opt.layers_critic
    if opt.layers_both != "config":
        config_dict["PPO_prop"]["actor_layers"] = opt.layers_both
        config_dict["PPO_prop"]["critic_layers"] = opt.layers_both
    if opt.DQNnetwork_layers != "config":
        config_dict["DQN_prop"]["network_layers"] = opt.DQNnetwork_layers
    if opt.start_datetime_mode != "config":
        config_dict["default_env_prop"]["start_datetime_mode"] = opt.start_datetime_mode


    print("-- TarMAC PPO agent --")
    ## TarMAC PPO agent
    if opt.actor_hidden_state_size != -1:
        print("Setting TarMAC actor_hidden_state_size to {}".format(opt.actor_hidden_state_size))
        config_dict["TarMAC_PPO_prop"]["actor_hidden_state_size"] = opt.actor_hidden_state_size
    if opt.communication_size != -1:
        print("Setting TarMAC communication_size to {}".format(opt.communication_size))
        config_dict["TarMAC_PPO_prop"]["communication_size"] = opt.communication_size
    if opt.key_size != -1:
        print("Setting TarMAC key_size to {}".format(opt.key_size))
        config_dict["TarMAC_PPO_prop"]["key_size"] = opt.key_size
    if opt.comm_num_hops != -1:
        print("Setting TarMAC comm_num_hops to {}".format(opt.comm_num_hops))
        config_dict["TarMAC_PPO_prop"]["comm_num_hops"] = opt.comm_num_hops
    if opt.number_agents_comm_tarmac != -1:
        print("Setting TarMAC number_agents_comm_tarmac to {}".format(opt.number_agents_comm_tarmac))
        config_dict["TarMAC_PPO_prop"]["number_agents_comm_tarmac"] = opt.number_agents_comm_tarmac
    if opt.tarmac_comm_mode != "config":
        print("Setting tarmac_comm_mode to {}".format(opt.tarmac_comm_mode))
        config_dict["TarMAC_PPO_prop"]["tarmac_comm_mode"] = opt.tarmac_comm_mode        
    if opt.tarmac_comm_defect_prob != -1:
        print("Setting tarmac_comm_defect_prob to {}".format(opt.tarmac_comm_defect_prob))
        config_dict["TarMAC_PPO_prop"]["tarmac_comm_defect_prob"] = opt.tarmac_comm_defect_prob
    if opt.critic_hidden_layer_size != -1:
        print("Setting TarMAC critic_hidden_layer_size to {}".format(opt.critic_hidden_layer_size))
        config_dict["TarMAC_PPO_prop"]["critic_hidden_layer_size"] = opt.critic_hidden_layer_size
    if opt.with_gru != 'config':
        print("Setting TarMAC with_gru to {}".format(opt.with_gru))
        if opt.with_gru == "True":
            config_dict["TarMAC_PPO_prop"]["with_gru"] = True
        else:
            config_dict["TarMAC_PPO_prop"]["with_gru"] = False
    if opt.with_comm != 'config':
        print("Setting TarMAC with_comm to {}".format(opt.with_comm))
        if opt.with_comm == "True":
            config_dict["TarMAC_PPO_prop"]["with_comm"] = True
        else:
            config_dict["TarMAC_PPO_prop"]["with_comm"] = False 

    ## State
    print("-- Agent observations --")
    if opt.state_solar_gain != "config":
        print("Setting state solar gain to {}".format(opt.state_solar_gain))
        if opt.state_solar_gain == "True":
            config_dict["default_env_prop"]["state_properties"]["solar_gain"] = True
        elif opt.state_solar_gain == "False":
            config_dict["default_env_prop"]["state_properties"]["solar_gain"] = False
        else:
            raise ValueError("Invalid value for state solar gain")

    if opt.state_hour != "config":
        print("Setting state hour to {}".format(opt.state_hour))
        if opt.state_hour == "True":
            config_dict["default_env_prop"]["state_properties"]["hour"] = True
        elif opt.state_hour == "False":
            config_dict["default_env_prop"]["state_properties"]["hour"] = False
        else:
            raise ValueError("Invalid value for state_hour")


    if opt.state_day != "config":
        print("Setting state day to {}".format(opt.state_day))
        if opt.state_day == "True":
            config_dict["default_env_prop"]["state_properties"]["day"] = True
        elif opt.state_day == "False":
            config_dict["default_env_prop"]["state_properties"]["day"] = False
        else:
            raise ValueError("Invalid value for state_day")

    if opt.state_thermal != "config":
        print("Setting state thermal to {}".format(opt.state_thermal))
        if opt.state_thermal == "True":
            config_dict["default_env_prop"]["state_properties"]["thermal"] = True
        elif opt.state_thermal == "False":
            config_dict["default_env_prop"]["state_properties"]["thermal"] = False
        else:
            raise ValueError("Invalid value for state_day")
    if opt.state_hvac != "config":
        if opt.state_hvac == "True":
            config_dict["default_env_prop"]["state_properties"]["hvac"] = True
        elif opt.state_hvac == "False":
            config_dict["default_env_prop"]["state_properties"]["hvac"] = False
        else:
            raise ValueError("Invalid value for state_day")

    if opt.message_thermal != "config":
        if opt.message_thermal == "True":
            config_dict["default_env_prop"]["message_properties"]["thermal"] = True
        elif opt.message_thermal == "False":
            config_dict["default_env_prop"]["message_properties"]["thermal"] = False
        else:
            raise ValueError("Invalid value for message_thermal")
    if opt.message_hvac != "config":
        if opt.message_hvac == "True":
            config_dict["default_env_prop"]["message_properties"]["hvac"] = True
        elif opt.state_hvac == "False":
            config_dict["default_env_prop"]["message_properties"]["hvac"] = False
        else:
            raise ValueError("Invalid value for message_hvac")

    if opt.active_artificial_ratio != -1:
        config_dict["default_env_prop"]["power_grid_prop"]["active_artificial_ratio"] = opt.active_artificial_ratio
        print("Setting active_artificial_ratio to {}".format(opt.active_artificial_ratio))
    if opt.reactive_artificial_ratio != -1:
        config_dict["default_env_prop"]["power_grid_prop"]["reactive_artificial_ratio"] = opt.reactive_artificial_ratio
        print("Setting reactive_artificial_ratio to {}".format(opt.reactive_artificial_ratio))


# 将噪声应用于环境特性
# Applying noise on environment properties
def applyPropertyNoise(
    # config中的参数,5组:环境的属性(非常多)、房屋建模参数、噪声、空调、屋子随机噪声、空调随机噪声
    default_env_prop,
    default_house_prop,
    noise_house_prop,
    default_hvac_prop,
    noise_hvac_prop,
):

    env_properties = deepcopy(default_env_prop)
    hvac_nb_agents = default_env_prop["cluster_prop"]["hvac_nb_agents"]

    # Creating the houses
    houses_properties = []
    hvac_agent_ids = []
    for i in range(hvac_nb_agents):
        house_prop = deepcopy(default_house_prop)
        apply_house_noise(house_prop, noise_house_prop)
        house_id = i
        house_prop["id"] = house_id
        hvac_prop = deepcopy(default_hvac_prop)
        apply_hvac_noise(hvac_prop, noise_hvac_prop)
        hvac_prop["id"] = house_id
        hvac_agent_ids.append(house_id)
        house_prop["hvac_properties"] = hvac_prop
        houses_properties.append(house_prop)

    env_properties["cluster_prop"]["houses_properties"] = houses_properties
    env_properties["hvac_agent_ids"] = hvac_agent_ids
    env_properties["nb_hvac"] = len(hvac_agent_ids)

    # Setting the date
    if env_properties["start_datetime_mode"] == "random":
        env_properties["start_datetime"] = get_random_date_time(
            datetime.strptime(default_env_prop["start_datetime"], "%Y-%m-%d %H:%M:%S")
        )  # Start date and time (Y,M,D, H, min, s)
    elif env_properties["start_datetime_mode"] == "fixed":
        env_properties["start_datetime"] = datetime.strptime(
            default_env_prop["start_datetime"], "%Y-%m-%d %H:%M:%S"
        )
    else:
        raise ValueError(
            "start_datetime_mode in default_env_prop in config.py must be random or fixed. Current value: {}.".format(
                env_properties["start_datetime_mode"] == "fixed"
            )
        )

    # Efan EV先不处理,还需要deepcopy建立5个不一样的EV?
    return env_properties


# Applying noise on properties
def apply_house_noise(house_prop, noise_house_prop):
    noise_house_mode = noise_house_prop["noise_mode"]
    noise_house_params = noise_house_prop["noise_parameters"][noise_house_mode]

    # Gaussian noise: target temp
    house_prop["init_air_temp"] += np.abs(
        random.gauss(0, noise_house_params["std_start_temp"])
    )
    house_prop["init_mass_temp"] += np.abs(
        random.gauss(0, noise_house_params["std_start_temp"])
    )
    house_prop["target_temp"] += np.abs(
        random.gauss(0, noise_house_params["std_target_temp"])
    )

    # Factor noise: house wall conductance, house thermal mass, air thermal mass, house mass surface conductance

    factor_Ua = random.triangular(
        noise_house_params["factor_thermo_low"],
        noise_house_params["factor_thermo_high"],
        1,
    )  # low, high, mode ->  low <= N <= high, with max prob at mode.
    house_prop["Ua"] *= factor_Ua

    factor_Cm = random.triangular(
        noise_house_params["factor_thermo_low"],
        noise_house_params["factor_thermo_high"],
        1,
    )  # low, high, mode ->  low <= N <= high, with max prob at mode.
    house_prop["Cm"] *= factor_Cm

    factor_Ca = random.triangular(
        noise_house_params["factor_thermo_low"],
        noise_house_params["factor_thermo_high"],
        1,
    )  # low, high, mode ->  low <= N <= high, with max prob at mode.
    house_prop["Ca"] *= factor_Ca

    factor_Hm = random.triangular(
        noise_house_params["factor_thermo_low"],
        noise_house_params["factor_thermo_high"],
        1,
    )  # low, high, mode ->  low <= N <= high, with max prob at mode.
    house_prop["Hm"] *= factor_Hm


def apply_hvac_noise(hvac_prop, noise_hvac_prop):
    noise_hvac_mode = noise_hvac_prop["noise_mode"]
    hvac_capacity = hvac_prop["cooling_capacity"]
    noise_hvac_params = noise_hvac_prop["noise_parameters"][noise_hvac_mode]

    hvac_prop["cooling_capacity"] = random.choices(
        noise_hvac_params["cooling_capacity_list"][hvac_capacity]
    )[0]


"""
    # Gaussian noise: latent_cooling_fraction
    hvac_prop["latent_cooling_fraction"] += random.gauss(
        0, noise_hvac_params["std_latent_cooling_fraction"]
    )

    # Factor noise: COP, cooling_capacity
    factor_COP = random.triangular(
        noise_hvac_params["factor_COP_low"], noise_hvac_params["factor_COP_high"], 1
    )  # low, high, mode ->  low <= N <= high, with max prob at mode.

    hvac_prop["COP"] *= factor_COP

    factor_cooling_capacity = random.triangular(
        noise_hvac_params["factor_cooling_capacity_low"],
        noise_hvac_params["factor_cooling_capacity_high"],
        1,
    )  # low, high, mode ->  low <= N <= high, with max prob at mode.
    hvac_prop["cooling_capacity"] *= factor_cooling_capacity
"""


def get_random_date_time(start_date_time):
    # Gets a uniformly sampled random date and time within a year from the start_date_time
    days_in_year = 364
    seconds_in_day = 60 * 60 * 24
    random_days = random.randrange(days_in_year)
    random_seconds = random.randrange(seconds_in_day)
    random_date = start_date_time + timedelta(days=random_days, seconds=random_seconds)

    return random_date


# Multi agent management
def get_actions(actors, obs_dict):
    if isinstance(actors, dict):            # One actor per agent 
        actions, discrete_actions, continuous_actions = {}, {}, {}
        for hvac_agent_id in actors.keys():
            actions[hvac_agent_id] = actors[hvac_agent_id].act(obs_dict)
        return actions, discrete_actions, continuous_actions
    else:                                   # One actor for all agents (may need to change to ensure decentralized - ex: TarMAC_PPO)
        discrete_actions, continuous_actions = actors.act(obs_dict)
        actions_dict = {}
        discrete_actions_dict = {}
        continuous_actions_dict = {}
        discrete_action_index = 0
        continuous_action_index = 0
        for agent_id in obs_dict.keys():
            if isinstance(agent_id, int):
                actions_dict[agent_id] = discrete_actions[discrete_action_index]
                discrete_actions_dict[agent_id] = discrete_actions[discrete_action_index]
                discrete_action_index += 1
            elif 'charging_station' in agent_id:
                actions_dict[agent_id] = continuous_actions[continuous_action_index]
                continuous_actions_dict[agent_id] = continuous_actions[continuous_action_index]
                continuous_action_index += 1
        return actions_dict, discrete_actions_dict, continuous_actions_dict


def datetime2List(dt):
    return [dt.year, dt.month, dt.day, dt.hour, dt.minute]


def superDict2List(SDict, id):
    tmp = SDict[id].copy()
    tmp["datetime"] = datetime2List(tmp["datetime"])
    for k, v in tmp.items():
        if not isinstance(tmp[k], list):
            tmp[k] = [v]
    return sum(list(tmp.values()), [])

# 定义一个简单的函数来处理除法，如果分母为0则返回0
def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0
    else:
        return numerator / denominator

# Efan's 添加init_state.   注意只是归一化状态.状态的数量由嵌入（Embedding）来编码消息.
def normStateDict(sDict, config_dict, returnDict=False, init_state = False):
    """
    输入len为21,输出len为51. 消息传递了10*5个状态,但会经过计算删掉sDict中的重复部分. 不仅包含了输入字典中的状态，还包括了一些额外计算出的状态和从消息中提取出的状态
    
    对输入的状态字典sDict进行标准化处理，使其值在一个合适的范围内，便于模型的训练。
    函数接受三个参数：状态字典sDict，配置字典config_dict，以及一个布尔值returnDict，用于决定函数返回的是字典还是扁平化的NumPy数组. 

    以下是函数的主要步骤和组件：
    提取默认属性：从config_dict中提取默认的房屋、HVAC和环境属性。
    初始化结果字典：创建一个空字典result，用于存储标准化后的状态值。
    温度和除数键的处理：
        k_temp：包含需要通过减去20并除以5来标准化的温度键。
        k_div：包含需要通过除以其默认值来标准化的键。
    状态值的标准化：
        对于k_temp中的键，执行(sDict[k] - 20) / 5操作。
        对于k_div中的键，根据它们在默认属性字典中的值进行除法操作。
    特殊状态值的处理：
        处理日期和时间相关的状态，如sin_day、cos_day、sin_hr和cos_hr。
        处理太阳能增益、HVAC状态和调节信号。
    消息的处理：
       如果状态字典中包含消息，则对消息中的每个元素进行标准化处理。
    返回结果：
        如果returnDict为True，则返回标准化后的状态字典, 传递的消息作为字典。
        如果returnDict为False，则返回扁平化的NumPy数组，其中包含所有标准化后的状态值, 传递的消息也扁平化。
    """
    default_house_prop = config_dict["default_house_prop"]
    default_hvac_prop = config_dict["default_hvac_prop"]
    default_ev_prop = config_dict["default_ev_prop"]
    default_env_prop = config_dict["default_env_prop"]
    state_prop = default_env_prop["state_properties"]

    # 从EV配置中找出最大电池容量和最大有功功率
    max_battery_capacity = max([vt['battery']['capacity'] for vt in config_dict["default_ev_prop"]["vehicle_types"]])
    mean_soc_diff_energy = default_ev_prop["soc_target"] - default_ev_prop["mean_soc"]
    max_active_power = max([vt['battery']['max_active_power'] for vt in config_dict["default_ev_prop"]["vehicle_types"]])

    result = {}
    # 共有状态
    result["grid_hvac_active_reg_signal"] = safe_divide(sDict["grid_hvac_active_reg_signal"] , (
        default_env_prop["reward_prop"]["norm_active_reg_sig"][0]* default_env_prop["cluster_prop"]["hvac_nb_agents"]))
    result["cluster_hvac_active_power"] = safe_divide(sDict["cluster_hvac_active_power"], (
        default_env_prop["reward_prop"]["norm_active_reg_sig"][0]* default_env_prop["cluster_prop"]["hvac_nb_agents"]))

    result["grid_ev_active_reg_signal"] = safe_divide(sDict["grid_ev_active_reg_signal"] , (
        default_env_prop["reward_prop"]["norm_active_reg_sig"][1] * default_env_prop["cluster_prop"]["station_nb_agents"]))
    result["cluster_ev_active_power"] = safe_divide(sDict["cluster_ev_active_power"], (
        default_env_prop["reward_prop"]["norm_active_reg_sig"][1] * default_env_prop["cluster_prop"]["station_nb_agents"]))
    result["grid_ev_reactive_reg_signal"] = safe_divide(sDict["grid_ev_reactive_reg_signal"] , (
        default_env_prop["reward_prop"]["norm_reactive_reg_sig"] * default_env_prop["cluster_prop"]["station_nb_agents"]))
    result["cluster_reactive_power"] = safe_divide(sDict["cluster_ev_reactive_power"], (
        default_env_prop["reward_prop"]["norm_reactive_reg_sig"] * default_env_prop["cluster_prop"]["station_nb_agents"]))
    # 根据智能体类型进行不同的处理
    if "house_mass_temp" in sDict:  # 如果是HVAC智能体

        k_temp = ["house_temp", "house_mass_temp", "house_target_temp"]
        k_div = ["hvac_cooling_capacity"]

        # 在状态中包含室外温度和室内温度参数。也可在交流信息中包含热参数。
        if state_prop["thermal"]:
            k_temp += ["OD_temp"]
            k_div += [
                "house_Ua",
                "house_Cm",
                "house_Ca",
                "house_Hm",
            ]
        
        if state_prop["hvac"]:
            k_div += [
                "hvac_COP",
                "hvac_latent_cooling_fraction",
            ]
        
        # 温度归一化: 假设温度在15到30度之间。
        # k_lockdown = ['hvac_seconds_since_off', 'hvac_lockout_duration']
        for k in k_temp:
            # Assuming the temperatures will be between 15 to 30, centered around 20 -> between -1 and 2, centered around 0.
            result[k] = (sDict[k] - 20) / 5
        result["house_deadband"] = sDict["house_deadband"]

        # 其他状态归一化: 包括时间、太阳能增益等。
        if state_prop["day"]:
            day = sDict["datetime"].timetuple().tm_yday
            result["sin_day"] = np.sin(day * 2 * np.pi / 365)
            result["cos_day"] = np.cos(day * 2 * np.pi / 365)
        if state_prop["hour"]:
            hour = sDict["datetime"].hour
            result["sin_hr"] = np.sin(hour * 2 * np.pi / 24)
            result["cos_hr"] = np.cos(hour * 2 * np.pi / 24)

        if state_prop["solar_gain"]:
            result["house_solar_gain"] = sDict["house_solar_gain"] / 1000

        for k in k_div:
            k1 = "_".join(k.split("_")[1:])
            if k1 in list(default_house_prop.keys()):
                result[k] = sDict[k] / default_house_prop[k1]
            elif k1 in list(default_hvac_prop.keys()):
                result[k] = sDict[k] / default_hvac_prop[k1]
            else:
                print(k)
                raise Exception("Error Key Matching.")
        result["hvac_turned_on"] = 1 if sDict["hvac_turned_on"] else 0
        result["hvac_lockout"] = 1 if sDict["hvac_lockout"] else 0

        result["hvac_seconds_since_off"] = (
            sDict["hvac_seconds_since_off"] / sDict["hvac_lockout_duration"])
        result["hvac_lockout_duration"] = (
            sDict["hvac_lockout_duration"] / sDict["hvac_lockout_duration"])

    elif "remaining_departure_time" in sDict:  # 如果是EV充电桩智能体

        # EV自己的状态的归一化
        result["battery_capacity"] = sDict["battery_capacity"] / max_battery_capacity # if sDict["battery_capacity"] != 0 else 0  # 留一个布尔值的标志位确定是否有ev
        result["soc_diff_energy"] = sDict["soc_diff_energy"] / max_battery_capacity # if sDict["battery_capacity"] != 0 else 0
        result["soc_target_energy"] = sDict["soc_target_energy"] / max_battery_capacity # if sDict["battery_capacity"] != 0 else 0
        result["current_battery_energy"] = sDict["current_battery_energy"] / max_battery_capacity # if sDict["battery_capacity"] != 0 else 0
        # result["max_ev_active_power"] = sDict["max_ev_active_power"] / max_active_power
        result["remaining_departure_time"] = sDict["remaining_departure_time"] / (default_ev_prop["mean_park"] * 3600)  # if sDict["battery_capacity"] != 0 else 0 # 假设一天内都可调度
        result["remaining_controllable_time"] = sDict["remaining_controllable_time"] / (default_ev_prop["mean_park"] * 3600)  # if sDict["battery_capacity"] != 0 else 0 # 离开前电量不足必须充的时候就不可控了
        result["max_schedulable_reactive_power"] = sDict["max_schedulable_reactive_power"] / max_active_power
        result["current_ev_active_power"] = sDict["current_ev_active_power"] / max_active_power # if sDict["battery_capacity"] != 0 else 0
        result["current_ev_reactive_power"] = sDict["current_ev_reactive_power"] / max_active_power
        
        # 公共信息
        result["ev_queue_count"] = sDict["ev_queue_count"] / default_env_prop["cluster_prop"]["station_nb_agents"]

        # Efan's 需要添加. 是否可控?即以最大功率充电则不可控.


        # 模仿HVAC的日期和时间处理
        if state_prop["day"]:
            day = sDict["datetime"].timetuple().tm_yday
            result["sin_day"] = np.sin(day * 2 * np.pi / 365)
            result["cos_day"] = np.cos(day * 2 * np.pi / 365)

        if state_prop["hour"]:
            hour = sDict["datetime"].hour
            result["sin_hr"] = np.sin(hour * 2 * np.pi / 24)
            result["cos_hr"] = np.cos(hour * 2 * np.pi / 24)

    # 消息归一化: 从其他智能体接收的消息。
    temp_messages = []
    for message in sDict["message"]:
        r_message = {}
        if message["agent_type"] == "HVAC":
            r_message["current_temp_diff_to_target"] = (
                message["current_temp_diff_to_target"] / 5
            )  # Already a difference, only need to normalize like k_temps
            r_message["hvac_seconds_since_off"] = (
                message["hvac_seconds_since_off"] / default_hvac_prop["lockout_duration"]
            )
            r_message["hvac_curr_consumption"] = (
                message["hvac_curr_consumption"]
                / default_env_prop["reward_prop"]["norm_active_reg_sig"][0]
            )
            r_message["hvac_max_consumption"] = (
                message["hvac_max_consumption"]
                / default_env_prop["reward_prop"]["norm_active_reg_sig"][0]
            )

            if config_dict["default_env_prop"]["message_properties"]["thermal"]:
                r_message["house_Ua"] = message["house_Ua"] / default_house_prop["Ua"]
                r_message["house_Cm"] = message["house_Cm"] / default_house_prop["Cm"]
                r_message["house_Ca"] = message["house_Ca"] / default_house_prop["Ca"]
                r_message["house_Hm"] = message["house_Hm"] / default_house_prop["Hm"]
            if config_dict["default_env_prop"]["message_properties"]["hvac"]:
                r_message["hvac_COP"] = message["hvac_COP"] / default_hvac_prop["COP"] 
                r_message["hvac_latent_cooling_fraction"] = message["hvac_latent_cooling_fraction"] / default_hvac_prop["latent_cooling_fraction"] 
                r_message["hvac_cooling_capacity"] = message["hvac_cooling_capacity"] / default_hvac_prop["cooling_capacity"]

        elif message["agent_type"] == "EV":  # 对Tarmac似乎无用, 其直接在状态上训练为信息通讯
            # 注意：确保归一化的分母不为零
            r_message["battery_capacity"] = message["battery_capacity"] / max_battery_capacity # if message["battery_capacity"] != 0 else 0 # 标志位确定是否有ev,就像空调是否锁死的状态一样
            r_message["soc_diff_energy"] = message["soc_diff_energy"] / max_battery_capacity # if message["battery_capacity"] != 0 else 0
            r_message["soc_target_energy"] = message["soc_target_energy"] / max_battery_capacity # if message["battery_capacity"] != 0 else 0
            r_message["current_battery_energy"] = message["current_battery_energy"] / max_battery_capacity # if message["battery_capacity"] != 0 else 0
            # r_message["max_ev_active_power"] = message["max_ev_active_power"] / max_active_power # if message["battery_capacity"] != 0 else 0
            r_message["remaining_departure_time"] = message["remaining_departure_time"] / (default_ev_prop["mean_park"] * 3600) # if message["battery_capacity"] != 0 else 0
            r_message["remaining_controllable_time"] = message["remaining_controllable_time"] / (default_ev_prop["mean_park"] * 3600) # if message["battery_capacity"] != 0 else 0
            r_message["max_schedulable_reactive_power"] = message["max_schedulable_reactive_power"] / max_active_power
            r_message["current_ev_active_power"] = message["current_ev_active_power"] / max_active_power # if message["battery_capacity"] != 0 else 0  # 
            r_message["current_ev_reactive_power"] = message["current_ev_reactive_power"] / max_active_power

        temp_messages.append(r_message)

    temp_self_state = result  # 临时记录除消息外的自身状态,仅用于补全不同智能体状态数
    if returnDict:
        result["message"] = temp_messages
        temp_self_state = result

    # 原来处理逻辑
    # else:  # Flatten the dictionary in a single np_array
    #     flat_messages = []
    #     for message in temp_messages:
    #         flat_message = list(message.values())
    #         flat_messages = flat_messages + flat_message
    #     result = np.array(list(result.values()) + flat_messages)
    else:
        # Efan 从已更新的配置中获取自身最大状态数和消息的最大状态数
        self_state_max_length = config_dict["default_env_prop"]["cluster_prop"]["max_self_num_state"]
        message_max_length = config_dict["default_env_prop"]["cluster_prop"]["max_message_num_state"]
        result = list(result.values())
        while len(result) < self_state_max_length:  # 初始化时=-1则不填充
            result.append(0)  # 使用0填充
        # 补全自身状态和消息状态，以保证状态数量一致
        flat_messages = []
        for message in temp_messages:
            flat_message = list(message.values())
            while len(flat_message) < message_max_length:
                flat_message.append(0)  # 使用0填充
            flat_messages += flat_message

        # 合并自身状态和消息状态
        final_result = result + flat_messages
        # 转换为NumPy数组
        result = np.array(final_result).astype(np.float32)

    if init_state == False:
        return result
    else:
        return result, temp_self_state, temp_messages


#%% Testing


def test_dqn_agent(agent, env, config_dict, opt, tr_time_steps):
    """
    Test dqn agent on an episode of nb_test_timesteps
    """
    env = deepcopy(env)
    cumul_hvac_avg_reward = 0
    cumul_temp_error = 0
    cumul_hvac_active_signal_error = 0

    nb_time_steps_test = config_dict["training_prop"]["nb_time_steps_test"]

    obs_dict = env.reset()
    with torch.no_grad():
        for t in range(nb_time_steps_test):
            action = {
                k: agent.select_action(normStateDict(obs_dict[k], config_dict))
                for k in obs_dict.keys()
            }
            obs_dict, rewards_dict, dones_dict, info_dict = env.step(action)
            for i in range(env.hvac_nb_agents):
                cumul_hvac_avg_reward += rewards_dict[i] / env.hvac_nb_agents
                cumul_temp_error += (
                    np.abs(obs_dict[i]["house_temp"] - obs_dict[i]["house_target_temp"])
                    / env.hvac_nb_agents
                )
                cumul_hvac_active_signal_error += np.abs(
                    obs_dict[i]["grid_hvac_active_reg_signal"] - obs_dict[i]["cluster_hvac_active_power"]
                ) / (env.hvac_nb_agents**2)

    mean_avg_return = cumul_hvac_avg_reward / nb_time_steps_test
    mean_temp_error = cumul_temp_error / nb_time_steps_test
    mean_signal_error = cumul_hvac_active_signal_error / nb_time_steps_test

    return {
        "Mean test return": mean_avg_return,
        "Test mean temperature error": mean_temp_error,
        "Test mean signal error": mean_signal_error,
        "Training steps": tr_time_steps,
    }


def test_ppo_agent(agent, env, config_dict, opt, tr_time_steps):
    """
    Test ppo agent on an episode of nb_test_timesteps, with
    """
    env = deepcopy(env)
    cumul_hvac_avg_reward = 0
    cumul_temp_error = 0
    cumul_hvac_active_signal_error = 0
    obs_dict = env.reset()
    nb_time_steps_test = config_dict["training_prop"]["nb_time_steps_test"]

    with torch.no_grad():
        for t in range(nb_time_steps_test):
            action_and_prob = {
                k: agent.select_action(normStateDict(obs_dict[k], config_dict))
                for k in obs_dict.keys()
            }
            action = {k: action_and_prob[k][0] for k in obs_dict.keys()}
            obs_dict, rewards_dict, dones_dict, info_dict = env.step(action)
            for i in range(env.hvac_nb_agents):
                cumul_hvac_avg_reward += rewards_dict[i] / env.hvac_nb_agents
                cumul_temp_error += (
                    np.abs(obs_dict[i]["house_temp"] - obs_dict[i]["house_target_temp"])
                    / env.hvac_nb_agents
                )
                cumul_hvac_active_signal_error += np.abs(
                    obs_dict[i]["grid_hvac_active_reg_signal"] - obs_dict[i]["cluster_hvac_active_power"]
                ) / (env.hvac_nb_agents**2)
    mean_avg_return = cumul_hvac_avg_reward / nb_time_steps_test
    mean_temp_error = cumul_temp_error / nb_time_steps_test
    mean_signal_error = cumul_hvac_active_signal_error / nb_time_steps_test

    return {
        "Mean test return": mean_avg_return,
        "Test mean temperature error": mean_temp_error,
        "Test mean signal error": mean_signal_error,
        "Training steps": tr_time_steps,
    }

def test_tarmac_ppo_agent(agent, env, config_dict, opt, tr_time_steps):
    """
    Test ppo agent on an episode of nb_test_timesteps, with
    """
    env = deepcopy(env)
    cumul_hvac_avg_reward = 0
    cumul_ev_avg_reward = 0
    cumul_temp_error = 0
    cumul_hvac_active_signal_error = 0
    cumul_ev_active_signal_error = 0
    cumul_ev_reactive_signal_error = 0
    obs_dict = env.reset()
    nb_time_steps_test = config_dict["training_prop"]["nb_time_steps_test"]

    with torch.no_grad():
        for t in range(nb_time_steps_test):

            obs_all = np.array([normStateDict(obs_dict[k], config_dict) for k in obs_dict.keys()]) 

            # actions_and_probs = agent.select_actions(obs_all, agent.all_agent_ids)
            # # action = {k: actions_and_probs[0][k] for k in obs_dict.keys()}  # Efan 这是原来的
            # action = {}
            # action_prob = {}
            # for i, k in enumerate(obs_dict.keys()):
            #     action[k] = actions_and_probs[0][i].flatten()  # 提取动作并转换为一维数组或单个值
            #     action_prob[k] = actions_and_probs[1][i].flatten()  # 提取概率并转换为一维数组或单个值

            discrete_actions, discrete_action_probs, continuous_actions, continuous_action_log_probs, continuous_means, continuous_stds = agent.select_actions(obs_all, agent.all_agent_ids)
            action = {}  # Efan's以后可能需要修改
            action_prob = {}
            continuous_mean = {}
            continuous_std = {}
            discrete_action_index = 0
            continuous_action_index = 0  # 索引用于访问连续动作和对应概率
            for agent_id in obs_dict.keys():
                # 对于HVAC智能体，使用离散动作
                if isinstance(agent_id, int):
                    action[agent_id] = discrete_actions[discrete_action_index].flatten()  # 提取动作并转换为一维数组或单个值
                    action_prob[agent_id] = discrete_action_probs[discrete_action_index].flatten()    # 提取概率并转换为一维数组或单个值
                    discrete_action_index += 1
                elif 'charging_station' in agent_id:
                    action[agent_id] = continuous_actions[continuous_action_index].flatten()  # 提取动作并转换为一维数组
                    # 这里简化了处理，连续动作的“概率”以均值和标准差的形式给出
                    continuous_mean[agent_id] = continuous_means[continuous_action_index].flatten()  # 提取连续动作的均值和标准差
                    continuous_std[agent_id] = continuous_stds[continuous_action_index].flatten()
                    continuous_action_index += 1


            obs_dict, rewards_dict, dones_dict, info_dict = env.step(action)
            for i in range(env.hvac_nb_agents):
                cumul_hvac_avg_reward += rewards_dict[i] / env.hvac_nb_agents
                cumul_temp_error += (
                    np.abs(obs_dict[i]["house_temp"] - obs_dict[i]["house_target_temp"])
                    / env.hvac_nb_agents
                )
                cumul_hvac_active_signal_error += np.abs(
                    obs_dict[i]["grid_hvac_active_reg_signal"] - obs_dict[i]["cluster_hvac_active_power"]
                ) / (env.hvac_nb_agents**2)
           
            for station_id in env.stations_agent_ids:
                cumul_ev_avg_reward += rewards_dict[station_id] / env.station_nb_agents
                cumul_ev_active_signal_error += np.abs(
                    obs_dict[station_id]["grid_ev_active_reg_signal"] - obs_dict[station_id]["cluster_ev_active_power"]
                ) / (env.station_nb_agents**2)
                cumul_ev_reactive_signal_error += np.abs(
                    obs_dict[station_id]["grid_ev_reactive_reg_signal"] - obs_dict[station_id]["cluster_ev_reactive_power"]
                ) / (env.station_nb_agents**2)

    mean_hvac_avg_return = cumul_hvac_avg_reward / nb_time_steps_test if env.hvac_nb_agents > 0 else 0
    mean_ev_avg_return = cumul_ev_avg_reward / nb_time_steps_test if env.station_nb_agents > 0 else 0
    mean_temp_error = cumul_temp_error / nb_time_steps_test if env.hvac_nb_agents > 0 else 0
    mean_hvac_active_signal_error = cumul_hvac_active_signal_error / nb_time_steps_test if env.hvac_nb_agents > 0 else 0
    mean_ev_active_signal_error = cumul_ev_active_signal_error / nb_time_steps_test if env.station_nb_agents > 0 else 0
    mean_ev_reactive_signal_error = cumul_ev_reactive_signal_error / nb_time_steps_test if env.station_nb_agents > 0 else 0

    return {
        "Mean hvac test return": mean_hvac_avg_return,
        "Mean ev test return": mean_ev_avg_return,
        "Test mean temperature error": mean_temp_error,
        "Test mean hvac active signal error": mean_hvac_active_signal_error,
        "Test mean ev active signal error": mean_ev_active_signal_error,
        "Test mean ev reactive signal error": mean_ev_reactive_signal_error,
        "Training steps": tr_time_steps,
    }

def test_tarmac_agent(agent, env, config_dict, opt, tr_time_steps, init_states, init_comms, init_masks):
    "Test tarmac agent on an episode of nb_test_timesteps"
    env = deepcopy(env)
    cumul_hvac_avg_reward = 0
    cumul_temp_error = 0
    cumul_hvac_active_signal_error = 0
    cumul_temp_offset = 0
    obs_dict = env.reset()
    nb_time_steps_test = config_dict["training_prop"]["nb_time_steps_test"]

    obs_shape = normStateDict(obs_dict[0], config_dict).shape       #(obs_size,)
    obs_torch = obs_dict2obs_torch(obs_shape, obs_dict, config_dict) # [1, nb agents, obs_size]

    _, actions, _, states, communications, _ = agent.act(               # Action is a tensor of shape [1, hvac_nb_agents, 1], value is a tensor of shape [1, 1], actions_log_prob is a tensor of shape [1, hvac_nb_agents, 1], 
                obs_torch, init_states,                                         # communication is a tensor of shape [1, hvac_nb_agents, COMMUNICATION_SIZE], states is a tensor of shape [1, hvac_nb_agents, STATE_SIZE],
                init_comms, init_masks,
            )

    actions_dict = actionsAC2actions_dict(actions)  # [1, hvac_nb_agents, 1 (action_size)]
    obs_dict, _, _, _ = env.step(actions_dict)
    obs = obs_dict2obs_torch(obs_shape, obs_dict, config_dict)            # [1, hvac_nb_agents, obs_size]

    with torch.no_grad():
        for t in range(1, nb_time_steps_test):
            _, actions, _, states, communications, _ = agent.act(               # Action is a tensor of shape [1, hvac_nb_agents, 1], value is a tensor of shape [1, 1], actions_log_prob is a tensor of shape [1, hvac_nb_agents, 1], 
                obs, states,                                         # communication is a tensor of shape [1, hvac_nb_agents, COMMUNICATION_SIZE], states is a tensor of shape [1, hvac_nb_agents, STATE_SIZE],
                communications, init_masks,
            )
            actions_dict = actionsAC2actions_dict(actions)  # [1, hvac_nb_agents, 1 (action_size)]
            obs_dict, rewards_dict, _, _ = env.step(actions_dict)
            obs = obs_dict2obs_torch(obs_shape, obs_dict, config_dict)            # [1, hvac_nb_agents, obs_size]

            for i in range(env.hvac_nb_agents):
                cumul_hvac_avg_reward += rewards_dict[i] / env.hvac_nb_agents
                cumul_temp_error += (
                    np.abs(obs_dict[i]["house_temp"] - obs_dict[i]["house_target_temp"])
                    / env.hvac_nb_agents
                )
                cumul_temp_offset += (obs_dict[i]["house_temp"] - obs_dict[i]["house_target_temp"]) / env.hvac_nb_agents
                cumul_hvac_active_signal_error += np.abs(
                    obs_dict[i]["grid_hvac_active_reg_signal"] - obs_dict[i]["cluster_hvac_active_power"]
                ) / (env.hvac_nb_agents**2)

    mean_avg_return = cumul_hvac_avg_reward / nb_time_steps_test
    mean_temp_error = cumul_temp_error / nb_time_steps_test
    mean_temp_offset = cumul_temp_offset / nb_time_steps_test
    mean_signal_error = cumul_hvac_active_signal_error / nb_time_steps_test

    return {
        "Mean test return": mean_avg_return,
        "Test mean temperature error": mean_temp_error,
        "Test mean temperature offset": mean_temp_offset,
        "Test mean signal error": mean_signal_error,
        "Training steps": tr_time_steps,
    }

def testAgentHouseTemperature(
    agent, state, low_temp, high_temp, config_dict, reg_signal
):
    """
    Receives an agent and a given state. Tests the agent probability output for 100 points
    given range of indoors temperature, returning a vector for the probability of True (on).
    """
    temp_range = np.linspace(low_temp, high_temp, num=100)
    prob_on = np.zeros(100)
    for i in range(100):
        temp = temp_range[i]
        state["house_temp"] = temp
        state["grid_hvac_active_reg_signal"] = reg_signal
        norm_state = normStateDict(state, config_dict)
        action, action_prob = agent.select_action(norm_state)
        if not action:  # we want probability of True
            prob_on[i] = 1 - action_prob
        else:
            prob_on[i] = action_prob
    return prob_on

def test_DDQP_agent(agent, env, config_dict, opt, tr_time_steps, init_states, init_comms, init_masks):
    #obs_torch = init_obs

    states = init_states
    communications = init_comms
    masks = init_masks
    obs_shape = normStateDict(obs_dict[0], config_dict).shape       #(obs_size,)
    nb_time_steps_test = config_dict["training_prop"]["nb_time_steps_test"]

    with torch.no_grad():
        for t in range(nb_time_steps_test):
            _, actions, _, states, communications, _ = agent.act(obs_torch, states, communications, masks)
            actions_dict = actionsAC2actions_dict(actions)  # [1, hvac_nb_agents, 1 (action_size)]
            obs_dict, rewards_dict, done_dict, _ = env.step(actions_dict)
            obs_torch = obs_dict2obs_torch(obs_shape, obs_dict, config_dict)            # [1, hvac_nb_agents, obs_size]
            masks = torch.FloatTensor([[0.0] if done_dict[i] else [1.0] for i in range(env.hvac_nb_agents)]).unsqueeze(0)  # [1, hvac_nb_agents, 1]

            for i in range(env.hvac_nb_agents):
                cumul_hvac_avg_reward += rewards_dict[i] / env.hvac_nb_agents
                cumul_temp_error += (
                    np.abs(obs_dict[i]["house_temp"] - obs_dict[i]["house_target_temp"])
                    / env.hvac_nb_agents
                )
                cumul_temp_offset += (obs_dict[i]["house_temp"] - obs_dict[i]["house_target_temp"])/ env.hvac_nb_agents
                cumul_hvac_active_signal_error += np.abs(
                    obs_dict[i]["grid_hvac_active_reg_signal"] - obs_dict[i]["cluster_hvac_active_power"]
                ) / (env.hvac_nb_agents**2)
    mean_avg_return = cumul_hvac_avg_reward / nb_time_steps_test
    mean_temp_error = cumul_temp_error / nb_time_steps_test
    mean_signal_error = cumul_hvac_active_signal_error / nb_time_steps_test
    mean_temp_offset = cumul_temp_offset / nb_time_steps_test

    return {
        "Mean test return": mean_avg_return,
        "Test mean temperature error": mean_temp_error,
        "Test mean signal error": mean_signal_error,
        "Test mean temperature offset": mean_temp_offset,
        "Training steps": tr_time_steps,
    } 

def obs_dict2obs_torch(obs_shape, obs_dict: dict, config_dict: dict) -> np.ndarray:
    obs_np_array = np.empty(obs_shape, dtype=np.float32).reshape(1, -1)
    for key in obs_dict.keys():
        obs = normStateDict(obs_dict[key], config_dict).reshape(1, -1)
        obs_np_array = np.concatenate((obs_np_array, obs), axis=0)
    obs_np_array = obs_np_array[1:,:]
    obs_np_array = np.expand_dims(obs_np_array, axis = 0)
    return torch.from_numpy(obs_np_array).float()

def actionsAC2actions_dict(actions: torch.tensor) -> dict:
    cpu_actions = actions.view(-1,1).cpu().numpy()
    actions_dict = {}
    for i, action in enumerate(cpu_actions):
        actions_dict[i] = action[0]
    return actions_dict

def reward_dict2reward_torch(reward_dict: dict) -> torch.tensor:
    reward_np = np.array(list(reward_dict.values()))
    reward_np_expanded_1 = np.expand_dims(reward_np, axis=1)
    reward_np_expanded_2 = np.expand_dims(reward_np_expanded_1, axis=0) # (1, hvac_nb_agents, 1)
    reward = torch.from_numpy(reward_np_expanded_2).float()
    return reward



def get_agent_test(agent, state, config_dict, reg_signal, low_temp=10, high_temp=30):
    """
    Receives an agent and a given state. Tests the agent output for 100 points
    given a range of indoors temperature, returning a vector of actions.
    """
    temp_range = np.linspace(low_temp, high_temp, num=100)
    actions = np.zeros(100)
    for i in range(100):
        temp = temp_range[i]
        state["house_temp"] = temp
        state["grid_hvac_active_reg_signal"] = reg_signal
        norm_state = normStateDict(state, config_dict)
        action = agent.select_action(norm_state)
        actions[i] = action
    return actions


def saveDDPGDict(agent, path, t=None):
    if not os.path.exists(path):
        os.makedirs(path)
    agents = agent.agents
    network_dict = {}
    for i in range(len(agents)):
        network_dict[i] = {
            "actor_state_dict": agents[i].hvac_actor_net.state_dict(),
            "actor_optimizer_state_dict": agents[i].actor_optimizer.state_dict(),
            "critic_state_dict": agents[i].critic_net.state_dict(),
            "critic_optimizer_state_dict": agents[i].critic_optimizer.state_dict(),
            "tgt_hvac_actor_net_state_dict": agents[i].tgt_hvac_actor_net.state_dict(),
            "tgt_critic_net_state_dict": agents[i].tgt_critic_net.state_dict(),
            "lr_hvac_actor": agents[i].lr_hvac_actor,
            "lr_critic": agents[i].lr_critic,
            "gamma": agents[i].gamma,
            "soft_tau": agents[i].soft_tau,
        }
    if t:
        torch.save(
            network_dict,
            os.path.join(path, "actor" + str(t) + ".pth"),
        )
    else:
        torch.save(network_dict, os.path.join(path, "actor.pth"))


def saveActorNetDict(agent, path, t=None):
    if not os.path.exists(path):
        os.makedirs(path)
    
    if hasattr(agent, 'hvac_actor_net'):
        hvac_actor_net = agent.hvac_actor_net
        if t:
            torch.save(
                hvac_actor_net.state_dict(), os.path.join(path, "hvac_actor" + str(t) + ".pth")
            )
        else:
            torch.save(hvac_actor_net.state_dict(), os.path.join(path, "hvac_actor.pth"))
    else:
        print("Warning: agent does not have hvac_actor_net attribute")

    if hasattr(agent, 'ev_actor_net'):
        ev_actor_net = agent.ev_actor_net
        if t:
            torch.save(
                ev_actor_net.state_dict(), os.path.join(path, "ev_actor" + str(t) + ".pth")
            )
        else:
            torch.save(ev_actor_net.state_dict(), os.path.join(path, "ev_actor.pth"))
    else:
        print("Warning: agent does not have ev_actor_net attribute")

def saveDQNNetDict(agent, path, t=None):
    if not os.path.exists(path):
        os.makedirs(path)
    policy_net = agent.policy_net
    if t:
        torch.save(policy_net.state_dict(), os.path.join(path, "DQN" + str(t) + ".pth"))
    else:
        torch.save(policy_net.state_dict(), os.path.join(path, "DQN.pth"))


def clipInterpolationPoint(point, parameter_dict):
    for key in point.keys():
        values = np.array(parameter_dict[key])
        if point[key] > np.max(values):
            point[key] = np.max(values)
        elif point[key] < np.min(values):
            point[key] = np.min(values)
    return point


def sortDictKeys(point, dict_keys):
    point2 = {}
    for key in dict_keys:
        point2[key] = point[key]
    return point2


class Perlin:
    def __init__(self, amplitude, nb_octaves, octaves_step, period, seed):

        self.amplitude = amplitude
        self.nb_octaves = nb_octaves
        self.octaves_step = octaves_step
        self.period = period

        self.seed = seed

        self.noise_list = []
        for i in range(self.nb_octaves):
            self.noise_list.append(
                PerlinNoise(octaves=2**i * octaves_step, seed=seed)
            )

    def calculate_noise(self, x):
        noise = 0

        for j in range(self.nb_octaves - 1):
            noise += self.noise_list[j].noise(x / self.period) / (2**j)
        noise += self.noise_list[-1].noise(x / self.period) / (2**self.nb_octaves - 1)
        return self.amplitude * noise

    def plot_noise(self, timesteps=500):
        l = []

        for x in range(timesteps):
            noise = self.calculate_noise(x)
            l.append(noise)

        plt.plot(l)
        plt.show()


def deadbandL2(target, deadband, value):
    # target：目标值，代理希望维持的值，如目标温度。
    # deadband：死区范围，代理可以在不受惩罚的情况下偏离目标值的范围。
    # value：实际值，代理当前的值，如实际温度。
    # 返回deadband_L2惩罚
    if target + deadband / 2 < value:
        deadband_L2 = (value - (target + deadband / 2)) ** 2
    elif target - deadband / 2 > value:
        deadband_L2 = ((target - deadband / 2) - value) ** 2
    else:
        deadband_L2 = 0.0

    return deadband_L2


def house_solar_gain(date_time, window_area, shading_coeff):
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
        y = date_time.month + date_time.day / 30 - 1
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

    solar_gain = window_area * shading_coeff * solar_cooling_load
    return solar_gain


def MaskedSoftmax(x, mask, dim=-1):
    # 减去最大值：对于每个1x3子矩阵，我们找到最大值并从所有元素中减去。这一步是为了数值稳定性。
    # torch.max(input, dim, keepdim)：返回指定维度上的最大值。keepdim=True保持输出的维度与输入相同，便于后续操作。
    x = x - torch.max(x, dim=dim, keepdim=True)[0]
    # 先对x中的每个元素应用指数函数。指数函数是softmax运算的关键组成部分，它可以放大输入值之间的差异，使得更大的值变得相对更重要，而较小的值变得更不重要。后若掩码全为1，则在应用exp后,x的值不会改变。
    # "*"运算符用于执行逐元素乘法，也就是说，两个张量的相同位置的元素相乘。不涉及矩阵乘法，只是简单地将位置相对应的元素相乘。与torch.matmul不同
    x = torch.exp(x) * mask
    # 将每个1x3子矩阵中的元素除以该子矩阵的总和，确保每个向量的元素和为1。可以有多行向量
    x = x / torch.sum(x, dim=dim, keepdim=True)
    # 将NaN值设为0可以防止计算错误
    x[torch.isnan(x)] = 0
    return x

    # 如其中mask:
    # tensor([[1, 1, 0, 1],
    #         [1, 1, 1, 0],
    #         [0, 1, 1, 1],
    #         [1, 0, 1, 1]], device='cuda:0')

    # torch.sum(x, dim=dim, keepdim=True)
    # tensor([[[2.9956],
    #          [2.9928],
    #          [2.9982],
    #          [2.9974]]], device='cuda:0')

    # x = x / torch.sum(x, dim=dim, keepdim=True)
    # tensor([[[0.3330, 0.3331, 0.0000, 0.3338],
    #          [0.3336, 0.3332, 0.3332, 0.0000],
    #          [0.0000, 0.3330, 0.3335, 0.3335],
    #          [0.3328, 0.0000, 0.3336, 0.3336]]], device='cuda:0')