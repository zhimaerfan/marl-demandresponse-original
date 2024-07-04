from utils import normStateDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
# Efan's
from torch.distributions import Normal
import os
from agents.network import Actor, DQN_network, DDPG_Network, TarMAC_Actor
import sys
import numpy as np
sys.path.append("..")


class PPOAgent():
    def __init__(self, agent_properties, config_dict, num_state=22, num_action=2):
        super(PPOAgent, self).__init__()
        self.id = agent_properties["id"]
        self.actor_name = agent_properties["actor_name"]
        self.actor_path = os.path.join(".", "actors", self.actor_name)
        self.config_dict = config_dict

        self.seed = agent_properties["net_seed"]
        torch.manual_seed(self.seed)
        self.hvac_actor_net = Actor(num_state=num_state, num_action=num_action, layers = config_dict["PPO_prop"]["actor_layers"])
        self.hvac_actor_net.load_state_dict(torch.load(os.path.join(self.actor_path, 'actor.pth'), map_location=torch.device('cpu')))
        self.hvac_actor_net.eval()


    def act(self, obs_dict):
        obs_dict = obs_dict[self.id]
        state = normStateDict(obs_dict, self.config_dict)
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.hvac_actor_net(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item()

class DQNAgent():
    def __init__(self, agent_properties, config_dict, num_state=22, num_action=2):
        super(DQNAgent, self).__init__()
        self.id = agent_properties["id"]
        self.agent_name = agent_properties["actor_name"]
        self.agent_path = os.path.join(".", "actors", self.agent_name)
        self.config_dict = config_dict

        self.seed = agent_properties["net_seed"]
        torch.manual_seed(self.seed)
        self.DQN_net = DQN_network(num_state=num_state, num_action=num_action, layers = config_dict["DQN_prop"]["network_layers"])
        self.DQN_net.load_state_dict(torch.load(os.path.join(self.agent_path, 'DQN.pth')))
        self.DQN_net.eval()


    def act(self, obs_dict):
        obs_dict = obs_dict[self.id]
        state = normStateDict(obs_dict, self.config_dict)
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            qs = self.DQN_net(state)
        action = torch.argmax(qs).item()
        return action

class DDPGAgent():
    def __init__(self, agent_properties, config_dict, num_state=22, num_action=2):
        super(DDPGAgent, self).__init__()
        self.id = agent_properties["id"]
        self.agent_name = agent_properties["actor_name"]
        self.agent_path = os.path.join(".", "actors", self.agent_name)
        self.config_dict = config_dict

        self.seed = agent_properties["net_seed"]
        torch.manual_seed(self.seed)
        self.DDPG_net = DDPG_Network(in_dim=num_state, out_dim=num_action, hidden_dim = config_dict["DDPG_prop"]["actor_hidden_dim"])
        self.DDPG_net.load_state_dict(torch.load(os.path.join(self.agent_path, 'DDPG.pth')))
        self.DDPG_net.eval()


    def act(self, obs_dict):
        obs_dict = obs_dict[self.id]
        state = normStateDict(obs_dict, self.config_dict)
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            qs = self.DDPG_net(state)
        action = torch.argmax(qs).item()
        return action

class TarmacPPOAgent():
    def __init__(self, agent_properties, config_dict, num_obs_hvac=22, num_obs_station=22, num_hvac_action=2, num_station_action=2):
        super(TarmacPPOAgent, self).__init__()
        self.actor_name = agent_properties["actor_name"]
        self.actor_path = os.path.join(".", "actors", self.actor_name)
        self.config_dict = config_dict

        self.actor_hidden_state_size = config_dict["TarMAC_PPO_prop"]['actor_hidden_state_size']
        self.communication_size = config_dict["TarMAC_PPO_prop"]['communication_size']
        self.key_size = config_dict["TarMAC_PPO_prop"]['key_size']
        self.comm_num_hops = config_dict["TarMAC_PPO_prop"]['comm_num_hops']
        self.critic_hidden_layer_size = config_dict["TarMAC_PPO_prop"]['critic_hidden_layer_size']
        self.with_gru = config_dict["TarMAC_PPO_prop"]['with_gru']
        self.with_comm = config_dict["TarMAC_PPO_prop"]['with_comm']
        self.number_agents_comm = config_dict["TarMAC_PPO_prop"]['number_agents_comm_tarmac']
        self.comm_mode = config_dict["TarMAC_PPO_prop"]['tarmac_comm_mode']       
        
        # Efan's
        self.hvac_nb_agents = config_dict["default_env_prop"]["cluster_prop"]["hvac_nb_agents"]
        self.station_nb_agents = config_dict["default_env_prop"]["cluster_prop"]["station_nb_agents"]
        self.all_nb_agents = self.hvac_nb_agents + self.station_nb_agents
        self.all_agent_ids = config_dict["default_env_prop"]["cluster_prop"]["all_agent_ids"]

        self.seed = agent_properties["net_seed"]
        torch.manual_seed(self.seed)
        self.hvac_actor_net = TarMAC_Actor(num_obs_hvac=num_obs_hvac, num_obs_station=num_obs_station, num_key=self.key_size, num_value=self.communication_size, hidden_state_size = self.actor_hidden_state_size, num_hvac_action=num_hvac_action, num_station_action=num_station_action, number_agents_comm=self.number_agents_comm, comm_mode=self.comm_mode, device=torch.device('cpu'), num_hops=self.comm_num_hops, with_gru=self.with_gru, with_comm=self.with_comm)
        self.ev_actor_net = TarMAC_Actor(num_obs_hvac=num_obs_hvac, num_obs_station=num_obs_station, num_key=self.key_size, num_value=self.communication_size, hidden_state_size = self.actor_hidden_state_size, num_hvac_action=num_hvac_action, num_station_action=num_station_action, number_agents_comm=self.number_agents_comm, comm_mode=self.comm_mode, device=torch.device('cpu'), num_hops=self.comm_num_hops, with_gru=self.with_gru, with_comm=self.with_comm)
        # self.hvac_actor_net = TarMAC_Actor(num_obs=self.num_state, num_key=self.key_size, num_value=self.communication_size, hidden_state_size=self.actor_hidden_state_size, num_action=num_action, number_agents_comm=self.number_agents_comm, comm_mode=self.comm_mode, device=torch.device('cpu'), num_hops=self.comm_num_hops, with_comm=self.with_comm)
        if self.hvac_nb_agents > 0:
            self.hvac_actor_net.load_state_dict(torch.load(os.path.join(self.actor_path, 'hvac_actor.pth'), map_location=torch.device('cpu')))
            self.hvac_actor_net.eval()
        if self.station_nb_agents > 0:
            self.ev_actor_net.load_state_dict(torch.load(os.path.join(self.actor_path, 'ev_actor.pth'), map_location=torch.device('cpu')))
            self.ev_actor_net.eval()

    def act(self, obs_dict):
        obs_all = np.array([normStateDict(obs_dict[k], self.config_dict) for k in obs_dict.keys()]) 
        obs_all = torch.from_numpy(obs_all).float().unsqueeze(0)

        with torch.no_grad():
            hvac_keys = [key for key in obs_dict.keys() if isinstance(key, int)]
            ev_keys = [key for key in obs_dict.keys() if isinstance(key, str) and 'charging_station' in key]

            # 获取所有智能体的索引
            all_keys = list(obs_dict.keys())
            hvac_indices = [all_keys.index(key) for key in hvac_keys]  # 智能体中对应位置的序号
            ev_indices = [all_keys.index(key) for key in ev_keys]
            if self.hvac_nb_agents > 0:
                action_probs, _,  _ = self.hvac_actor_net(obs_all[:, hvac_indices, :], hvac_keys)
                action_probs = action_probs.squeeze(0)  # if isinstance(action_probs, torch.Tensor):  action_probs = action_probs.squeeze(0)可能没有HVAC智能体,则action_probs为[],否则为torch.Tensor
            if self.station_nb_agents > 0:
                _, means, stds = self.ev_actor_net(obs_all[:, ev_indices, :], ev_keys)
                means = means.squeeze(0)
                stds = stds.squeeze(0)

        discrete_actions, discrete_action_probs, continuous_actions, continuous_action_log_probs, continuous_means, continuous_stds = [],[],[],[],[],[]
        hvac_index = 0  # Track index for HVAC agents in action_probs
        ev_index = 0  # Track index for EV charging stations in means and stds

        # 遍历所有智能体根据类型选择动作
        for agent_id in obs_dict.keys():
            # 离散动作的处理（HVAC）
            if isinstance(agent_id, int):
                # 使用unsqueeze(0)增加一个批量维度
                action_prob = action_probs[hvac_index].unsqueeze(0).cpu()  # 应该action_prob.dim() == 2, 是二维的，表示多个动作的概率分布，即使只有一个智能体，它也应该有一个批次维度。
                # action_prob.shape=torch.Size([1, 2]),而action_prob=tensor([[0.52705, 0.47295]])
                c = Categorical(probs=action_prob)
                action = c.sample()
                # action.shape=torch.Size([1, 1]),而action=tensor([[1]])
                discrete_actions.append(action.numpy())  # 加action.numpy().reshape(1, -1)?, Reshape to (1, 1), 但似乎是多余的
                hvac_index += 1
            # 连续动作的处理（EV充电站）
            elif 'charging_station' in agent_id:
                mean = means[ev_index].unsqueeze(0) # 调整索引，因为means和stds仅适用于EV
                std = stds[ev_index].unsqueeze(0)
                normal_dist = Normal(mean, std)
                action = normal_dist.sample()
                continuous_actions.append(action.cpu().numpy())
                ev_index += 1
        discrete_actions = np.array(discrete_actions).reshape(-1, 1) if discrete_actions else np.array([])
        continuous_actions = np.array(continuous_actions).reshape(-1, 2) if continuous_actions else np.array([])
        return discrete_actions, continuous_actions

        # return actions_np
