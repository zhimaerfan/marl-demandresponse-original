from utils import normStateDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
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
        self.actor_net = Actor(num_state=num_state, num_action=num_action, layers = config_dict["PPO_prop"]["actor_layers"])
        self.actor_net.load_state_dict(torch.load(os.path.join(self.actor_path, 'actor.pth'), map_location=torch.device('cpu')))
        self.actor_net.eval()


    def act(self, obs_dict):
        obs_dict = obs_dict[self.id]
        state = normStateDict(obs_dict, self.config_dict)
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state)
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
    def __init__(self, agent_properties, config_dict, num_state=11, num_action=2):
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
        self.num_state = num_state 

        self.seed = agent_properties["net_seed"]
        torch.manual_seed(self.seed)
        self.actor_net = TarMAC_Actor(num_obs=self.num_state, num_key=self.key_size, num_value=self.communication_size, hidden_state_size=self.actor_hidden_state_size, num_action=num_action, number_agents_comm=self.number_agents_comm, comm_mode=self.comm_mode, device=torch.device('cpu'), num_hops=self.comm_num_hops, with_comm=self.with_comm)
        self.actor_net.load_state_dict(torch.load(os.path.join(self.actor_path, 'actor.pth'), map_location=torch.device('cpu')))
        self.actor_net.eval()

    def act(self, obs_dict):
        obs_all = np.array([normStateDict(obs_dict[k], self.config_dict) for k in obs_dict.keys()]) 
        obs_all = torch.from_numpy(obs_all).float().unsqueeze(0)

        with torch.no_grad():
            action_prob = self.actor_net(obs_all).squeeze(0)

        c = Categorical(action_prob.cpu()) 
        actions = c.sample()        #(Agents, 1)
        actions_np = actions.numpy()


        return actions_np
