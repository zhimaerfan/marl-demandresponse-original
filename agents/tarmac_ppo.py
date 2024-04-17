import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.distributions import Categorical 

# Efan's 还需能处理连续动作:
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler 
import os 
from time import perf_counter
import wandb 
import numpy as np 
from agents.network import TarMAC_Actor, TarMAC_Critic
import pprint
 
class TarMAC_PPO: 
    def __init__(self, config_dict, opt, num_obs_hvac=22, num_obs_station=22, num_hvac_action=2, num_station_action=2, wandb_run=None): 
        super(TarMAC_PPO, self).__init__() 
        self.seed = opt.net_seed 
        torch.manual_seed(self.seed) 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.actor_hidden_state_size = config_dict["TarMAC_PPO_prop"]['actor_hidden_state_size']
        self.communication_size = config_dict["TarMAC_PPO_prop"]['communication_size']
        self.key_size = config_dict["TarMAC_PPO_prop"]['key_size']
        self.comm_num_hops = config_dict["TarMAC_PPO_prop"]['comm_num_hops']
        self.critic_hidden_layer_size = config_dict["TarMAC_PPO_prop"]['critic_hidden_layer_size']
        self.with_gru = config_dict["TarMAC_PPO_prop"]['with_gru']
        # Efan's
        self.hvac_nb_agents = config_dict["default_env_prop"]["cluster_prop"]["hvac_nb_agents"]
        self.station_nb_agents = config_dict["default_env_prop"]["cluster_prop"]["station_nb_agents"]
        self.all_nb_agents = self.hvac_nb_agents + self.station_nb_agents
        self.all_agent_ids = config_dict["default_env_prop"]["cluster_prop"]["all_agent_ids"]

        self.with_comm = config_dict["TarMAC_PPO_prop"]['with_comm']
        self.number_agents_comm = config_dict["TarMAC_PPO_prop"]['number_agents_comm_tarmac']
        self.comm_mode = config_dict["TarMAC_PPO_prop"]['tarmac_comm_mode']
        self.comm_defect_prob = config_dict["TarMAC_PPO_prop"]['tarmac_comm_defect_prob']

        # if True: 
        #    self.actor_net = OldActor(num_state=num_state, num_action=num_action) 
        #    self.critic_net = OldCritic(num_state=num_state) 
        
        # .to(self.device)这部分代码的作用是将创建的网络实例移动到指定的计算设备上。通过调用.to()方法，你可以确保网络的所有参数和后续的计算都会在这个指定的设备上执行。
        # 定义网络结构和它的运行方式。GPT:这些参数包括观察空间的大小、用于通信的键和值的大小、隐藏状态的大小、动作空间的大小、参与通信的智能体数量、通信模式、通信故障概率、计算设备、通信的跳数、是否使用GRU层和是否开启通信等信息。
        # 怎么处理num_action?
        self.actor_net = TarMAC_Actor(num_obs_hvac=num_obs_hvac, num_obs_station=num_obs_station, num_key=self.key_size, num_value=self.communication_size, hidden_state_size = self.actor_hidden_state_size, num_hvac_action=num_hvac_action, num_station_action=num_station_action, number_agents_comm=self.number_agents_comm, comm_mode=self.comm_mode, comm_defect_prob = self.comm_defect_prob, device=self.device, num_hops=self.comm_num_hops, with_gru=self.with_gru, with_comm=self.with_comm).to(self.device)
        # self.actor_net = TarMAC_Actor(num_obs=num_state, num_key=self.key_size, num_value=self.communication_size, hidden_state_size = self.actor_hidden_state_size, num_action=num_action, number_agents_comm=self.number_agents_comm, comm_mode=self.comm_mode, comm_defect_prob = self.comm_defect_prob, device=self.device, num_hops=self.comm_num_hops, with_gru=self.with_gru, with_comm=self.with_comm).to(self.device) 
        self.critic_net = TarMAC_Critic(num_agents_hvac=self.hvac_nb_agents, num_agents_station=self.station_nb_agents, num_obs_hvac=num_obs_hvac, num_obs_station=num_obs_station, hidden_layer_size=self.critic_hidden_layer_size).to(self.device)
        
        self.batch_size = config_dict["TarMAC_PPO_prop"]["batch_size"] 
        self.ppo_update_time = config_dict["TarMAC_PPO_prop"]["ppo_update_time"] 
        self.max_grad_norm = config_dict["TarMAC_PPO_prop"]["max_grad_norm"] 
        self.clip_param = config_dict["TarMAC_PPO_prop"]["clip_param"] 
        self.gamma = config_dict["TarMAC_PPO_prop"]["gamma"] 
        self.lr_actor = config_dict["TarMAC_PPO_prop"]["lr_actor"] 
        self.lr_critic = config_dict["TarMAC_PPO_prop"]["lr_critic"] 
        self.wandb_run = wandb_run 
        self.log_wandb = not opt.no_wandb 

        # Efan 修改为根据不同id来索引
        # Initialize buffer
        self.buffer = {}
        for agent in self.all_agent_ids:
            self.buffer[agent] = []

        print("-------------------------------para---------------------------------")
        print( 
            "ppo_update_time: {}, max_grad_norm: {}, clip_param: {}, gamma: {}, batch_size: {}, lr_actor: {}, lr_critic: {}".format( 
                self.ppo_update_time, 
                self.max_grad_norm, 
                self.clip_param, 
                self.gamma, 
                self.batch_size, 
                self.lr_actor, 
                self.lr_critic, 
            ) 
        ) 
        self.training_step = 0 
 
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), self.lr_actor) 
        self.critic_net_optimizer = optim.Adam( 
            self.critic_net.parameters(), self.lr_critic 
        ) 
 
    def select_action(self, obs, all_agent_ids): 
        " Select action for one agent given its obs"
        obs = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0).to(self.device) 

        with torch.no_grad(): 
            action_prob, mean, std = self.actor_net(obs, all_agent_ids)   # Efan 需要改
            if isinstance(action_prob, torch.Tensor):  # 可能没有HVAC智能体,则action_probs为[],否则为torch.Tensor
                action_prob = action_prob.squeeze(0)
            if isinstance(mean, torch.Tensor):
                mean = mean.squeeze(0)
            if isinstance(std, torch.Tensor):
                std = std.squeeze(0)
        c = Categorical(action_prob.cpu()) 
        action = c.sample() 
        return action.item(), action_prob[:, action.item()].item()

    def select_actions(self, obs, all_agent_ids):
        """ Select actions for all agents at once, supporting both discrete (HVAC) and continuous (EV charging stations) actions.
            Args:
                obs: Observations for all agents.
                all_agent_ids: Identifiers indicating the type of each agent."""
        # 转换观测数据为张量，准备GPU计算, 注意之前有.unsqueeze(0)
        obs_batch = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)  # (Agents, State dims)如(6, 27),Efan's 必须加.unsqueeze(0),变成(1,6,27)

        # 获取动作概率(HVAC)、均值(EV)和对数标准差(EV)
        with torch.no_grad():
            action_probs, means, stds = self.actor_net(obs_batch, all_agent_ids)  # Efan's必须是批次的obs_batch,以满足后续训练. 概率用不到批次,所以降维
            # action_probs.shape=torch.Size([4, 2]), 而action_probs=tensor([[0.5287, 0.4713],[0.5345, 0.4655],[0.5313, 0.4687],[0.5425, 0.4575]], device='cuda:0'), 对应4个HVAC智能体动作0和1的概率
            # means.shape=torch.Size([2, 2]),而means=tensor([[-0.2666,  0.2328],[-0.2011,  0.2604]], device='cuda:0'), 对应2个EV station智能体的2个动作的均值
            # stds.shape=torch.Size([2, 2]),而stds=tensor([[-0.1046,  0.2614],[-0.0381,  0.3305]], device='cuda:0'),对应2个EV station智能体的2个动作的stds
            if isinstance(action_probs, torch.Tensor):  # 可能没有HVAC智能体,则action_probs为[],否则为torch.Tensor
                action_probs = action_probs.squeeze(0)
            if isinstance(means, torch.Tensor):
                means = means.squeeze(0)
            if isinstance(stds, torch.Tensor):
                stds = stds.squeeze(0)
        discrete_actions, discrete_action_probs, continuous_actions, continuous_action_log_probs, continuous_means, continuous_stds = [],[],[],[],[],[]
        hvac_index = 0  # Track index for HVAC agents in action_probs
        ev_index = 0  # Track index for EV charging stations in means and stds

        # 遍历所有智能体根据类型选择动作
        for agent_id in all_agent_ids:
            # 离散动作的处理（HVAC）
            if isinstance(agent_id, int):
                # 使用unsqueeze(0)增加一个批量维度
                action_prob = action_probs[hvac_index].unsqueeze(0).cpu()  # 应该action_prob.dim() == 2, 是二维的，表示多个动作的概率分布，即使只有一个智能体，它也应该有一个批次维度。
                # action_prob.shape=torch.Size([1, 2]),而action_prob=tensor([[0.52705, 0.47295]])
                c = Categorical(probs=action_prob)
                action = c.sample()
                # action.shape=torch.Size([1, 1]),而action=tensor([[1]])
                discrete_actions.append(action.numpy())  # 加action.numpy().reshape(1, -1)?, Reshape to (1, 1), 但似乎是多余的
                # list类型, discrete_actions=[array([[1]]), array([[1]]), array([[1]]), array([[1]])]
                action_prob_selected = action_prob.gather(1, action).squeeze(-1)
                # action_prob_selected.shape=torch.Size([1]),而action_prob_selected=tensor([0.4658])
                discrete_action_probs.append(action_prob_selected.numpy())  # 添加选择的动作概率
                # discrete_action_probs=[array([0.47238994], dtype=float32), array([0.46517742], dtype=float32), array([0.4686556], dtype=float32), array([0.4657611], dtype=float32)]
                hvac_index += 1
            # 连续动作的处理（EV充电站）
            elif 'charging_station' in agent_id:
                mean = means[ev_index].unsqueeze(0) # 调整索引，因为means和stds仅适用于EV
                # mean.shape=torch.Size([1, 2]), mean=tensor([[-0.1606,  0.1390]], device='cuda:0')
                std = stds[ev_index].unsqueeze(0)
                # 不加.unsqueeze(0):  std.shape=torch.Size([2]), std=tensor([-0.0168,  0.2043], device='cuda:0')
                # 加.unsqueeze(0): std.shape=torch.Size([1, 2]), std=tensor([[-0.0168,  0.2043]], device='cuda:0')
                # std.shape=torch.Size([1, 2]), std=tensor([[0.9834, 1.2267]], device='cuda:0')
                normal_dist = Normal(mean, std)
                action = normal_dist.sample()
                # action.shape=torch.Size([1, 2]),action=tensor([[-0.7841, -0.6499]], device='cuda:0')
                action_log_prob = normal_dist.log_prob(action).sum(dim=-1)  # Sum log probs for all actions

                continuous_actions.append(action.cpu().numpy())
                # list, 而continuous_actions=[array([[-0.4540839,  3.3096793]], dtype=float32), array([[-0.7840779, -0.6499443]], dtype=float32)]
                # 将均值和标准差合并为一个二维数组，每行是一个动作的概率参数
                continuous_means.append(mean.cpu().numpy())
                continuous_stds.append(std.cpu().numpy())
                # list, 而continuous_means_stds=[array([[[-0.15407643,  0.13813038], [ 1.0125877 ,  1.1849842 ]]], dtype=float32), array([[[-0.16062945,  0.13896576], [ 0.98338234,  1.22667   ]]], dtype=float32)]
                continuous_action_log_probs.append(action_log_prob.cpu().numpy())
                ev_index += 1
        # 如果列表非空，转换为Numpy数组；否则返回空数组
        discrete_actions = np.array(discrete_actions).reshape(-1, 1) if discrete_actions else np.array([])
        # discrete_actions.shape=(4, 1), discrete_actions=array([[1],[1],[1],[1]])
        discrete_action_probs = np.array(discrete_action_probs).reshape(-1, 1) if discrete_action_probs else np.array([])
        # discrete_action_probs.shape=(4, 1),discrete_action_probs=array([[0.47238994],[0.46517742],[0.4686556 ],[0.4657611 ]], dtype=float32)
        continuous_actions = np.array(continuous_actions).reshape(-1, 2) if continuous_actions else np.array([])
        # continuous_actions.shape=(2, 2),continuous_actions=array([[-0.53094816,  3.6897826 ], [-0.8189303 , -0.62561333]], dtype=float32)
        continuous_means = np.array(continuous_means).reshape(-1, 2) if continuous_means else np.array([])
        continuous_stds = np.array(continuous_stds).reshape(-1, 2) if continuous_stds else np.array([])
        continuous_action_log_probs = np.array(continuous_action_log_probs).reshape(-1, 1) if continuous_action_log_probs else np.array([])
        return discrete_actions, discrete_action_probs, continuous_actions, continuous_action_log_probs, continuous_means, continuous_stds
        # " Select actions for all agents at once"
        # obs_batch = torch.from_numpy(obs).float().unsqueeze(0).to(self.device) # (1, Agents, State dims)

        # if all_agent_ids == None:  # 原有的HVAC逻辑, 没提供id, 直接处理
        #     with torch.no_grad(): 
        #         action_prob = self.actor_net(obs_batch).squeeze(0)  # (Agents, action dims)
        #     # 创建一个表示分类分布的对象Categorical(probs: torch.Size([4, 2]))
        #     # 注意这里重写了连续动作采样、对数概率、熵等方法:
        #     c = Categorical(action_prob.cpu()) 
        #     actions = c.sample()        #(Agents, 1)
        #     # actions = tensor([[1],
        #     # [1],
        #     # [0],
        #     # [0]])

        #     action_prob = action_prob.cpu().gather(1, actions) # (Agents, 1)
        #     actions_np = actions.numpy()
        #     action_probs_np = action_prob.numpy()
        #     # actions_np=[[1],
        #     # [1],
        #     # [0],
        #     # [0]]
        #     # action_probs_np=[[0.6],
        #     # [0.5],
        #     # [0.7],
        #     # [0.8]]
        #     # 返回每个智能体相应的动作, 即0和1中选一个, 和该动作对应的概率.  同时所有智能体都选, 最终是num_agent个
        #     return actions_np, action_probs_np
        # else:  # 处理异构智能体,连续动作和离散动作
        #     with torch.no_grad():
        #         action_outputs = self.actor_net(obs_batch).squeeze(0)  # (Agents, action_dims)

        #     actions_np = []
        #     action_probs_np = []

        #     for i, agent_id in enumerate(all_agent_ids):
        #         # 根据agent_id前缀判断智能体类型
        #         if isinstance(agent_id, str) and agent_id.startswith('charging_station'):
        #             # EV充电桩智能体处理有功功率的连续动作, 只考虑有功功率的均值和对数标准差
        #             mean = action_outputs[i, 0].unsqueeze(0)  # 均值，确保为1维张量
        #             std = action_outputs[i, 1].unsqueeze(0)  # 对数标准差，确保为1维张量
        #             dist = DiagGaussian_1()
        #             action = dist.sample(mean, std)  # 采样动作
        #             action_prob = dist.log_probs(mean, std, action)  # 计算动作概率
        #         else:
        #             # HVAC智能体处理离散动作
        #             action_prob = torch.softmax(action_outputs[i, :2], dim=-1)
        #             dist = Categorical(action_prob)
        #             action = dist.sample()  # 离散动作采样
        #             action_prob = dist.log_probs(action).exp()  # 将对数概率转换为概率

        #         actions_np.append(action.cpu().numpy())
        #         action_probs_np.append(action_prob.cpu().numpy())

        #     return np.array(actions_np), np.array(action_probs_np)
 
    def get_value(self, state): 

        state = state.to(self.device) 
        state = state.unsqueeze(0) # one-sized "batch" --> (1, num_agents, num_obs)

        with torch.no_grad(): 
            value = self.critic_net(state) 
        return value

    def reset_buffer(self):
        self.buffer = {}
        for agent in self.all_agent_ids:
            self.buffer[agent] = []
 
    def store_transition(self, transition, agent): 
        self.buffer[agent].append(transition) 
 
    def update(self, time_step): 
        """
        # # 修改为直接遍历self.buffer的键，这样可以处理任何形式的键，包括字符串. 为了确定有多少时间步的数据，我用next(iter(self.buffer.values()))来获取self.buffer中第一个元素的值，这假设所有智能体在self.buffer中有相同数量的条目。这是一个安全的假设，只要所有智能体在每个时间步都被更新。如果智能体数量在self.buffer初始化后可能会变化（例如，动态添加或移除智能体），你可能需要调整代码来处理这种情况。
        # state_np = np.array([[self.buffer[agent_id][time_step].state for agent_id in self.buffer.keys()] for time_step in range(len(next(iter(self.buffer.values()))))])
        # next_state_np = np.array([[self.buffer[agent_id][time_step].next_state for agent_id in self.buffer.keys()] for time_step in range(len(next(iter(self.buffer.values()))))])
        # discrete_action_np = np.array([[self.buffer[agent_id][time_step].discrete_action for agent_id in self.buffer.keys()] for time_step in range(len(next(iter(self.buffer.values()))))])
        # old_discrete_action_log_prob_np = np.array([[self.buffer[agent_id][time_step].discrete_a_log_prob for agent_id in self.buffer.keys()] for time_step in range(len(next(iter(self.buffer.values()))))])
        # continuous_action_np = np.array([[self.buffer[agent_id][time_step].continuous_action for agent_id in self.buffer.keys()] for time_step in range(len(next(iter(self.buffer.values()))))])
        # continuous_means_stds_np = np.array([[self.buffer[agent_id][time_step].continuous_means_stds for agent_id in self.buffer.keys()] for time_step in range(len(next(iter(self.buffer.values()))))])
        # reward_np = np.array([[self.buffer[agent_id][time_step].reward for agent_id in self.buffer.keys()] for time_step in range(len(next(iter(self.buffer.values()))))])
        # done_np = np.array([[self.buffer[agent_id][time_step].done for agent_id in self.buffer.keys()] for time_step in range(len(next(iter(self.buffer.values()))))])
        
        # state = torch.tensor(state_np, dtype=torch.float).to(self.device)                                   # (Time steps, Agents, State dim)
        # next_state = torch.tensor(next_state_np, dtype=torch.float).to(self.device)                         # (Time steps, Agents, State dim)
        # discrete_action = torch.tensor(discrete_action_np, dtype=torch.int).to(self.device)                                  # (Time steps, Agents, Action dim)
        # old_discrete_action_log_prob = torch.tensor(old_discrete_action_log_prob_np, dtype=torch.float).to(self.device)       # (Time steps, Agents, Action dim)
        # continuous_action = torch.tensor(continuous_action_np, dtype=torch.float).to(self.device)
        # continuous_means_stds = torch.tensor(continuous_means_stds_np, dtype=torch.float).to(self.device)
        # reward = torch.tensor(reward_np, dtype=torch.float).unsqueeze(2).to(self.device)                     # (Time steps, Agents, 1)
        # done = torch.tensor(done_np, dtype=torch.long).unsqueeze(2).to(self.device)                        # (Time steps, Agents, 1)
        """
        # num_time_steps = len(next(iter(self.buffer.values())))
        
        # # 初始化数组存储
        # states, next_states, rewards, dones = [], [], [], []
        # discrete_actions, discrete_log_probs = [], []
        # continuous_actions, continuous_means_stds = [], []
        
        # for time_step in range(num_time_steps):
        #     states.append([self.buffer[agent_id][time_step].state for agent_id in self.buffer])
        #     next_states.append([self.buffer[agent_id][time_step].next_state for agent_id in self.buffer])
        #     rewards.append([self.buffer[agent_id][time_step].reward for agent_id in self.buffer])
        #     dones.append([self.buffer[agent_id][time_step].done for agent_id in self.buffer])
            
        #     # 分离离散和连续动作及其概率
        #     discrete_actions.append([self.buffer[agent_id][time_step].discrete_action if hasattr(self.buffer[agent_id][time_step], 'discrete_action') else np.nan for agent_id in self.buffer])
        #     discrete_log_probs.append([self.buffer[agent_id][time_step].discrete_a_log_prob if hasattr(self.buffer[agent_id][time_step], 'discrete_a_log_prob') else np.nan for agent_id in self.buffer])
        #     continuous_actions.append([self.buffer[agent_id][time_step].continuous_action if hasattr(self.buffer[agent_id][time_step], 'continuous_action') else np.nan for agent_id in self.buffer])
        #     continuous_means_stds.append([self.buffer[agent_id][time_step].continuous_means_stds if hasattr(self.buffer[agent_id][time_step], 'continuous_means_stds') else np.nan for agent_id in self.buffer])

        # # 将列表转换为NumPy数组，并指定dtype=object处理不同数据类型
        # def safely_convert_to_tensor(data, dtype, default_value=0):
        #     # 尝试找出数组中最大的维度
        #     max_dim1 = 0
        #     max_dim2 = 0
        #     for row in data:
        #         for item in row:
        #             if isinstance(item, np.ndarray):
        #                 max_dim1 = max(max_dim1, item.shape[0])
        #                 if len(item.shape) > 1:
        #                     max_dim2 = max(max_dim2, item.shape[1])

        #     # 根据维度是否存在来决定数组的初始化方式
        #     if max_dim2 > 0:
        #         # 如果数据中有二维数组，创建足够大的二维数组空间
        #         processed_data = np.full((len(data), len(data[0]), max_dim1, max_dim2), default_value, dtype=float)
        #     else:
        #         # 否则创建一维数组空间
        #         processed_data = np.full((len(data), len(data[0]), max_dim1), default_value, dtype=float)

        #     # 填充数据
        #     for i, row in enumerate(data):
        #         for j, item in enumerate(row):
        #             if isinstance(item, np.ndarray):
        #                 if len(item.shape) == 1:
        #                     processed_data[i][j][:len(item)] = item
        #                 elif len(item.shape) == 2:
        #                     processed_data[i][j][:item.shape[0], :item.shape[1]] = item

        #     # 将处理后的NumPy数组转换为Tensor
        #     return torch.tensor(processed_data, dtype=dtype).to(self.device)

        # # 转换所有数据为Tensor
        # state = safely_convert_to_tensor(states, torch.float)
        # next_state = safely_convert_to_tensor(next_states, torch.float)
        
        # reward = torch.tensor(rewards, dtype=torch.float).unsqueeze(2).to(self.device)                     # (Time steps, Agents, 1)
        # done = torch.tensor(dones, dtype=torch.long).unsqueeze(2).to(self.device) 

        # # reward = safely_convert_to_tensor(rewards, torch.float).unsqueeze(2)
        # # done = safely_convert_to_tensor(dones, torch.long).unsqueeze(2)

        # discrete_action = safely_convert_to_tensor(discrete_actions, torch.long)
        # old_discrete_action_log_prob = safely_convert_to_tensor(discrete_log_probs, torch.float)
        # continuous_action = safely_convert_to_tensor(continuous_actions, torch.float)
        # continuous_means_stds = safely_convert_to_tensor(continuous_means_stds, torch.float)

        # 首先，我们创建一个列表，只包含HVAC智能体的键（索引）
        hvac_keys = [key for key in self.buffer.keys() if isinstance(key, int)]
        ev_keys = [key for key in self.buffer.keys() if isinstance(key, str) and 'charging_station' in key]

        state_np = np.array([[self.buffer[agent_id][time_step].state for agent_id in self.buffer.keys()] for time_step in range(len(next(iter(self.buffer.values()))))])
        next_state_np = np.array([[self.buffer[agent_id][time_step].next_state for agent_id in self.buffer.keys()] for time_step in range(len(next(iter(self.buffer.values()))))])
        reward_np = np.array([[self.buffer[agent_id][time_step].reward for agent_id in self.buffer.keys()] for time_step in range(len(next(iter(self.buffer.values()))))])
        done_np = np.array([[self.buffer[agent_id][time_step].done for agent_id in self.buffer.keys()] for time_step in range(len(next(iter(self.buffer.values()))))])

        # 使用这些键，我们从buffer中提取出所有HVAC智能体的discrete_action
        # 注意：我们假设所有的离散动作都是标量或可以直接使用numpy数组表示
        discrete_action_np = np.array([
            [self.buffer[agent_id][time_step].discrete_action if self.buffer[agent_id][time_step].discrete_action is not None else np.array([0])  # 使用[0]作为默认动作, 如果HVAC中存在None
            for agent_id in hvac_keys]  # 这里只循环HVAC智能体的键
            for time_step in range(len(next(iter(self.buffer.values()))))
        ])
        old_discrete_action_log_prob_np = np.array([
            [self.buffer[agent_id][time_step].discrete_a_log_prob if self.buffer[agent_id][time_step].discrete_a_log_prob is not None else np.array([0.0])  # 使用[0.0]作为默认概率值
            for agent_id in hvac_keys]  # 只选择HVAC智能体
            for time_step in range(len(next(iter(self.buffer.values()))))
        ])
        continuous_action_np = np.array([
            [self.buffer[agent_id][time_step].continuous_action if self.buffer[agent_id][time_step].continuous_action is not None else np.array([0.0, 0.0])  # 使用[0.0, 0.0]作为默认动作值
            for agent_id in ev_keys]  # 只选择EV站点智能体
            for time_step in range(len(next(iter(self.buffer.values()))))
        ])
        continuous_action_log_prob_np = np.array([
            [self.buffer[agent_id][time_step].continuous_action_log_prob if self.buffer[agent_id][time_step].continuous_action_log_prob is not None else np.array([0])  # 使用[0]作为默认动作, 如果EV站点中存在None
            for agent_id in ev_keys]  # 这里只循环EV站点智能体的键
            for time_step in range(len(next(iter(self.buffer.values()))))
        ])
        continuous_mean_np = np.array([
            [self.buffer[agent_id][time_step].continuous_mean if self.buffer[agent_id][time_step].continuous_mean is not None else np.array([0.0, 0.0])  # 使用均值0.0和标准差1.0作为默认值
            for agent_id in ev_keys]  # 只选择EV站点智能体
            for time_step in range(len(next(iter(self.buffer.values()))))
        ])
        continuous_std_np = np.array([
            [self.buffer[agent_id][time_step].continuous_std if self.buffer[agent_id][time_step].continuous_std is not None else np.array([1.0, 1.0])  # 使用均值0.0和标准差1.0作为默认值
            for agent_id in ev_keys]  # 只选择EV站点智能体
            for time_step in range(len(next(iter(self.buffer.values()))))
        ])
        
        state = torch.tensor(state_np, dtype=torch.float).to(self.device)                                   # (Time steps, Agents, State dim)
        next_state = torch.tensor(next_state_np, dtype=torch.float).to(self.device)                         # (Time steps, Agents, State dim)
        reward = torch.tensor(reward_np, dtype=torch.float).unsqueeze(2).to(self.device)                     # (Time steps, Agents, 1)
        done = torch.tensor(done_np, dtype=torch.long).unsqueeze(2).to(self.device)                        # (Time steps, Agents, 1)
        discrete_action = torch.tensor(discrete_action_np, dtype=torch.long).to(self.device)                                  # (Time steps, Agents, Action dim)
        old_discrete_action_log_prob = torch.tensor(old_discrete_action_log_prob_np, dtype=torch.float).to(self.device)       # (Time steps, Agents, Action dim)
        continuous_action = torch.tensor(continuous_action_np, dtype=torch.float).to(self.device)
        old_continuous_action_log_prob = torch.tensor(continuous_action_log_prob_np, dtype=torch.float).to(self.device)
        old_continuous_mean = torch.tensor(continuous_mean_np, dtype=torch.float).to(self.device)
        old_continuous_std = torch.tensor(continuous_std_np, dtype=torch.float).to(self.device)
        # 调试用: Add initial checks for NaNs or Infs in your buffers
        def check_tensors(*args, name="Check Point"):
            for i, arg in enumerate(args):
                if torch.isnan(arg).any() or torch.isinf(arg).any():
                    print(f"{name} - NaN or Inf found in tensor {i}")
        # Check for NaNs or Infs after conversion
        check_tensors(state, next_state, reward, done, name="state, next_state, reward, done")

        # Assume old_discrete_action_log_prob and other similar tensors are set up correctly before this point
        check_tensors(discrete_action, old_discrete_action_log_prob, name="discrete_action, old_discrete_action_log_prob")

        # Before using these for any computations, especially where NaNs or Infs can break the logic
        check_tensors(continuous_action, old_continuous_action_log_prob, name="continuous_action, old_continuous_action_log_prob")

        # Specifically, before creating any distributions with means and stds
        if torch.isnan(old_continuous_mean).any() or torch.isinf(old_continuous_mean).any():
            print("NaN or Inf found in 'means' before creating Normal distribution")
        if torch.isnan(old_continuous_std).any() or torch.isinf(old_continuous_std).any():
            print("NaN or Inf found in 'std' before creating Normal distribution")

        try:
            # Use means and stds to create a normal distribution
            normal_dist = Normal(old_continuous_mean, old_continuous_std)
        except ValueError as e:
            print(f"Failed to create normal distribution: {e}")

        num_time_steps, num_agents, _ = state.shape

        # Efan 重要计算奖励 Compute the returns
        Gt = torch.zeros(1, num_agents, 1).to(self.device)     # Artificially initialize the return tensor
        for i in reversed(range(num_time_steps)): 
            if done[i][0]:     # Efan's 需要检查格式 # All agents are done at the same time as done is only when environment is reset
                R = self.get_value(next_state[i]).unsqueeze(2)   # When last state of episode, start from estimated value of next state (1, num_agents, 1)
            R = reward[i].unsqueeze(0) + self.gamma * R    #(i, num_agents, 1)
            Gt = torch.cat([R, Gt], dim=0)          # Concatenate the returns for each time step (new return is in front of old returns)
        Gt = Gt[:-1,:,:]  # Remove last element as it was artificially added # (Time steps, Agents, 1)

        # Efan 重要更新网络 Update actor and critic
        ratios = np.array([]) 
        actor_gradient_norms = np.array([]) 
        critic_gradient_norms = np.array([]) 
        action_losses = np.array([])
        value_losses = np.array([])
        print("The agent is updating....") 


        for i in range(self.ppo_update_time): 
            for index in BatchSampler( 
                # SubsetRandomSampler从数据集中随机选择样本，而不是按顺序选择。首先，它接收一个索引列表（通常是数据集大小的整数序列）。然后，它会打乱这个列表，以随机的顺序提供索引。BatchSampler将Sampler（例如SubsetRandomSampler）返回的索引分组成指定大小的批次如256。每次迭代提供一组索引，对应于数据集中的一个批次。
                SubsetRandomSampler(range(num_time_steps)), self.batch_size, False 
            ): 
                if self.training_step % 1000 == 0: 
                    print("Time step: {}，train {} times".format(time_step, self.training_step)) 

                # 从计算好的返回值（Gt）中选取当前批次的数据。即预期回报（expected return）。再调用批评者网络（Critic network）来估算当前状态的价值V。再计算了“delta”或者说是TD（Temporal Difference）误差，即预期回报(实际收益)与当前状态价值估计之间的差异。最后，将TD误差赋值给优势函数（Advantage），这里使用.detach()方法是为了阻止这个计算参与梯度回传过程。在训练过程中，我们希望利用优势函数来指导策略的更新，但我们不希望这个优势函数本身的计算影响到网络参数的梯度。如果优势函数值为正，表示执行这个动作比平均情况要好。
                Gt_index = Gt[index]        # (Batch size, num_agents, 1)
                V = self.critic_net(state[index]).unsqueeze(2)   # (Batch size, num_agents) --> (Batch size, num_agents, 1)

                delta = Gt_index - V                    #(Batch size, num_agents, 1)

                advantage = delta.detach()              # Detach from the graph to avoid backpropagating


                state_batch = state[index]
                discrete_action_batch = discrete_action[index]
                old_discrete_action_log_prob_batch = old_discrete_action_log_prob[index]
                continuous_action_batch = continuous_action[index]
                old_continuous_action_log_prob_batch = old_continuous_action_log_prob[index]
                old_continuous_mean_batch = old_continuous_mean[index]
                old_continuous_std_batch = old_continuous_std[index]

                # 获取模型输出
                action_prob, means, stds = self.actor_net(state_batch, self.buffer.keys())

                # 初始化损失
                action_loss = 0

                # 离散动作损失（HVAC）
                if any(isinstance(x, int) for x in self.buffer.keys()):
                    hvac_indices = [idx for idx, key in enumerate(self.buffer.keys()) if isinstance(key, int)] # 所有智能体中对应位置的序号

                    selected_hvac_action_prob = action_prob.gather(2, discrete_action_batch)
                    ratio = selected_hvac_action_prob / old_discrete_action_log_prob_batch
                    clipped_ratio = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)
                    ratios = np.append(ratios, ratio.cpu().detach().numpy()) 
                    hvac_advantage = advantage[:, hvac_indices]  # 根据critic输入的state_batch即state[index]计算出来的
                    surr1 = ratio * hvac_advantage
                    surr2 = clipped_ratio * hvac_advantage
                    action_loss += -torch.min(surr1, surr2).mean()

                # 连续动作损失（EV）
                if any('charging_station' in str(x) for x in self.buffer.keys()):
                    ev_indices = [idx for idx, key in enumerate(self.buffer.keys()) if 'charging_station' in str(key)] # 对应位置的序号, 如4, 5
 
                    # ev_log_prob = old_continuous_action_log_prob_batch[:, ev_indices]  # 这里必须改,只包括连续动作的智能体,却用的所有智能体的索引,挑选是错误的. 扩展别的智能体,这个应该是? 

                    # 计算高斯损失
                    if torch.isnan(means).any() or torch.isnan(stds).any():  # 临时调试检查是否有错
                        print("Nan found in means or std")
                        print("Nan in means:", torch.isnan(means).any())
                        print("Nan in std:", torch.isnan(stds).any())
                    normal_dist = Normal(means, stds)
                    log_prob = normal_dist.log_prob(continuous_action_batch).sum(axis=-1, keepdim=True)
                    old_log_prob = old_continuous_action_log_prob_batch
                    ev_advantage = advantage[:, ev_indices]
                    ratio = torch.exp(log_prob - old_log_prob)
                    clipped_ratio = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)
                    ratios = np.append(ratios, ratio.cpu().detach().numpy()) 
                    surr1 = ratio * ev_advantage
                    surr2 = clipped_ratio * ev_advantage
                    action_loss += -torch.min(surr1, surr2).mean()

                # 更新网络
                self.actor_optimizer.zero_grad()
                action_losses = np.append(action_losses, action_loss.cpu().detach())
                action_loss.backward()
                
                actor_gradient_norm = nn.utils.clip_grad_norm_( 
                    self.actor_net.parameters(), self.max_grad_norm 
                ) 
                # 将计算得到的梯度范数记录下来。这一步骤通常用于监控训练过程，确保梯度保持在合理的范围内。使用.cpu().detach()方法是为了将梯度数据从GPU（如果使用的话）转移到CPU，并且从计算图中分离出来，以便于处理和存储而不影响梯度计算。 .step()为执行一步参数更新。
                actor_gradient_norms = np.append(actor_gradient_norms, actor_gradient_norm.cpu().detach()) 
                self.actor_optimizer.step()




                # # epoch iteration, PPO core 

                # action_prob, means, stds = self.actor_net(state[index], self.buffer.keys())  # (Batch size, num_agents, state dim) --> (Batch size, num_agents, action_choices * action dim)

                # action_prob = action_prob.gather(2, discrete_action[index])          # New policy's action probability --> (Batch size, num_agents, action dim)

                # ratio = action_prob / old_discrete_action_log_prob[index]  # Efan's 没有变化?
                # clipped_ratio = torch.clamp( 
                #     ratio, 1 - self.clip_param, 1 + self.clip_param 
                # ) 
                # ratios = np.append(ratios, ratio.cpu().detach().numpy()) 
                # surr1 = ratio * advantage               # (Batch size, num_agents, 1)
                # surr2 = clipped_ratio * advantage       # (Batch size, num_agents, 1)


                # # update actor network 
                # action_loss = -torch.min(surr1, surr2).mean()  # (Batch size, num_agents, 1) --> (Batch size, num_agents, 1) -->  1 (average over all batch and all agents).   MAX->MIN desent 
                # #在进行反向传播之前，先清零梯度。
                # self.actor_optimizer.zero_grad() 
                # action_losses = np.append(action_losses, action_loss.cpu().detach())

                # # action_loss.backward()计算action_loss的梯度。
                # action_loss.backward() 
                # # 梯度裁剪（Gradient Clipping）。这是防止梯度爆炸的一种常用技术。clip_grad_norm_函数将梯度向量的L2范数限制在self.max_grad_norm指定的最大范数之内。如果梯度的L2范数超过了这个值，那么会将梯度乘以一个缩放因子使其范数缩小到这个最大值。这里的actor_gradient_norm变量存储的是裁剪后的梯度范数值。
                # actor_gradient_norm = nn.utils.clip_grad_norm_( 
                #     self.actor_net.parameters(), self.max_grad_norm 
                # ) 
                # # 将计算得到的梯度范数记录下来。这一步骤通常用于监控训练过程，确保梯度保持在合理的范围内。使用.cpu().detach()方法是为了将梯度数据从GPU（如果使用的话）转移到CPU，并且从计算图中分离出来，以便于处理和存储而不影响梯度计算。 .step()为执行一步参数更新。
                # actor_gradient_norms = np.append(actor_gradient_norms, actor_gradient_norm.cpu().detach()) 
                # self.actor_optimizer.step() 
 
                # update critic network 
                # 通过对delta的平方取均值，实现了均方误差（MSE）损失的计算，这是评估价值函数预测准确性的常用方法。mean(0).mean(0)的调用确保了对所有维度（包括批次和智能体维度）的损失进行平均。
                value_loss = torch.pow(delta, 2).mean(0).mean(0)
                #print(value_loss.shape)
                #a = b #.mean(0) #F.mse_loss(Gt_index, V) 
                self.critic_net_optimizer.zero_grad() 
                value_losses = np.append(value_losses, value_loss.cpu().detach())
                
                value_loss.backward() 
                critic_gradient_norm = nn.utils.clip_grad_norm_( 
                    self.critic_net.parameters(), self.max_grad_norm 
                ) 
                critic_gradient_norms = np.append(critic_gradient_norms, critic_gradient_norm.cpu().detach()) 

                self.critic_net_optimizer.step() 
                self.training_step += 1 
 
        if self.log_wandb: 
 
            max_ratio = np.max(ratios) 
            mean_ratio = np.mean(ratios) 
            median_ratio = np.median(ratios) 
            min_ratio = np.min(ratios) 
            per95_ratio = np.percentile(ratios, 95) 
            per75_ratio = np.percentile(ratios, 75) 
            per25_ratio = np.percentile(ratios, 25) 
            per5_ratio = np.percentile(ratios, 5) 
            max_agradient_norm = np.max(actor_gradient_norms) 
            mean_agradient_norm = np.mean(actor_gradient_norms) 
            median_agradient_norm = np.median(actor_gradient_norms) 
            min_agradient_norm = np.min(actor_gradient_norms) 
            per95_agradient_norm = np.percentile(actor_gradient_norms, 95) 
            per75_agradient_norm = np.percentile(actor_gradient_norms, 75) 
            per25_agradient_norm = np.percentile(actor_gradient_norms, 25) 
            per5_agradient_norm = np.percentile(actor_gradient_norms, 5) 
            max_cgradient_norm = np.max(critic_gradient_norms) 
            mean_cgradient_norm = np.mean(critic_gradient_norms) 
            median_cgradient_norm = np.median(critic_gradient_norms) 
            min_cgradient_norm = np.min(critic_gradient_norms) 
            per95_cgradient_norm = np.percentile(critic_gradient_norms, 95) 
            per75_cgradient_norm = np.percentile(critic_gradient_norms, 75) 
            per25_cgradient_norm = np.percentile(critic_gradient_norms, 25) 
            per5_cgradient_norm = np.percentile(critic_gradient_norms, 5) 

            mean_action_loss = np.mean(action_losses)
            mean_value_loss = np.mean(value_losses)
            median_action_loss = np.median(action_losses)
            median_value_loss = np.median(value_losses)

 
            self.wandb_run.log( 
                { 
                    "PPO max ratio": max_ratio, 
                    "PPO mean ratio": mean_ratio, 
                    "PPO median ratio": median_ratio, 
                    "PPO min ratio": min_ratio, 
                    "PPO ratio 95 percentile": per95_ratio, 
                    "PPO ratio 5 percentile": per5_ratio, 
                    "PPO ratio 75 percentile": per75_ratio, 
                    "PPO ratio 25 percentile": per25_ratio, 
                    "Actor PPO max gradient norm": max_agradient_norm, 
                    "Actor PPO mean gradient norm": mean_agradient_norm, 
                    "Actor PPO median gradient norm": median_agradient_norm, 
                    "Actor PPO min gradient norm": min_agradient_norm, 
                    "Actor PPO gradient norm 95 percentile": per95_agradient_norm, 
                    "Actor PPO gradient norm 5 percentile": per5_agradient_norm, 
                    "Actor PPO gradient norm 75 percentile": per75_agradient_norm, 
                    "Actor PPO gradient norm 25 percentile": per25_agradient_norm, 
                    "Critic PPO max gradient norm": max_cgradient_norm, 
                    "Critic PPO mean gradient norm": mean_cgradient_norm, 
                    "Critic PPO median gradient norm": median_cgradient_norm, 
                    "Critic PPO min gradient norm": min_cgradient_norm, 
                    "Critic PPO gradient norm 95 percentile": per95_cgradient_norm, 
                    "Critic PPO gradient norm 5 percentile": per5_cgradient_norm, 
                    "Critic PPO gradient norm 75 percentile": per75_cgradient_norm, 
                    "Critic PPO gradient norm 25 percentile": per25_cgradient_norm, 
                    "PPO mean action loss": mean_action_loss,
                    "PPO mean value loss": mean_value_loss,
                    "PPO median action loss": median_action_loss,
                    "PPO median value loss": median_value_loss,
                    "Training steps": time_step, 
                } 
            ) 
 
        # 在循环结束后清空缓冲区
        self.reset_buffer()  # clear experience 
 