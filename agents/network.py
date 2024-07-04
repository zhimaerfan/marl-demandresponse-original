#%% Imports

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
import numpy as np
from utils import MaskedSoftmax

from agents.tarmac.distributions import DiagGaussian

#%% Classes


class Actor(nn.Module):  # PPO
    def __init__(self, num_state, num_action, layers):
        super(Actor, self).__init__()
        if isinstance(layers, str):
            layers = json.loads(layers)
            layers = [int(x) for x in layers]
        self.layers = layers

        self.fc = nn.ModuleList([nn.Linear(num_state, layers[0])])
        self.fc.extend(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(0, len(layers) - 1)]
        )
        self.fc.append(nn.Linear(layers[-1], num_action))
        print(self)

    def forward(self, x):
        for i in range(0, len(self.layers)):
            x = F.relu(self.fc[i](x))
        action_prob = F.softmax(self.fc[len(self.layers)](x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, num_state, layers):
        super(Critic, self).__init__()
        if isinstance(layers, str):
            layers = json.loads(layers)
            layers = [int(x) for x in layers]
        self.layers = layers

        self.fc = nn.ModuleList([nn.Linear(num_state, layers[0])])
        self.fc.extend(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(0, len(layers) - 1)]
        )
        self.fc.append(nn.Linear(layers[-1], 1))
        print(self)

    def forward(self, x):
        for i in range(0, len(self.layers)):
            x = F.relu(self.fc[i](x))
        value = self.fc[len(self.layers)](x)
        return value


class DQN_network(nn.Module):
    def __init__(self, num_state, num_action, layers):
        super(DQN_network, self).__init__()
        if isinstance(layers, str):
            layers = json.loads(layers)
            layers = [int(x) for x in layers]
        self.layers = layers

        self.fc = nn.ModuleList([nn.Linear(num_state, layers[0])])
        self.fc.extend(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(0, len(layers) - 1)]
        )
        self.fc.append(nn.Linear(layers[-1], num_action))
        print(self)

    def forward(self, x):
        for i in range(0, len(self.layers)):
            x = F.relu(self.fc[i](x))
        value = self.fc[len(self.layers)](x)
        return value


class DDPG_Network(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, non_linear=nn.ReLU()):
        super(DDPG_Network, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, out_dim),
        ).apply(self.init)

    @staticmethod
    def init(m):
        """init parameter of the module"""
        gain = nn.init.calculate_gain("relu")
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.net(x)


class TarMAC_Comm(nn.Module):
    def __init__(self, num_states, num_key, num_value, num_hops, number_agents_comm, mask_mode, comm_defect_prob, device):
        # super(TarMAC_Comm, self): 这部分获取TarMAC_Comm的父类，即nn.Module，self参数是指当前TarMAC_Comm类的实例。.__init__(): 这部分调用父类的构造函数。目的是确保nn.Module类的初始化代码被执行,下面的其他类都是这么super的.
        super(TarMAC_Comm, self).__init__()
        self.num_states = num_states
        self.num_hops = num_hops
        self.num_key = num_key
        self.number_agents_comm = number_agents_comm
        self.mask_mode = mask_mode
        self.device = device
        self.comm_defect_prob = comm_defect_prob

        self.hidden2key = nn.Sequential(
            nn.Linear(num_states, num_states),
            nn.Tanh(),
            nn.Linear(num_states, num_key)
        )

        self.hidden2value = nn.Sequential(
            nn.Linear(num_states, num_states),
            nn.Tanh(),
            nn.Linear(num_states, num_value)
        )

        # 查询向量query是由接收智能体的隐藏状态预测出来的，用于与发送智能体的签名进行匹配。通过这种方式，接收智能体可以确定哪些发送者的信息对当前情况最为相关。
        self.hidden2query = nn.Sequential(
            nn.Linear(num_states, num_states),
            nn.Tanh(),
            nn.Linear(num_states, num_key)
        )

        self.msg_state2state = nn.Sequential(
            nn.Linear(num_states + num_value, num_states + num_value),
            nn.Tanh(),
            nn.Linear(num_states + num_value, num_states)
        )

    def make_masks(self, number_agents):
        number_agents_comm = self.number_agents_comm
        if number_agents_comm >= number_agents:
            number_agents_comm = number_agents - 1
        if self.mask_mode == 'all':
            # 行数*列数的矩阵都是1
            mask = torch.ones(number_agents, number_agents)
        elif self.mask_mode == 'none':
            mask = torch.zeros(number_agents, number_agents)
        elif self.mask_mode == 'neighbours':
            mask_np = np.zeros((number_agents, number_agents))
            for i in range(number_agents_comm+1):
                if i == 0:
                    # 自我通信
                    # np.eye生成对角单位阵I,k是偏移量若是正数则只主对角线上方对角线填充1其余为0.
                    mask_np += np.eye(number_agents, k=0)
                elif i%2 == 1:
                    # 奇数, 它在主对角线上方和下方各添加了一条对角线，分别代表直接右侧邻居和直接左侧邻居的通信。
                    k_value = int((i+1)/2)
                    mask_np += np.eye(number_agents, k=k_value)
                    mask_np += np.eye(number_agents, k=-number_agents+k_value)
                else:
                    # 添加第二近邻的对角线
                    k_value = -int(i/2)
                    mask_np += np.eye(number_agents, k=k_value)
                    mask_np += np.eye(number_agents, k=number_agents+k_value)
                # number_agents = 5和number_agents_comm = 3的情况,需要首尾相连形成闭环:
                # [[1, 1, 1, 0, 1],
                #  [1, 1, 1, 1, 0],
                #  [0, 1, 1, 1, 1],
                #  [1, 0, 1, 1, 1],
                #  [1, 1, 0, 1, 1]]

            if self.comm_defect_prob > 0:
                for i in range(number_agents):
                    rand = np.random.rand()
                    if rand < self.comm_defect_prob:
                        # 某一列全变为0,后面再对角线填充1表示可与自己通信
                        mask_np[:, i] = 0
            mask = torch.from_numpy(mask_np).long() 
            mask.fill_diagonal_(1)          # The agent should always be able to speak to itself.
  
        elif self.mask_mode == 'random_sample':
            mask = torch.zeros(number_agents, number_agents)
            for i in range(number_agents):
                possible_choices = np.arange(i)
                possible_choices = np.append(possible_choices, np.arange(i+1, number_agents))
                random_list = np.random.choice(possible_choices, number_agents_comm, replace=False)
                mask[i, random_list] = 1   
                mask[i, i] = 1 
        else:
            raise ValueError('Unknown TarMAC communication mode')
        return mask


    def forward(self, hidden_states):
        # hidden_states: (batch_size, num_agents, num_states)
        for i in range(self.num_hops):
            if i>0:
                # i=0第一轮先初始化,第一轮通信主要是基于各个智能体自身的初始隐藏状态hidden_states来生成键（key）、值（value）和查询（query），然后计算注意力分数和comm（从其他智能体接收到的信息）,给后面i>0用
                # comm 是前一跳的通信输出, 使用 torch.cat 函数在第三个维度（dim=2）上拼接 comm 和 hidden_states. 拼接后，新的张量维度将是 (batch_size, num_agents, num_states + num_value)
                # msg_state2state 的作用是结合当前的隐藏状态和通过通信获得的额外信息（comm），以产生更新后的隐藏状态。输出的新隐藏状态维度重新变为 (batch_size, num_agents, num_states)，与原始的 hidden_states 维度相同。
                hidden_states = self.msg_state2state(torch.cat([comm, hidden_states], dim=2)) # (batch_size, num_agents, num_states + num_value) -> (batch_size, num_agents, num_states)
            
            # compute key, value and query
            key = self.hidden2key(hidden_states)        # (batch_size, num_agents, num_key)
            value = self.hidden2value(hidden_states)    # (batch_size, num_agents, num_value)
            query = self.hidden2query(hidden_states)    # (batch_size, num_agents, num_key)

            # .shape[1]表示 hidden_states 这个张量的第二个维度的大小，智能体的数量（num_agents）
            # 如mask = torch.tensor([
            #     [1, 0, 1],  # 第一个智能体只关注第1和第3个智能体
            #     [1, 1, 1],  # 第二个智能体关注所有智能体
            #     [0, 1, 1]   # 第三个智能体不关注第一个智能体
            # ])
            mask = self.make_masks(hidden_states.shape[1]).to(self.device) # (num_agents, num_agents)
            # 生成分数矩阵 scores，其中每个元素表示一个智能体对另一个智能体的注意力强度。例如，scores[b, i, j] 表示在第 b 个批次中，第 i 个智能体对第 j 个智能体的注意力分数。
            # query的形状为(batch_size, num_agents, num_key)，而key的形状也为(batch_size, num_agents, num_key)。为了进行点积操作，我们需要将key矩阵的最后两个维度进行转置. query和key开始都是(1, 3, 8),后来key变为(1, 8, 3),执行矩阵乘法. 
            # / math.sqrt(self.num_key) 这部分是将得分除以键（key）的维度的平方根(√num_key)。这是注意力机制中常见的缩放点积注意力（scaled dot-product attention）的一部分，目的是为了防止分数过高导致的梯度消失问题。
            
            # 如scores为:
            # tensor([[[0.0185, 0.0224, 0.0214, 0.0211],
            # [0.0180, 0.0231, 0.0210, 0.0196],
            # [0.0068, 0.0120, 0.0108, 0.0087],
            # [0.0072, 0.0113, 0.0099, 0.0092]]], device='cuda:0')
            
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.num_key) # (batch_size, num_agents, num_key) x (batch_size, num_key, num_agents) -> (batch_size, num_agents, num_agents)
            #scores = torch.mul(scores, mask) # (batch_size, num_agents, num_agents) * (num_agents, num_agents) -> (batch_size, num_agents, num_agents)
            #scores = scores.masked_fill(mask == 0, -10**15)
            # softmax + weighted sum

            # 如Z=[1,2,3]，让我们计算它的Softmax转换,e^1/(e^1+e^2+e^3)=0.090, 转换后的概率分布大约是 [0.090,0.245,0.665] # dim=-1表示最后一个维度
            attn = MaskedSoftmax(scores, mask, dim=-1)    # (batch_size, num_agents, num_agents)
            # 加权求和,如comm(1x3x16) = (1x3x3)x(1x3x16)或comm(1x4x16) = (1x4x4)x(1x4x16),3和4是num_agents
            # attn: 注意力概率即注意力权重矩阵. 形状为[1, 4, 4], 每一行表示一个智能体对其他所有智能体的关注度分布, 即其中每个元素attn索引[0, i, j]表示第i个智能体对第j个智能体的注意力权重。这是注意力机制的核心步骤，它允许模型动态地聚焦于最相关的信息。表示在当前处理步骤中, 每个智能体应该给予每个其他智能体多少“注意力”, 这些权重是通过处理原始的注意力分数（scores）并应用Softmax函数获得的，还进行了掩码处理。
            # value: 值向量矩阵, 形状为[1, 4, 16], 每一行表示一个智能体的信息或特征, 即value[0, i, :]表示第i个智能体的值向量。它包含了每个智能体的信息，这些信息需要根据注意力权重被聚合。
            # comm中的向量，即得到每个智能体的新value向量，这个新向量是根据该智能体的注意力分布加权求和(各权值相加=1,比对各智能体的信息值求平均更好)其他所有智能体的value向量得到的。comm索引[0, 0, :]即第0行是第一个智能体根据它对其他所有智能体的注意力权重（mask忽略了第3个智能体）对它们的value向量进行加权求和得到的结果,如comm[0, 0, 0]=0.32*v00+0.33*v10+0*v20+0.34*v30。
            comm = torch.matmul(attn, value)   # (batch_size, num_agents, num_agents) x (batch_size, num_agents, num_value) -> (batch_size, num_agents, num_value)
        return comm


class EntityEncoder(nn.Module):
    """
    编码不同类型智能体的信息。
    
    定义EntityEncoder：这是一个新的编码器类，用于将每个实体（智能体）的信息编码为固定长度的向量。这里使用Transformer编码器层来捕获实体之间的相互关系，并通过线性层输出固定大小的向量。
    后续在TarMAC_Actor中集成EntityEncoder：为不同类型的智能体（如hvac和station）分别实例化EntityEncoder，并在forward方法中使用它们对不同类型的智能体状态进行编码。
    处理异构输入：根据智能体类型选择相应的编码器进行状态编码，并将编码后的向量用于后续的决策过程。
    """
    def __init__(self, input_size, output_size):
        super(EntityEncoder, self).__init__()
        self.transformer = nn.TransformerEncoderLayer(d_model=input_size, nhead=4)
        self.output_layer = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        # 假设x是(batch_size, num_entities, feature_size)的形状
        transformed = self.transformer(x)
        # 取平均得到整体的实体Embedding
        entity_embedding = transformed.mean(dim=1)
        output = self.output_layer(entity_embedding)
        return output

class TarMAC_Actor(nn.Module):
    def __init__(self, num_obs_hvac, num_obs_station, num_key, num_value, hidden_state_size, num_hvac_action, num_station_action, number_agents_comm, comm_mode, device, comm_defect_prob=0, num_hops=1, with_gru=False, with_comm=True):

    # def __init__(self, num_obs, num_key, num_value, hidden_state_size, num_action, number_agents_comm, comm_mode, device, comm_defect_prob = 0, num_hops=1, with_gru=False, with_comm=True):
        super(TarMAC_Actor, self).__init__()
        self.with_gru = with_gru        # Not implemented yet
        if self.with_gru:
            raise NotImplementedError("GRU is not implemented yet")
        self.with_comm = with_comm

        # Efan's
        self.device = device
        self.num_hvac_action = num_hvac_action
        self.num_station_action = num_station_action

        # 分别处理HVAC和EV充电站智能体的观测到隐藏状态的转换
        self.hvac_obs2hidden = nn.Sequential(
            nn.Linear(num_obs_hvac, hidden_state_size),
            nn.ReLU(),
            nn.Linear(hidden_state_size, hidden_state_size),
        )
        self.station_obs2hidden = nn.Sequential(
            nn.Linear(num_obs_station, hidden_state_size),
            nn.ReLU(),
            nn.Linear(hidden_state_size, hidden_state_size),
        )

        if self.with_comm:
            self.comm = TarMAC_Comm(hidden_state_size, num_key, num_value, num_hops, number_agents_comm, comm_mode, comm_defect_prob, device)

            # 对于HVAC，使用通信后的隐藏状态进行动作决策
            self.comm_hidden2action_hvac = nn.Sequential(
                nn.Linear(num_value + hidden_state_size, hidden_state_size),
                nn.ReLU(),
                # num_action为什么是2？启停使用两个动作并通过概率分布来选择，是更常见也更灵活的做法
                nn.Linear(hidden_state_size, num_hvac_action)
            )
            # 对于EV充电站，计算连续动作的均值和标准差
            
            self.comm_hidden2mean_station = nn.Sequential(
                nn.Linear(num_value + hidden_state_size, hidden_state_size),
                nn.ReLU(),
                nn.Linear(hidden_state_size, num_station_action)
            )
            self.comm_hidden2log_std_station = nn.Sequential(
                nn.Linear(num_value + hidden_state_size, hidden_state_size),
                nn.ReLU(),
                nn.Linear(hidden_state_size, num_station_action),
                # nn.Softplus()  # 确保标准差为正
            )

        else:
            # HVAC的离散动作输出
            self.hidden2action_hvac = nn.Sequential(
                nn.Linear(hidden_state_size, hidden_state_size),
                nn.ReLU(),
                nn.Linear(hidden_state_size, num_hvac_action),
            )
            self.hidden2mean_station = nn.Sequential(
                nn.Linear(hidden_state_size, hidden_state_size),
                nn.ReLU(),
                nn.Linear(hidden_state_size, num_station_action)
            )
            self.hidden2log_std_station = nn.Sequential(
                nn.Linear(hidden_state_size, hidden_state_size),
                nn.ReLU(),
                nn.Linear(hidden_state_size, num_station_action),  
                # nn.Softplus()
            )
        # 理论上log_std对数标准差的取值范围是所有实数(−∞,+∞);log_std 的初始化很重要，通常初始化为较小的负数或零，如 -0.5 或 0。这可以保证在学习初期策略不会表现得过于随机。log_std = torch.clamp(log_std, min=-20, max=2)从而使得通过 exp(log_std) 计算得到的 std 保持在一个合理的范围内，有助于避免梯度爆炸或消失问题。
        # nn.Softplus() 确保标准差为正,可以直接生成分布,不需要再 torch.exp(log_stds)了,不然会导致标准差值非常大


        # # Efan's 为hvac和station类型的智能体分别设置encoder
        # self.hvac_encoder = EntityEncoder(input_size=num_obs_hvac, output_size=hidden_state_size)
        # self.station_encoder = EntityEncoder(input_size=num_obs_station, output_size=hidden_state_size)

        # # 通信模块
        # if self.with_comm:
        #     self.comm = TarMAC_Comm(hidden_state_size, num_key, num_value, num_hops, number_agents_comm, comm_mode, comm_defect_prob, device)
        #     self.comm_hidden2action_hvac = nn.Sequential(
        #         nn.Linear(num_value + hidden_state_size, hidden_state_size),
        #         nn.ReLU(),
        #         # num_hvac_action为什么是2？启停使用两个动作并通过概率分布来选择，是更常见也更灵活的做法
        #         nn.Linear(hidden_state_size, num_hvac_action)
        #     )
        #     self.comm_hidden2action_station = nn.Sequential(
        #         nn.Linear(num_value + hidden_state_size, hidden_state_size),
        #         nn.ReLU(),
        #         # 假设连续动作使用mean和std输出, 如果我们假设EV连续动作的输出是参数，如均值和标准差，那么对于每个动作维度，模型需要输出两个参数，因此总共需要输出2 * 2 = 4个参数。
        #         nn.Linear(hidden_state_size, num_station_action * 2)
        #     )
        # else:
        #     self.hidden2action_hvac = nn.Sequential(
        #         nn.Linear(hidden_state_size, hidden_state_size),
        #         nn.ReLU(),
        #         nn.Linear(hidden_state_size, num_hvac_action)
        #     )
        #     self.hidden2action_station = nn.Sequential(
        #         nn.Linear(hidden_state_size, hidden_state_size),
        #         nn.ReLU(),
        #         nn.Linear(hidden_state_size, num_station_action * 2)
        #     )


        # # 原来的obs2hidden. 移除了,因为我们将使用hvac_encoder和station_encoder分别处理状态
        # self.obs2hidden = nn.Sequential(
        #     nn.Linear(num_obs, hidden_state_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_state_size, hidden_state_size),
        # )

        # if self.with_comm:
        #     self.comm_hidden2action = nn.Sequential(
        #         nn.Linear(num_value+hidden_state_size, hidden_state_size),
        #         nn.ReLU(),
        #         # num_action为什么是2？启停使用两个动作并通过概率分布来选择，是更常见也更灵活的做法
        #         nn.Linear(hidden_state_size, num_action)
        #     )
        #     self.comm = TarMAC_Comm(hidden_state_size, num_key, num_value, num_hops, number_agents_comm, comm_mode, comm_defect_prob, device)
        # else:
        #     self.hidden2action = nn.Sequential(
        #         nn.Linear(hidden_state_size, hidden_state_size),
        #         nn.ReLU(),
        #         nn.Linear(hidden_state_size, num_action)
        #     )


    def forward(self, obs_batch, all_agent_ids):

        if torch.isnan(obs_batch).any():
            print("NaN detected in input observations")

        # 初始化隐藏状态列表
        hidden_states = []

        # 转换所有智能体的观测数据为隐藏状态
        for i, agent_id in enumerate(all_agent_ids):
            current_obs = obs_batch[:, i, :]  # 不再需要unsqueeze(0)，因为批次已经包含在obs_batch中,   [batch_size, num_features]
            if isinstance(agent_id, int):  # 对于HVAC智能体
                hidden_state = self.hvac_obs2hidden(current_obs)
            elif "charging_station" in agent_id:  # 对于EV充电站智能体
                hidden_state = self.station_obs2hidden(current_obs)
            if torch.isnan(hidden_state).any():
                print(f"NaN detected in hidden states for agent {agent_id}")
            hidden_states.append(hidden_state)

        # 将所有智能体的隐藏状态合并为一个批次
        hidden_states_tensor = torch.stack(hidden_states, dim=1)  # [batch_size, num_agents, hidden_state_size]

        # 通信处理
        if self.with_comm:
            comm_output = self.comm(hidden_states_tensor)  # [batch_size, num_agents, num_value]
            if torch.isnan(comm_output).any():
                print("NaN detected in communication outputs")


        # 根据智能体类型和通信输出处理决策
        action_probs, means, stds = [], [], []
        for i, agent_id in enumerate(all_agent_ids):
            if isinstance(agent_id, int):  # HVAC智能体决策
                if self.with_comm:
                    x = torch.cat([hidden_states_tensor[:, i, :], comm_output[:, i, :]], dim=-1)  # [batch_size, hidden_state_size + num_value]
                    action_logit = self.comm_hidden2action_hvac(x)
                action_prob = F.softmax(action_logit, dim=-1)
                action_probs.append(action_prob.unsqueeze(1))  # 保持批次和智能体维度
            elif "charging_station" in agent_id:  # EV充电站智能体决策
                if self.with_comm:
                    x = torch.cat([hidden_states_tensor[:, i, :], comm_output[:, i, :]], dim=-1)
                    mean = self.comm_hidden2mean_station(x)
                    log_std = self.comm_hidden2log_std_station(x) + 1e-6
                    log_std = torch.clamp(log_std, min=-20, max=2)
                    std = torch.exp(log_std)  # 计算标准差

                if torch.isnan(mean).any() or torch.isnan(std).any():
                    print(f"NaN detected in mean or std for charging station {agent_id}")

                means.append(mean.unsqueeze(1))  # 保持批次和智能体维度
                stds.append(std.unsqueeze(1))
                pass

        # 合并所有智能体的结果
        if action_probs:
            action_probs = torch.cat(action_probs, dim=1)  # [batch_size, num_agents, num_actions]
        if means:
            means = torch.cat(means, dim=1)  # [batch_size, num_agents, num_continuous_actions]
        if stds:
            stds = torch.cat(stds, dim=1)  # [batch_size, num_agents, num_continuous_actions]

        return action_probs, means, stds


class TarMAC_Critic(nn.Module):
    def __init__(self, num_agents_hvac, num_agents_station, num_obs_hvac, num_obs_station, hidden_layer_size):
        super(TarMAC_Critic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_obs_hvac*num_agents_hvac + num_obs_station*num_agents_station, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            # 有几个智能体num_agents, 则输出几个Value
            nn.Linear(hidden_layer_size, num_agents_hvac + num_agents_station)
        )


    def forward(self, obs):
        # obs: (nb_batch x num_agents x num_obs)
        x = obs.reshape(obs.shape[0], -1) # (nb_batch x num_agents x num_obs) -> (nb_batch x (num_agents x num_obs))
        # 生成n个智能体的value值
        value = self.critic(x) # (nb_batch x (num_agents x num_obs)) -> (nb_batch x num_agents)

        return value
# %%

class TarMAC_Actor_update(nn.Module):
    def __init__(self, num_obs_hvac, num_obs_station, num_key, num_value, hidden_state_size, num_hvac_action, num_station_action, number_agents_comm, comm_mode, device, comm_defect_prob=0, num_hops=1, with_gru=False, with_comm=True):

        super(TarMAC_Actor_update, self).__init__()
        self.with_gru = with_gru        # Not implemented yet
        if self.with_gru:
            raise NotImplementedError("GRU is not implemented yet")
        self.with_comm = with_comm

        # Efan's
        self.device = device
        self.num_hvac_action = num_hvac_action
        self.num_station_action = num_station_action

        # 分别处理HVAC和EV充电站智能体的观测到隐藏状态的转换
        self.hvac_obs2hidden = nn.Sequential(
            nn.Linear(num_obs_hvac, hidden_state_size),
            nn.ReLU(),
            nn.Linear(hidden_state_size, hidden_state_size),
        )
        self.station_obs2hidden = nn.Sequential(
            nn.Linear(num_obs_station, hidden_state_size),
            nn.ReLU(),
            nn.Linear(hidden_state_size, hidden_state_size),
        )

        if self.with_comm:
            self.comm = TarMAC_Comm(hidden_state_size, num_key, num_value, num_hops, number_agents_comm, comm_mode, comm_defect_prob, device)

            # 对于HVAC，使用通信后的隐藏状态进行动作决策
            self.comm_hidden2action_hvac = nn.Sequential(
                nn.Linear(num_value + hidden_state_size, hidden_state_size),
                nn.ReLU(),
                # num_action为什么是2？启停使用两个动作并通过概率分布来选择，是更常见也更灵活的做法
                nn.Linear(hidden_state_size, num_hvac_action)
            )
            # 对于EV充电站，计算连续动作的均值和标准差
            
            self.comm_hidden2mean_station = nn.Sequential(
                nn.Linear(num_value + hidden_state_size, hidden_state_size),
                nn.ReLU(),
                nn.Linear(hidden_state_size, num_station_action)
            )
            self.comm_hidden2log_std_station = nn.Sequential(
                nn.Linear(num_value + hidden_state_size, hidden_state_size),
                nn.ReLU(),
                nn.Linear(hidden_state_size, num_station_action),
                # nn.Softplus()  # 确保标准差为正
            )

        else:
            # HVAC的离散动作输出
            self.hidden2action_hvac = nn.Sequential(
                nn.Linear(hidden_state_size, hidden_state_size),
                nn.ReLU(),
                nn.Linear(hidden_state_size, num_hvac_action),
            )
            self.hidden2mean_station = nn.Sequential(
                nn.Linear(hidden_state_size, hidden_state_size),
                nn.ReLU(),
                nn.Linear(hidden_state_size, num_station_action)
            )
            self.hidden2log_std_station = nn.Sequential(
                nn.Linear(hidden_state_size, hidden_state_size),
                nn.ReLU(),
                nn.Linear(hidden_state_size, num_station_action),  
                # nn.Softplus()
            )
        # 理论上log_std对数标准差的取值范围是所有实数(−∞,+∞);log_std 的初始化很重要，通常初始化为较小的负数或零，如 -0.5 或 0。这可以保证在学习初期策略不会表现得过于随机。log_std = torch.clamp(log_std, min=-20, max=2)从而使得通过 exp(log_std) 计算得到的 std 保持在一个合理的范围内，有助于避免梯度爆炸或消失问题。
        # nn.Softplus() 确保标准差为正,可以直接生成分布,不需要再 torch.exp(log_stds)了,不然会导致标准差值非常大


        # # Efan's 为hvac和station类型的智能体分别设置encoder
        # self.hvac_encoder = EntityEncoder(input_size=num_obs_hvac, output_size=hidden_state_size)
        # self.station_encoder = EntityEncoder(input_size=num_obs_station, output_size=hidden_state_size)

        # # 通信模块
        # if self.with_comm:
        #     self.comm = TarMAC_Comm(hidden_state_size, num_key, num_value, num_hops, number_agents_comm, comm_mode, comm_defect_prob, device)
        #     self.comm_hidden2action_hvac = nn.Sequential(
        #         nn.Linear(num_value + hidden_state_size, hidden_state_size),
        #         nn.ReLU(),
        #         # num_hvac_action为什么是2？启停使用两个动作并通过概率分布来选择，是更常见也更灵活的做法
        #         nn.Linear(hidden_state_size, num_hvac_action)
        #     )
        #     self.comm_hidden2action_station = nn.Sequential(
        #         nn.Linear(num_value + hidden_state_size, hidden_state_size),
        #         nn.ReLU(),
        #         # 假设连续动作使用mean和std输出, 如果我们假设EV连续动作的输出是参数，如均值和标准差，那么对于每个动作维度，模型需要输出两个参数，因此总共需要输出2 * 2 = 4个参数。
        #         nn.Linear(hidden_state_size, num_station_action * 2)
        #     )
        # else:
        #     self.hidden2action_hvac = nn.Sequential(
        #         nn.Linear(hidden_state_size, hidden_state_size),
        #         nn.ReLU(),
        #         nn.Linear(hidden_state_size, num_hvac_action)
        #     )
        #     self.hidden2action_station = nn.Sequential(
        #         nn.Linear(hidden_state_size, hidden_state_size),
        #         nn.ReLU(),
        #         nn.Linear(hidden_state_size, num_station_action * 2)
        #     )


        # # 原来的obs2hidden. 移除了,因为我们将使用hvac_encoder和station_encoder分别处理状态
        # self.obs2hidden = nn.Sequential(
        #     nn.Linear(num_obs, hidden_state_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_state_size, hidden_state_size),
        # )

        # if self.with_comm:
        #     self.comm_hidden2action = nn.Sequential(
        #         nn.Linear(num_value+hidden_state_size, hidden_state_size),
        #         nn.ReLU(),
        #         # num_action为什么是2？启停使用两个动作并通过概率分布来选择，是更常见也更灵活的做法
        #         nn.Linear(hidden_state_size, num_action)
        #     )
        #     self.comm = TarMAC_Comm(hidden_state_size, num_key, num_value, num_hops, number_agents_comm, comm_mode, comm_defect_prob, device)
        # else:
        #     self.hidden2action = nn.Sequential(
        #         nn.Linear(hidden_state_size, hidden_state_size),
        #         nn.ReLU(),
        #         nn.Linear(hidden_state_size, num_action)
        #     )


    def forward(self, obs_batch, all_agent_ids):

        if torch.isnan(obs_batch).any():
            print("NaN detected in input observations")

        # 初始化隐藏状态列表
        hidden_states = []

        # 转换所有智能体的观测数据为隐藏状态
        for i, agent_id in enumerate(all_agent_ids):
            current_obs = obs_batch[:, i, :]  # 不再需要unsqueeze(0)，因为批次已经包含在obs_batch中,   [batch_size, num_features]
            if isinstance(agent_id, int):  # 对于HVAC智能体
                hidden_state = self.hvac_obs2hidden(current_obs)
            elif "charging_station" in agent_id:  # 对于EV充电站智能体
                hidden_state = self.station_obs2hidden(current_obs)
            if torch.isnan(hidden_state).any():
                print(f"NaN detected in hidden states for agent {agent_id}")
            hidden_states.append(hidden_state)

        # 将所有智能体的隐藏状态合并为一个批次
        hidden_states_tensor = torch.stack(hidden_states, dim=1)  # [batch_size, num_agents, hidden_state_size]

        # 通信处理
        if self.with_comm:
            comm_output = self.comm(hidden_states_tensor)  # [batch_size, num_agents, num_value]
            if torch.isnan(comm_output).any():
                print("NaN detected in communication outputs")


        # 根据智能体类型和通信输出处理决策
        action_probs, means, stds = [], [], []
        for i, agent_id in enumerate(all_agent_ids):
            if isinstance(agent_id, int):  # HVAC智能体决策
                if self.with_comm:
                    x = torch.cat([hidden_states_tensor[:, i, :], comm_output[:, i, :]], dim=-1)  # [batch_size, hidden_state_size + num_value]
                    action_logit = self.comm_hidden2action_hvac(x)
                action_prob = F.softmax(action_logit, dim=-1)
                action_probs.append(action_prob.unsqueeze(1))  # 保持批次和智能体维度
            elif "charging_station" in agent_id:  # EV充电站智能体决策
                if self.with_comm:
                    x = torch.cat([hidden_states_tensor[:, i, :], comm_output[:, i, :]], dim=-1)
                    mean = self.comm_hidden2mean_station(x)
                    log_std = self.comm_hidden2log_std_station(x) + 1e-6
                    log_std = torch.clamp(log_std, min=-20, max=2)
                    std = torch.exp(log_std)  # 计算标准差

                if torch.isnan(mean).any() or torch.isnan(std).any():
                    print(f"NaN detected in mean or std for charging station {agent_id}")

                means.append(mean.unsqueeze(1))  # 保持批次和智能体维度
                stds.append(std.unsqueeze(1))
                pass

        # 合并所有智能体的结果
        if action_probs:
            action_probs = torch.cat(action_probs, dim=1)  # [batch_size, num_agents, num_actions]
        if means:
            means = torch.cat(means, dim=1)  # [batch_size, num_agents, num_continuous_actions]
        if stds:
            stds = torch.cat(stds, dim=1)  # [batch_size, num_agents, num_continuous_actions]

        return action_probs, means, stds