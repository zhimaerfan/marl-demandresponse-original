import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.tarmac.utils_tarmac import init, init_normc_, AddBias
"""
Modify standard PyTorch distributions so they are compatible with this code.
这段代码的核心是扩展和修改PyTorch的标准概率分布类，以适应特定的使用场景。

灵活性和控制：这些修改和扩展提供了更多的灵活性和控制能力，以适应强化学习的需求，特别是在处理不同类型动作空间（离散和连续）时。
策略优化：在强化学习中，优化策略是核心任务之一。通过精确控制概率分布的参数（如均值和方差），可以更好地学习和优化策略，实现高效的探索和利用平衡。
算法适应性：修改后的分布类使得它们能够更好地集成到各种强化学习算法中，如PPO和Actor-Critic方法，提高了算法的通用性和效率。
"""

FixedCategorical = torch.distributions.Categorical

# 旧.sample()方法用于从分类分布中随机采样。它会返回一个索引值，该值表示被采样的类别. unsqueeze(-1)是为了确保采样结果的维度与期望保持一致，特别是在批量操作中。
# 创建匿名函数格式为lambda arguments: expression, arguments代表函数的参数，expression是函数体，lambda函数会返回表达式的计算结果。
old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

# 它允许对动作的对数概率（log probabilities, 是概率值的对数表示, 特别是在处理极小概率值时，因为它能够避免数值下溢问题即数值太小计算机无法精确处理, 同时简化乘法运算为加法运算. 在计算机科学和深度学习领域，更常使用自然对数e为底的概率）进行计算，同时自动处理输入和输出的形状，使其更加适合特定的使用场景. 例如，如果actions的形状是(N, 1)，squeeze(-1)会将其变为(N,)。
log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).unsqueeze(-1)

# 扩展现有类的功能: 计算并返回分布中概率最高的类别, 即概率分布的众数（Mode, 是指在该分布中具有最大概率值的元素）. 通过直接查找最大概率对应的索引，即可得到最可能的动作或类别。
FixedCategorical.mode = lambda self: self.probs.argmax(dim=1, keepdim=True)

# 修改了PyTorch的Normal正态分布类（torch.distributions.Normal），它定义了一个新的方法log_probs来计算动作（actions）的对数概率并对结果求和，适用于处理连续动作空间的场景，这在强化学习、机器学习建模和统计分析中很常见，尤其是使用策略梯度方法（如Proximal Policy Optimization, PPO）时，代理（agent）需要在连续的动作空间中作出决策。为这类问题提供了一种有效的概率建模和评估工具，使模型能够在考虑所有可能动作的概率后，做出更加精确和理性的决策。计算整个组合动作的总体概率，模型可以学习如何在风险和收益之间做出权衡. 这允许智能体在探索环境时具有一定的随机性，同时还能确保大部分动作围绕着最优解进行。通过从这个分布中采样动作，模型能够考虑到执行动作时的微小变化，因为每次采样可能会略有不同，这些微小的变化允许模型探索接近最优动作的其他可能性。这种方法在自动驾驶中尤为重要，因为它允许车辆以更平滑和自然的方式做出调整，比如平滑地加速或转向，而不是机械地重复同一个动作。
# 通过聚合（.sum(-1, keepdim=True)）多维动作的对数概率，可以简化对复杂动作的概率评估: 为了评估整个动作（由多个维度组成,如角度和力量）的总概率，需要将这些独立维度的对数概率相加（因为在概率论中，独立事件的联合概率等于各自概率的乘积，在对数空间中，乘积变为求和）。将所有维度的对数概率相加，得到一个综合的对数概率值，简化了复杂动作的概率评估。
# 在强化学习的连续动作空间问题中，策略（policy）通常由神经网络表示，网络的输出定义了动作概率分布的参数，即每个动作维度的均值和标准差。
# 动作均值（μ）：网络输出动作的期望值，表示在给定状态下模型认为最优的动作。
# 动作标准差（σ）：网络同时输出动作值的标准差，表示模型对于最优动作选择的不确定性。较小的标准差则表示较高的确定性。
FixedNormal = torch.distributions.Normal  # 以下5行代码确保 FixedNormal 类能够正确处理新的标准差计算方式，并且在整个模型中保持一致性
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)

# 覆盖原有的entropy方法，以便于在多维情况下对熵值进行求和处理。这种修改主要用于处理多维动作空间的情况。我们可能对整个动作空间的总体不确定性更感兴趣，而不是单个维度的不确定性。通过对所有维度的熵值进行求和，可以得到一个表示整个动作空间不确定性的单一度量。这对于评估策略的探索性质（即策略如何探索不同动作的能力）非常有用。
# 在强化学习算法，如PPO或MAPPO中，熵被用作一个额外的奖励信号来鼓励策略探索新的动作。策略的熵越高，意味着它的行为越随机，探索性越强。通过优化一个包含熵项的目标函数，算法不仅鼓励学习获得高奖励的行为，同时也鼓励保持一定程度的行为多样性，避免过早地收敛到次优策略。修改后的entropy方法，使得计算整个动作空间熵的过程变得简单直接，有助于实现这一目标。
entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: entropy(self).sum(-1)

# 这行代码为FixedNormal类（代表正态分布的PyTorch类）添加了一个名为mode的新方法，通过一个lambda函数实现。在统计学中，对于正态分布而言，众数（mode）是分布中出现概率最高的值，对于正态分布来说，其均值（mean）、中位数（median）和众数（mode）是相等的，都位于分布的中心。这行代码的作用是直接将正态分布的均值作为其众数返回。在正态分布中，由于其对称性，均值即代表了分布的中心位置，也就是最高点，因此它也是众数。
# 这种方法的优势在于其简洁性和直观性。对于正态分布，均值提供了一个明确的、易于理解的动作或决策倾向指标。在实际应用中，这允许模型或算法快速识别并采取最有可能带来最佳结果的动作，同时还保留了通过分布的其他属性（如标准差）来探索动作空间的能力。
FixedNormal.mode = lambda self: self.mean


# Categorical类封装了从输入特征到离散动作概率分布的映射过程, 设计用于生成离散动作空间中的动作概率分布。
# Categorical类结合了utils_tarmac.py中定义的初始化策略和模块，提供了一种生成和处理离散动作概率分布的高效方式。通过线性变换输入特征并创建分类分布，这个类为强化学习或分类任务中的动作决策提供了必要的工具。利用FixedCategorical，模型可以基于当前策略和状态，选择执行的动作，并计算这些动作的概率，以便进一步优化策略。
class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        # 使用自定义的初始化方法init_，该方法应用正交初始化（nn.init.orthogonal_）于权重，将偏置初始化为0（通过lambda x: nn.init.constant_(x, 0)），并通过gain=0.01调整初始化的缩放因子。
        init_ = lambda m: init(m,
              nn.init.orthogonal_,
              lambda x: nn.init.constant_(x, 0),
              gain=0.01)

        # self.linear定义了一个线性层，其输入和输出维度由num_inputs和num_outputs确定，且没有偏置项（bias=False）。
        self.linear = init_(nn.Linear(num_inputs, num_outputs, bias=False))

    def forward(self, x):
        # 作用：接收输入x，通过self.linear进行线性变换，然后使用FixedCategorical(logits=x)根据变换后的结果创建一个分类分布。
        x = self.linear(x)
        # 返回：FixedCategorical对象，代表基于输入x的线性变换结果的离散动作概率分布。
        return FixedCategorical(logits=x)


# DiagGaussian类专为连续动作空间设计，通过预测动作的均值和对数标准差，构造一个对角高斯（正态）分布。这种表示方式允许模型考虑到每个动作维度的独立性，同时通过学习调整标准差，控制探索的程度。
# 用于在强化学习中生成具有对角高斯分布（DiagGaussian）的动作概率分布。使用对角高斯分布来建模连续动作空间的策略是一种常见且有效的方法，尤其是在需要精细控制的场景下，如机器人操作、自动驾驶等。这种分布通常用于连续动作空间，其中每个动作维度的概率分布被假设为独立的，因此协方差矩阵是对角的。
# 协方差衡量了两个随机变量同时变化的趋势。如果两个变量倾向于一起增加或减少，它们的协方差为正；如果一个变量增加时另一个减少，它们的协方差为负；如果两个变量之间没有线性关系，它们的协方差为零。当协方差矩阵是对角的，即矩阵的非对角线元素（即不同动作维度之间的协方差）都是0，而对角线元素（每个动作维度的方差）可以是非零值。这表明每个动作维度相互独立，一个维度上的动作选择不会影响其他维度上的动作选择。
# 对角线元素表示变量自身的方差，因此通常是正的，因为方差衡量了变量取值的分散程度。方差为0意味着该变量在所有观测中都是常数，没有变化。
# 通过学习动作的均值和标准差，模型可以在探索环境（通过采样不同的动作）和利用当前最佳策略（通过选择均值作为动作）之间进行权衡。这种机制使得模型既能有效地学习环境，又能在确定性环境下表现出最优行为。
class DiagGaussian(nn.Module):
    # 参数：num_inputs表示输入特征的维度，num_outputs表示输出动作的维度，即高斯分布的均值向量的维度。
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()
        # 初始化均值层
        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        # 线性层 self.fc_mean：使用自定义的初始化方法init_初始化一个全连接层（nn.Linear），该层负责从输入特征预测每个动作维度的均值。权重使用init_normc_函数进行归一化初始，偏置项初始化为0。
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        # # 初始化log_std层. 对数标准差 self.logstd：使用AddBias类初始化为全0向量，该向量表示每个动作维度的对数标准差。通过对0向量添加偏置来实现，允许模型学习并调整每个动作维度的标准差。
        self.logstd = AddBias(torch.zeros(num_outputs))
        # self.logstd = AddBias(torch.full((num_outputs,), -0.5))  # Efan's 初始化为-0.5

    def forward(self, x):
        # 动作均值 action_mean：通过前面定义的全连接层self.fc_mean计算输入x对应的动作均值。
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        # 动作对数标准差 action_logstd：
        # 创建一个与action_mean相同大小的全0张量，根据输入x是否在CUDA设备上，相应地将0张量移动到CUDA设备。
        # 使用self.logstd对这个全0张量进行操作，生成每个动作维度的对数标准差。这是一个技巧，目的是利用AddBias的机制来动态调整学习到的对数标准差。
        action_logstd = self.logstd(zeros)
        action_logstd = torch.clamp(action_logstd, min=-20, max=2)  # Efan's 添加, 限制log_std的范围
        # 返回值：使用均值action_mean和标准差的指数action_logstd.exp()（将对数标准差转换回标准差）,创建一个FixedNormal分布对象。这个分布对象可以用来采样动作或计算给定动作的概率密度函数（PDF）。
        return FixedNormal(action_mean, action_logstd.exp())

