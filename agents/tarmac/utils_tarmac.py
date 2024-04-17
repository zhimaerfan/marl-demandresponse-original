import torch
import torch.nn as nn


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    # 用于为输入x添加偏置项。这在某些网络结构中非常有用，特别是在手动控制偏置项的情况下。
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def init(module, weight_init, bias_init, gain=1):
    # 对PyTorch模块（如线性层）进行权重和偏置的初始化。这个函数接受一个初始化模块、权重初始化函数、偏置初始化函数和可选的增益参数gain。
    weight_init(module.weight.data, gain=gain)
    if hasattr(module.bias, 'data'):
        bias_init(module.bias.data)
    return module


# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    # 以特定方式初始化权重，首先按照正态分布随机分配值，然后调整这些值，使得每个输出维度的权重向量的长度等于gain参数。
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))
