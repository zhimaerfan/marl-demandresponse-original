import numpy as np

class Metrics:
    """
    用于在训练过程中收集和记录不同的性能指标, 帮助监控和评估模型的性能。
    这些性能指标包括平均回报、温度偏差、温度误差、信号偏差、信号误差等。这些指标有助于评估和监控模型的性能，确保模型正在正确学习并作出预期的决策。
    具体来说，Metrics类提供了以下功能：
    __init__: 初始化，设置所有累积指标的初始值为0。
    update: 更新累积指标的值。这个方法接受四个参数：k（代表当前的agent编号）、next_obs_dict（下一个状态的字典）、rewards_dict（奖励的字典）和env（环境对象）。这个方法使用这些参数来更新累积指标的值。
    log: 计算并返回所有累积指标的平均值。这个方法接受两个参数：t（当前的训练步数）和time_steps_train_log（训练日志的时间步数）。这个方法返回一个字典，包含所有计算出的平均指标值。
    reset: 重置所有累积指标的值为0。
    """
    def __init__(self):
        self.cumul_avg_reward = 0
        self.cumul_temp_offset = 0
        self.cumul_temp_error = 0
        self.cumul_signal_offset = 0
        self.cumul_signal_error = 0
        self.cumul_next_signal_offset = 0
        self.cumul_next_signal_error = 0
        
    def update(self, k, next_obs_dict, rewards_dict, env):
        self.cumul_temp_offset += (next_obs_dict[k]["house_temp"] - next_obs_dict[k]["house_target_temp"]) / env.nb_agents
        self.cumul_temp_error += np.abs(next_obs_dict[k]["house_temp"] - next_obs_dict[k]["house_target_temp"]) / env.nb_agents
        self.cumul_avg_reward += rewards_dict[k] / env.nb_agents
        self.cumul_next_signal_offset += (next_obs_dict[k]["reg_signal"] - next_obs_dict[k]["cluster_hvac_power"])/(env.nb_agents**2)
        self.cumul_next_signal_error += np.abs(next_obs_dict[k]["reg_signal"] - next_obs_dict[k]["cluster_hvac_power"])/(env.nb_agents**2)
        self.cumul_signal_offset += (next_obs_dict[k]["reg_signal"] - next_obs_dict[k]["cluster_hvac_power"])/(env.nb_agents**2)
        self.cumul_signal_error += np.abs(next_obs_dict[k]["reg_signal"] - next_obs_dict[k]["cluster_hvac_power"])/(env.nb_agents**2)

    def log(self, t, time_steps_train_log):
        mean_avg_return = self.cumul_avg_reward / time_steps_train_log
        mean_temp_offset = self.cumul_temp_offset / time_steps_train_log
        mean_temp_error = self.cumul_temp_error / time_steps_train_log
        mean_next_signal_offset = self.cumul_next_signal_offset / time_steps_train_log
        mean_next_signal_error = self.cumul_next_signal_error / time_steps_train_log
        mean_signal_offset = self.cumul_signal_offset / time_steps_train_log
        mean_signal_error = self.cumul_signal_error / time_steps_train_log
        metrics = {"Mean train return": mean_avg_return,
                   "Mean temperature offset": mean_temp_offset,
                   "Mean temperature error": mean_temp_error,
                   "Mean next signal offset": mean_next_signal_offset,
                   "Mean next signal error": mean_next_signal_error,
                   "Mean signal error": mean_signal_error,
                   "Mean signal offset": mean_signal_offset,
                   "Training steps": t}
        return metrics
    
    def reset(self):
        self.cumul_avg_reward = 0
        self.cumul_temp_offset = 0
        self.cumul_temp_error = 0
        self.cumul_signal_offset = 0
        self.cumul_signal_error = 0
        self.cumul_next_signal_offset = 0
        self.cumul_next_signal_error = 0