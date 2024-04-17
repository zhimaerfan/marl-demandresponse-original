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
        self.cumul_hvac_reward = 0
        self.cumul_ev_reward = 0
        self.cumul_temp_offset = 0
        self.cumul_temp_error = 0
        self.cumul_active_signal_offset = 0
        self.cumul_active_signal_error = 0
        # self.cumul_next_active_signal_offset = 0
        # self.cumul_next_active_signal_error = 0
        self.cumul_ev_queue_count = 0  # 充电队列的长度
        
    def update(self, k, next_obs_dict, rewards_dict, env):
        if k in env.hvac_agent_ids:
            self.cumul_temp_offset += (next_obs_dict[k]["house_temp"] - next_obs_dict[k]["house_target_temp"]) / env.hvac_nb_agents
            self.cumul_temp_error += np.abs(next_obs_dict[k]["house_temp"] - next_obs_dict[k]["house_target_temp"]) / env.hvac_nb_agents
            self.cumul_avg_reward += rewards_dict[k] / env.hvac_nb_agents
            # self.cumul_next_active_signal_offset += (next_obs_dict[k]["grid_active_reg_signal"] - next_obs_dict[k]["cluster_hvac_active_power"])/(env.hvac_nb_agents**2)
            # self.cumul_next_active_signal_error += np.abs(next_obs_dict[k]["grid_active_reg_signal"] - next_obs_dict[k]["cluster_hvac_active_power"])/(env.hvac_nb_agents**2)
            self.cumul_active_signal_offset += (next_obs_dict[k]["grid_active_reg_signal"] - next_obs_dict[k]["cluster_hvac_active_power"])/(env.hvac_nb_agents**2)
            self.cumul_active_signal_error += np.abs(next_obs_dict[k]["grid_active_reg_signal"] - next_obs_dict[k]["cluster_hvac_active_power"])/(env.hvac_nb_agents**2)
            
            # 注意，上面cumul_next_active... 和 cumul_active... 似乎在 update 方法中被赋予了相同的值，这可能是一个错误，因为通常这两个指标应该反映不同的计算。

        elif k.startswith('charging_station'):
            # 新增：更新EV特有的指标
            self.cumul_ev_queue_count += next_obs_dict[k]["ev_queue_count"] / env.station_nb_agents

            # 更新共有的指标
            self.cumul_avg_reward += rewards_dict[k] / env.station_nb_agents  # 继续HVAC累加计算EV的
            # self.cumul_next_active_signal_offset += (next_obs_dict[k]["grid_active_reg_signal"] - next_obs_dict[k]["cluster_hvac_active_power"])/(env.station_nb_agents**2)
            # self.cumul_next_active_signal_error += np.abs(next_obs_dict[k]["grid_active_reg_signal"] - next_obs_dict[k]["cluster_hvac_active_power"])/(env.station_nb_agents**2)
            self.cumul_active_signal_offset += (next_obs_dict[k]["grid_active_reg_signal"] - next_obs_dict[k]["cluster_hvac_active_power"])/(env.station_nb_agents**2)
            self.cumul_active_signal_error += np.abs(next_obs_dict[k]["grid_active_reg_signal"] - next_obs_dict[k]["cluster_hvac_active_power"])/(env.station_nb_agents**2)



    def log(self, t, time_steps_train_log):
        mean_avg_return = self.cumul_avg_reward / time_steps_train_log
        mean_temp_offset = self.cumul_temp_offset / time_steps_train_log
        mean_temp_error = self.cumul_temp_error / time_steps_train_log
        # mean_next_active_signal_offset = self.cumul_next_active_signal_offset / time_steps_train_log
        # mean_next_active_signal_error = self.cumul_next_active_signal_error / time_steps_train_log
        mean_active_signal_offset = self.cumul_active_signal_offset / time_steps_train_log
        mean_active_signal_error = self.cumul_active_signal_error / time_steps_train_log
        mean_ev_queue_count = self.cumul_ev_queue_count / time_steps_train_log  # 新增：平均EV队列长度

        metrics = {"Mean train return": mean_avg_return,
                   "Mean temperature offset": mean_temp_offset,
                   "Mean temperature error": mean_temp_error,
                #    "Mean next active signal offset": mean_next_active_signal_offset,
                #    "Mean next active signal error": mean_next_active_signal_error,
                   "Mean signal active error": mean_active_signal_error,
                   "Mean signal active offset": mean_active_signal_offset,
                   "Mean EV queue count": mean_ev_queue_count,  # 新增
                   "Training steps": t}
        return metrics
    
    def reset(self):
        self.cumul_avg_reward = 0
        self.cumul_temp_offset = 0
        self.cumul_temp_error = 0
        self.cumul_active_signal_offset = 0
        self.cumul_active_signal_error = 0
        # self.cumul_next_active_signal_offset = 0
        # self.cumul_next_active_signal_error = 0
        self.cumul_ev_queue_count = 0  # 新增