#%% Imports

import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import os
import random
import uuid
import wandb
import pandas as pd  # Efan 从csv读取表来画

from utils import normStateDict

#%% Functions

def plot_env_test(env, action_type='off', n_steps=1000):
    assert action_type in ['off', 'on', 'random'], 'Action types available: off/on/random' 
    action_types = {'on': 1, 'off': 0, 'random': 0}
    
    # Reset environment
    obs_dict = env.reset()
    
    # Initialize arrays
    reward = np.empty(n_steps)
    hvac = np.empty(n_steps)
    temp = np.empty(n_steps)
    
    # Act on environment and save reward, hvac status and temperature
    for t in range(n_steps):
        if action_type == 'random':
            action = {"0_1": random.randint(0,1)}
        else:
            action = {"0_1": action_types[action_type]}
        next_obs_dict, rewards_dict, dones_dict, info_dict = env.step(action)
        
        # Save data in arrays
        reward[t] = rewards_dict["0_1"]
        hvac[t] = next_obs_dict["0_1"]["hvac_turned_on"]
        temp[t] = next_obs_dict["0_1"]["house_temp"]

    plt.scatter(np.arange(len(hvac)), hvac, s=1, marker='.', c='orange')
    plt.plot(reward)
    plt.title('HVAC state vs. Reward')
    plt.show()
    plt.plot(temp)
    plt.title('Temperature')
    plt.show()
        
def plot_agent_test(env, agent, config_dict, n_steps=1000):      
    # Reset environment
    obs_dict = env.reset()
    cumul_avg_reward = 0
    
    # Initialize arrays
    reward = np.empty(n_steps)
    hvac = np.empty(n_steps)
    actions = np.empty(n_steps)
    temp = np.empty(n_steps)
    
    # Act on environment and save reward, hvac status and temperature
    for t in range(n_steps):
        action = {"0_1": agent.select_action(normStateDict(obs_dict["0_1"], config_dict))}
        next_obs_dict, rewards_dict, dones_dict, info_dict = env.step(action)
        
        # Save data in arrays
        actions[t] = action["0_1"]
        reward[t] = rewards_dict["0_1"]
        hvac[t] = next_obs_dict["0_1"]["hvac_turned_on"]
        temp[t] = next_obs_dict["0_1"]["house_temp"]
        
        cumul_avg_reward += rewards_dict["0_1"] / env.hvac_nb_agents
        
        obs_dict = next_obs_dict

    print(cumul_avg_reward/n_steps)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
    ax1.plot(actions)
    ax1.plot(hvac)
    ax1.title.set_text('HVAC state vs. Agent action')
    ax2.plot(reward)
    ax2.title.set_text("Reward")
    ax3.plot(temp)
    ax3.title.set_text('Temperature')
    plt.show()

#%%

def colorPlotTestAgentHouseTemp(prob_on_per_training_on, prob_on_per_training_off, low_temp, high_temp, time_steps_test_log, log_wandb):
    '''
    Makes a color plot of the probability of the agent to turn on given indoors temperature, with the training
    '''
    prob_on_per_training_on = prob_on_per_training_on[1:]
    prob_on_per_training_off = prob_on_per_training_off[1:]
    
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8,8.5), dpi=80)
    print(axes)

    normalizer = clr.Normalize(vmin=0, vmax=1)
    map0 = axes[0].imshow(np.transpose(prob_on_per_training_on), extent=[0, np.size(prob_on_per_training_on, 1)*time_steps_test_log, high_temp, low_temp], norm=normalizer)
    map1 = axes[1].imshow(np.transpose(prob_on_per_training_off), extent=[0, np.size(prob_on_per_training_off, 1)*time_steps_test_log, high_temp, low_temp], norm=normalizer)
    #axes[0] = plt.gca()
    axes[0].invert_yaxis()
    axes[1].invert_yaxis()

    forceAspect(axes[0], aspect=2.0)
    forceAspect(axes[1], aspect=2.0)

    axes[0].set_xlabel("Training time steps")
    axes[1].set_xlabel("Training time steps")
    axes[0].set_ylabel("Indoors temperature")
    axes[1].set_ylabel("Indoors temperature")
    axes[0].set_title("Power: ON")
    axes[1].set_title("Power: OFF")

    cb = fig.colorbar(map0, ax=axes[:], shrink=0.6)

    if log_wandb:
        name = uuid.uuid1().hex + "probTestAgent.png"
        plt.savefig(name)
        wandb.log(
            {"Probability of agent vs Indoor temperature vs Episode ": wandb.Image(name)})
        os.remove(name)

    else:
        plt.show()
    return 0

def forceAspect(ax, aspect):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

def plotEV(file_path):
    # Load the CSV file  'path_to_your_file.csv'  # 替换为您的文件路径
    data = pd.read_csv(file_path)

    # 确定电动汽车的数量
    num_evs = sum(['soc' in col for col in data.columns if 'target' not in col])  # 根据soc列的数量判断电动汽车的数量

    # 获取时间间隔
    time_intervals = data.iloc[:, 0]
    step = time_intervals[1] - time_intervals[0]

    # 绘制所有soc的变化曲线
    plt.figure(figsize=(10, 6))
    colors = plt.cm.get_cmap('tab10', num_evs)
    for i in range(1, num_evs + 1):
        color = colors(i - 1)
        plt.plot(time_intervals, data[f'soc{i}'], label=f'SOC {i}', color=color, alpha=0.7)
        if f'soc_target{i}' in data.columns:
            plt.plot(time_intervals, data[f'soc_target{i}'], label=f'Target SOC {i}', linestyle='--', color=color, alpha=0.7)
    plt.xlabel(f'Time (intervals of {step} seconds)')
    plt.ylabel('SOC')
    plt.title('SOC Variation')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 绘制所有有功功率曲线
    plt.figure(figsize=(10, 6))
    for i in range(1, num_evs + 1):
        plt.plot(time_intervals, data[f'ev active power{i}'], label=f'EV Active Power {i}', alpha=0.7)
    plt.xlabel(f'Time (intervals of {step} seconds)')
    plt.ylabel('Active Power')
    plt.title('EV Active Power')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 绘制所有无功功率曲线
    plt.figure(figsize=(10, 6))
    for i in range(1, num_evs + 1):
        plt.plot(time_intervals, data[f'ev reactive power{i}'], label=f'EV Reactive Power {i}', alpha=0.7)
    plt.xlabel(f'Time (intervals of {step} seconds)')
    plt.ylabel('Reactive Power')
    plt.title('EV Reactive Power')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 绘制剩下的曲线
    plt.figure(figsize=(10, 6))
    plt.plot(time_intervals, data['SOC Difference'], label='SOC Difference', alpha=0.7)
    plt.plot(time_intervals, data['EV Consumption'], label='EV Consumption', alpha=0.7)
    plt.plot(time_intervals, data['EV Active Signal'], label='EV Active Signal', alpha=0.7)
    plt.plot(time_intervals, data['HVAC Active Signal'], label='HVAC Active Signal', alpha=0.7)
    plt.plot(time_intervals, data['Total EV Reactive Power'], label='Total EV Reactive Power', alpha=0.7)
    plt.plot(time_intervals, data['EV Reactive Signal'], label='EV Reactive Signal', alpha=0.7)
    plt.xlabel(f'Time (intervals of {step} seconds)')
    plt.ylabel('Values')
    plt.title('Other Signals')
    plt.legend()
    plt.grid(True)
    plt.show()

    
# plotEV("/home/ef/Documents/code_results/marl-demandresponse-original/log/copyTarmacPPO20240703-17:14:51155950HVAC0-Station5-EV70-h11.99-20240704-11:11:10.csv")