config_dict = {
    # House properties
    # The house is modelled as a 10mx10m square with 2.5m height, 8 windows of 1.125 m² and 2 doors of 2m² each.
    # The formulas for Ua, Cm, Ca and Hm are mainly taken here: http://gridlab-d.shoutwiki.com/wiki/Residential_module_user's_guide
    "default_house_prop": {
        "id": 1,
        "init_air_temp": 20,
        "init_mass_temp": 20,
        "target_temp": 20,
        "deadband": 0,
        # 下面4个参数与论文一致
        "Ua": 2.18e02,  # House walls conductance (W/K). Multiplied by 3 to account for drafts (according to https://dothemath.ucsd.edu/2012/11/this-thermal-house/)
        "Cm": 3.45e06,  # House thermal mass (J/K) (area heat capacity:: 40700 J/K/m2 * area 100 m2)
        "Ca": 9.08e05,  # Air thermal mass in the house (J/K): 3 * (volumetric heat capacity: 1200 J/m3/K, default area 100 m2, default height 2.5 m)
        "Hm": 2.84e03,  # House mass surface conductance (W/K) (interioor surface heat tansfer coefficient: 8.14 W/K/m2; wall areas = Afloor + Aceiling + Aoutwalls + Ainwalls = A + A + (1+IWR)*h*R*sqrt(A/R) = 455m2 where R = width/depth of the house (default R: 1.5) and IWR is I/O wall surface ratio (default IWR: 1.5))
        "window_area": 7.175,  # Gross window area, in m^2
        "shading_coeff": 0.67,  # Window Solar Heat Gain Coefficient, look-up table in Gridlab reference
        "solar_gain_bool": True,  # Boolean to model the solar gain
    },
    "noise_house_prop": {
        "noise_mode": "big_start_temp",  # Can be: no_noise, small_noise, big_noise, small_start_temp, big_start_temp
        "noise_parameters": {
            "no_noise": {
                "std_start_temp": 0,  # Std noise on starting temperature
                "std_target_temp": 0,  # Std Noise on target temperature
                "factor_thermo_low": 1,  # Lowest random factor for Ua, Cm, Ca, Hm
                "factor_thermo_high": 1,  # Highest random factor for Ua, Cm, Ca, Hm
            },
            "dwarf_noise": {
                "std_start_temp": 0.05,  # Std noise on starting temperature
                "std_target_temp": 0.05,  # Std Noise on target temperature
                "factor_thermo_low": 1,  # Lowest random factor for Ua, Cm, Ca, Hm
                "factor_thermo_high": 1,  # Highest random factor for Ua, Cm, Ca, Hm
            },
            "house_small_noise": {
                "std_start_temp": 0,  # Std noise on starting temperature
                "std_target_temp": 0,  # Std Noise on target temperature
                "factor_thermo_low": 0.9,  # Lowest random factor for Ua, Cm, Ca, Hm
                "factor_thermo_high": 1.1,  # Highest random factor for Ua, Cm, Ca, Hm
            },
            "house_medium_noise": {
                "std_start_temp": 0,  # Std noise on starting temperature
                "std_target_temp": 0,  # Std Noise on target temperature
                "factor_thermo_low": 0.8,  # Lowest random factor for Ua, Cm, Ca, Hm
                "factor_thermo_high": 1.2,  # Highest random factor for Ua, Cm, Ca, Hm
            },
            "house_big_noise": {
                "std_start_temp": 0,  # Std noise on starting temperature
                "std_target_temp": 0,  # Std Noise on target temperature
                "factor_thermo_low": 0.5,  # Lowest random factor for Ua, Cm, Ca, Hm
                "factor_thermo_high": 1.5,  # Highest random factor for Ua, Cm, Ca, Hm
            },

            "small_noise": {
                "std_start_temp": 3,  # Std noise on starting temperature
                "std_target_temp": 1,  # Std Noise on target temperature
                "factor_thermo_low": 0.9,  # Lowest random factor for Ua, Cm, Ca, Hm
                "factor_thermo_high": 1.1,  # Highest random factor for Ua, Cm, Ca, Hm
            },
            "big_noise": {
                "std_start_temp": 5,  # Std noise on starting temperature
                "std_target_temp": 2,  # Std Noise on target temperature
                "factor_thermo_low": 0.8,  # Lowest random factor for Ua, Cm, Ca, Hm
                "factor_thermo_high": 1.2,  # Highest random factor for Ua, Cm, Ca, Hm
            },
            "small_start_temp": {
                "std_start_temp": 3,  # Std noise on starting temperature
                "std_target_temp": 0,  # Std Noise on target temperature
                "factor_thermo_low": 1,  # Lowest random factor for Ua, Cm, Ca, Hm
                "factor_thermo_high": 1,  # Highest random factor for Ua, Cm, Ca, Hm
            },
            "big_start_temp": {
                "std_start_temp": 5,  # Std noise on starting temperature
                "std_target_temp": 0,  # Std Noise on target temperature
                "factor_thermo_low": 1,  # Lowest random factor for Ua, Cm, Ca, Hm
                "factor_thermo_high": 1,  # Highest random factor for Ua, Cm, Ca, Hm
            },
        },
    },

    "noise_house_prop_test": { 
        "noise_mode": "small_start_temp",  # Can be: no_noise, small_noise, big_noise, small_start_temp, big_start_temp 
        "noise_parameters": { 
            "no_noise": { 
                "std_start_temp": 0,  # Std noise on starting temperature 
                "std_target_temp": 0,  # Std Noise on target temperature 
                "factor_thermo_low": 1,  # Lowest random factor for Ua, Cm, Ca, Hm 
                "factor_thermo_high": 1,  # Highest random factor for Ua, Cm, Ca, Hm 
            }, 
            "dwarf_noise": { 
                "std_start_temp": 0.05,  # Std noise on starting temperature 
                "std_target_temp": 0.05,  # Std Noise on target temperature 
                "factor_thermo_low": 1,  # Lowest random factor for Ua, Cm, Ca, Hm 
                "factor_thermo_high": 1,  # Highest random factor for Ua, Cm, Ca, Hm 
            }, 
            "small_noise": { 
                "std_start_temp": 3,  # Std noise on starting temperature 
                "std_target_temp": 1,  # Std Noise on target temperature 
                "factor_thermo_low": 0.9,  # Lowest random factor for Ua, Cm, Ca, Hm 
                "factor_thermo_high": 1.1,  # Highest random factor for Ua, Cm, Ca, Hm 
            }, 
            "big_noise": { 
                "std_start_temp": 5,  # Std noise on starting temperature 
                "std_target_temp": 2,  # Std Noise on target temperature 
                "factor_thermo_low": 0.8,  # Lowest random factor for Ua, Cm, Ca, Hm 
                "factor_thermo_high": 1.2,  # Highest random factor for Ua, Cm, Ca, Hm 
            }, 
            "small_start_temp": { 
                "std_start_temp": 3,  # Std noise on starting temperature 
                "std_target_temp": 0,  # Std Noise on target temperature 
                "factor_thermo_low": 1,  # Lowest random factor for Ua, Cm, Ca, Hm 
                "factor_thermo_high": 1,  # Highest random factor for Ua, Cm, Ca, Hm 
            }, 
            "big_start_temp": { 
                "std_start_temp": 5,  # Std noise on starting temperature 
                "std_target_temp": 0,  # Std Noise on target temperature 
                "factor_thermo_low": 1,  # Lowest random factor for Ua, Cm, Ca, Hm 
                "factor_thermo_high": 1,  # Highest random factor for Ua, Cm, Ca, Hm 
            }, 
        }, 
    }, 
    # HVAC properties
    "default_hvac_prop": {
        "id": 1,
        # 性能系数（消耗的功率与排出的热量之比）
        "COP": 2.5,  # Coefficient of performance (power spent vs heat displaced)
        "cooling_capacity": 15000,  # Cooling capacity (W)
        "latent_cooling_fraction": 0.35,  # Fraction of latent cooling w.r.t. sensible cooling
        "lockout_duration": 40,  # In seconds
        "lockout_noise": 0,  # In seconds
    },
    "noise_hvac_prop": {
        "noise_mode": "no_noise",  # Can be: no_noise, small_noise, big_noise
        "noise_parameters": {
            "no_noise": {
                "cooling_capacity_list": {10000: [10000], 15000: [15000]}
                # "std_latent_cooling_fraction": 0,     # Std Gaussian noise on latent_cooling_fraction
                # "factor_COP_low": 1,   # Lowest random factor for COP
                # "factor_COP_high": 1,   # Highest random factor for COP
                # "factor_cooling_capacity_low": 1,   # Lowest random factor for cooling_capacity
                # "factor_cooling_capacity_high": 1,   # Highest random factor for cooling_capacity
            },
            "small_noise": {
                "cooling_capacity_list": {10000: [9000, 10000, 11000], 15000: [12500, 15000, 17500]}
                # "std_latent_cooling_fraction": 0.05,     # Std Gaussian noise on latent_cooling_fraction
                # "factor_COP_low": 0.95,   # Lowest random factor for COP
                # "factor_COP_high": 1.05,   # Highest random factor for COP
                # "factor_cooling_capacity_low": 0.9,   # Lowest random factor for cooling_capacity
                # "factor_cooling_capacity_high": 1.1,   # Highest random factor for cooling_capacity
            },
            "big_noise": {
                "cooling_capacity_list": {
                    10000: [7500, 9000, 10000, 11000, 12500],
                    15000: [10000, 12500, 15000, 17500, 20000],
                }
                # "std_latent_cooling_fraction": 0.1,     # Std Gaussian noise on latent_cooling_fraction
                # "factor_COP_low": 0.85,   # Lowest random factor for COP
                # "factor_COP_high": 1.15,   # Highest random factor for COP
                # "factor_cooling_capacity_low": 0.6666667,   # Lowest random factor for cooling_capacity
                # "factor_cooling_capacity_high": 1.3333333333,   # Highest random factor for cooling_capacity
            },
        },
    },
    "noise_hvac_prop_test": {
        "noise_mode": "no_noise",  # Can be: no_noise, small_noise, big_noise
        "noise_parameters": {
            "no_noise": {
                "std_latent_cooling_fraction": 0,  # Std Gaussian noise on latent_cooling_fraction
                "factor_COP_low": 1,  # Lowest random factor for COP
                "factor_COP_high": 1,  # Highest random factor for COP
                "factor_cooling_capacity_low": 1,  # Lowest random factor for cooling_capacity
                "factor_cooling_capacity_high": 1,  # Highest random factor for cooling_capacity
            },
            "small_noise": {
                "std_latent_cooling_fraction": 0.05,  # Std Gaussian noise on latent_cooling_fraction
                "factor_COP_low": 0.95,  # Lowest random factor for COP
                "factor_COP_high": 1.05,  # Highest random factor for COP
                "factor_cooling_capacity_low": 0.9,  # Lowest random factor for cooling_capacity
                "factor_cooling_capacity_high": 1.1,  # Highest random factor for cooling_capacity
            },
            "big_noise": {
                "std_latent_cooling_fraction": 0.1,  # Std Gaussian noise on latent_cooling_fraction
                "factor_COP_low": 0.85,  # Lowest random factor for COP
                "factor_COP_high": 1.15,  # Highest random factor for COP
                "factor_cooling_capacity_low": 0.6666667,  # Lowest random factor for cooling_capacity
                "factor_cooling_capacity_high": 1.3333333333,  # Highest random factor for cooling_capacity
            },
        },
    },
    # 环境的属性，关于模拟的时间、温度模式、智能体（这里指的是房屋）的数量、它们之间的通信方式、状态、消息、电网和奖励
    # Env properties
    "default_env_prop": {
        "start_datetime": "2021-01-01 00:00:00",  # Start date and time (Y-m-d H:M:S)
        # 开始日期和时间的模式。原来是random（模拟的开始时间是在原始设定的 start_datetime 之后的一年内随机选择的）或fixed（始终从配置文件中指定的 start_datetime 开始）。
        "start_datetime_mode": "random",  # Can be random (randomly chosen in the year after original start_datetime) or fixed (stays as the original start_datetime)
        "time_step": 4,  # Time step in seconds
        # HVAC
        "cluster_prop": {
            "max_num_state":-1,  # 扩展智能体们到共同的最大状态数
            "max_self_num_state": -1, # 除消息外各智能体的自身最大状态数,用于扩展不同智能体状态数一致
            "max_message_num_state": -1, # 所有智能体获得的消息的最大状态数,用于扩展不同智能体状态数一致,等于max_num_state-max_self_num_state
            # Efan 先取充电桩数为智能体数量
            "station_nb_agents": None,
            # 单个EV可以与之通信的智能体的最大数量,不包括自己。先保持与HVAC的一致
            "station_nb_agents_comm": 2,
            "shuffle_ids": False, #多种智能体通讯时是否先打乱id的顺序,因为默认id列表是HVAC在前EV在后
            
            # 温度模式:三种
            "temp_mode": "noisy_sinusoidal_heatwave",  # Can be: constant, sinusoidal, noisy_sinusoidal
            # 每种模式都有其特定的参数，如day_temp（白天的温度）、night_temp（夜晚的温度）、temp_std（温度的噪声标准差）和random_phase_offset（是否有随机的相位偏移）。
            "temp_parameters": {
                "constant": {
                    "day_temp": 26.5,  # Day temperature
                    "night_temp": 26.5,  # Night temperature
                    "temp_std": 0,  # Noise std dev on the temperature
                    "random_phase_offset": False,
                },
                "sinusoidal": {
                    "day_temp": 30,
                    "night_temp": 23,
                    "temp_std": 0,
                    "random_phase_offset": False,
                },
                "sinusoidal_hot": {
                    "day_temp": 30,
                    "night_temp": 28,
                    "temp_std": 0,
                    "random_phase_offset": False,
                },
                "sinusoidal_heatwave": {
                    "day_temp": 34,
                    "night_temp": 28,
                    "temp_std": 0,
                    "random_phase_offset": False,
                },
                "sinusoidal_hot_heatwave": {
                    "day_temp": 38,
                    "night_temp": 32,
                    "temp_std": 0,
                    "random_phase_offset": False,
                },
                "sinusoidal_cold_heatwave": {
                    "day_temp": 30,
                    "night_temp": 24,
                    "temp_std": 0,
                    "random_phase_offset": False,
                },
                "sinusoidal_cold": {
                    "day_temp": 24,
                    "night_temp": 22,
                    "temp_std": 0,
                    "random_phase_offset": False,
                },
                "noisy_sinusoidal": {
                    "day_temp": 30,
                    "night_temp": 23,
                    "temp_std": 0.5,
                    "random_phase_offset": False,
                },
                "noisy_sinusoidal_hot": {
                    "day_temp": 30,
                    "night_temp": 28,
                    "temp_std": 0.5,
                    "random_phase_offset": False,
                },
                "noisy_sinusoidal_heatwave": {
                    "day_temp": 34,
                    "night_temp": 28,
                    "temp_std": 0.5,
                    "random_phase_offset": False,
                },
                "noisier_sinusoidal_heatwave": {
                    "day_temp": 34,
                    "night_temp": 28,
                    "temp_std": 2,
                    "random_phase_offset": False,
                },
                "noisy_sinusoidal_cold": {
                    "day_temp": 24,
                    "night_temp": 22,
                    "temp_std": 0.5,
                    "random_phase_offset": False,
                },
                "shifting_sinusoidal": {
                    "day_temp": 30,
                    "night_temp": 23,
                    "temp_std": 0,
                    "random_phase_offset": True,
                },
                "shifting_sinusoidal_heatwave": {
                    "day_temp": 34,
                    "night_temp": 28,
                    "temp_std": 0,
                    "random_phase_offset": True,
                },
            },
            # 默认为1,这只是房子,但房子里应该都有空调. 不包括EV的智能体数量
            "hvac_nb_agents": 10,  # Number of houses
            # 单个房屋可以与之通信的智能体的最大数量,不包括自己,测试时与训练时的该最大值必须一致,以保证输入神经网络的状态个数一致. 但可通过调number_agents_comm_tarmac控制可交流的数量,必须小于等于智能体总数否则训练报错
            # Efan 重要 与EV一起通讯时,与EV智能体数相加做为最大限制. 
            "hvac_nb_agents_comm": 2,  # Maximal number of houses a single house communicates with
            "agents_comm_mode": "neighbours",  # Communication mode: neighbours closed_groups random_sample random_fixed neighbours_2D no_message
            "comm_defect_prob": 0,  # Probability of a communication link being broken
            # 通信模式的参数:row_size（行的边长）和distance_comm（两个通信房屋之间的最大距离）。
            "agents_comm_parameters": {
                "neighbours_2D": {
                    # 智能体的总数应该能够被row_size整除
                    "row_size": 5,  # Row side length
                    "distance_comm": 2,  # Max distance between two communicating houses
                },
            },
        },
        # 状态中包含的属性,是否包含小时、天、太阳能增益、热、空调的信息
        # 这里指定某些状态不应该被包含，那么这些状态将被忽略。
        "state_properties": {
            "hour": False,
            "day": False,
            "solar_gain": False,
            "thermal": False,
            "hvac": False,
        },
        # 消息中包含的属性,是否包含热、HVAC信息
        "message_properties": {
            "thermal": False,
            "hvac": False,
        },
        # 电网属性
        "power_grid_prop": {
            # 基础功率模式，constant模式使用固定的功率值，而"interpolation"模式则使用插值方法根据提供的数据（基于死区 "砰砰 "控制器）动态计算功率。
            "base_power_mode": "interpolation",  # Interpolation (based on deadband bang-bang controller) or constant
            "EV_base_power_mode":"constant",  # Efan 都基于charging_events. 动态dynamic=不受控制的来了就正常充的总充电功率曲线(未实现) 或 常数 constant=当前时刻平均EV数*EV抵达的平均SoC充到离开的目标SoC平均值的总最小功率
            # 定义了各种基础功率模式的参数。
            "base_power_parameters": {
                "constant": {
                    "avg_power_per_hvac": 4200,  # 4200 Per hvac. In Watts.
                    # 每个HVAC的初始信号,开始时的功率?
                    "init_signal_per_hvac": 910,  # Per hvac.
                },
                "interpolation": {
                    # 可能包含与电网相关的某些数据，这些数据将用于插值。
                    "path_datafile": "./monteCarlo/mergedGridSearchResultFinal.npy",
                    # 可能包含用于插值的参数。
                    "path_parameter_dict": "./monteCarlo/interp_parameters_dict.json",
                    # 该文件可能包含与上述.json文件中的参数对应的键。
                    "path_dict_keys": "./monteCarlo/interp_dict_keys.csv",
                    # 插值更新的周期，单位为秒。这表示每隔300秒，插值会基于提供的数据进行更新。
                    "interp_update_period": 300,  # Seconds
                    "interp_hvac_nb_agents": 2,  # 100. 不能为0. Max number of agents over which the interpolation is run
                },
            },
            # 在训练期间，每个阶段随机乘以或除以的人工乘法因子的范围。例如：1不会改变信号。3将使信号介于计算值的 33% 和 300% 之间。
            "artificial_signal_ratio_range": 1,  # Scale of artificial multiplicative factor randomly multiplied (or divided) at each episode during training. Ex: 1 will not modify signal. 3 will have signal between 33% and 300% of what is computed.
            "active_artificial_ratio": 1.0,
            "reactive_artificial_ratio": 0.5,
            # 信号模式:平坦的、正弦、阶梯形、perlin及其变种
            "signal_mode": "perlin",  # Mode of the signal. Currently available: flat, sinusoidal, regular_steps, perlin
            "signal_parameters": {
                "flat": {},
                "sinusoidals": {
                    "periods": [400, 1200],  # In seconds
                    "amplitude_ratios": [0.1, 0.3],  # As a ratio of avg_power_per_hvac
                },
                "regular_steps": {
                    "amplitude_per_hvac": 6000,  # In watts
                    "period": 300,  # In seconds
                },
                "perlin": {
                    # Perlin噪声的振幅比例
                    "amplitude_ratios": 0.9,
                    # Perlin噪声的八度数
                    "nb_octaves": 5,
                    # 每个八度之间的步长
                    "octaves_step": 5,
                    # Perlin噪声的周期，单位为秒
                    "period": 400,
                },
                "amplitude+_perlin": {
                    "amplitude_ratios": 0.9*1.1,
                    "nb_octaves": 5,
                    "octaves_step": 5,
                    "period": 400,
                },
                "amplitude++_perlin": {
                    "amplitude_ratios": 0.9*1.3,
                    "nb_octaves": 5,
                    "octaves_step": 5,
                    "period": 400,
                },
                "fast+_perlin": {
                    "amplitude_ratios": 0.9,
                    "nb_octaves": 5,
                    "octaves_step": 5,
                    "period": 300,
                },
                "fast++_perlin": {
                    "amplitude_ratios": 0.9,
                    "nb_octaves": 5,
                    "octaves_step": 5,
                    "period": 200,
                },
            },
        },
        "reward_prop": {
            # 温度在损失函数中的权衡参数，用于在损失函数中调整温度的惩罚。具体来说，损失函数将是alpha_temp * 温度惩罚 + alpha_sig * 调节信号惩罚。
            "alpha_temp": 1,  # Tradeoff parameter for temperature in the loss function: alpha_temp * temperature penalty + alpha_sig * regulation signal penalty.
            "alpha_sig": 1,  # Tradeoff parameter for signal in the loss function: alpha_temp * temperature penalty + alpha_sig * regulation signal penalty.
            # 用于信号标准化的平均用电量
            "norm_active_reg_sig": 7500,  # Average power use, for signal normalization
            "norm_reactive_reg_sig": 7500,
            # 表示用于计算温度惩罚的模式。它可以是以下四种之一：
            # individual_L2--温度惩罚是基于单个房屋与其目标温度之间的L2范数差异来计算的
            # common_L2--温度惩罚是基于所有房屋与其目标温度之间的L2范数差异的平均值来计算的
            # common_max--温度惩罚是基于所有房屋与其目标温度之间的L2范数差异的最大值来计算的
            # mixture--温度惩罚是基于上述三种模式的混合来计算的。具体的混合权重由alpha_ind_L2、alpha_common_L2和alpha_common_max三个参数确定,见cli.py
            "temp_penalty_mode": "individual_L2",  # Mode of temperature penalty
            "temp_penalty_parameters": {
                "individual_L2": {},
                "common_L2": {},
                "common_max_error": {},
                "mixture": {
                    "alpha_ind_L2": 1,
                    "alpha_common_L2": 1,
                    "alpha_common_max": 0,
                },
            },
            # 信号惩罚的模式
            "sig_penalty_mode": "common_L2",  # Mode of signal penalty
        },
    },

    "default_ev_prop": {  # 注意此表格式不能变,不然不能生成EV事件
        "vehicle_types": [
            {
                "battery": {
                    "capacity": 100 * 1000,  # Wh
                    "efficiency": 0.9,  # 充电/放电效率
                    "max_active_power": 16.7 * 1000,  # W, 最大有功功率
                    "max_apparent_power": 16.7 * 1000,  # VA, 最大视在功率
                    "max_charge_power": 16.7 * 1000,  # 假设的最大充电功率值
                    "min_charge_power": 0 * 1000  # W, 最小充电功率
                },
                "brand": "Tesla",
                "model": "Model S",
                "probability": 0  # 出现概率
            },
            {
                "battery": {
                    "capacity": 100 * 1000,
                    "efficiency": 0.9,
                    "max_active_power": 17.6 * 1000,
                    "max_apparent_power": 17.6 * 1000,
                    "max_charge_power": 17.6 * 1000,  # 假设的最大充电功率值
                    "min_charge_power": 0 * 1000
                },
                "brand": "Nio",
                "model": "ES8",
                "probability": 1
            }
        ],
        # EV的其他配置
        "start_date": "2020-12-31T00:00:00",
        "end_date": "2022-01-01T00:00:00",  # 假设仿真时长为一天
        "resolution": "0:15:00",  # 4秒的时间间隔好像没必要
        "num_charging_events": 15, # 每周多少辆车
        "station_type": "private", # "public" or "private". 首先生成的充电事件, 私人充电桩则接入随机的充电桩, 来模拟每个充电桩有车的概率均等; 公用充电桩则按顺序接如入充电(如优先接入序号靠前的快充闲置充电桩, 序号靠后的慢充充电桩可能一直没有在充电)
        "num_stations": 0,  # 5个、10个、20个、50个、500个、1000个,EV事件数量如果一天1件,就应该乘7天.
        "mean_park": 23.99,  # 平均停车时间
        "std_deviation_park": 1,  # 停车时间的标准偏差
        "mean_soc": 0.5,
        "std_deviation_soc": 0.1,
        "soc_target": 0.8,  # 平均目标soc
        "std_soc_target":0.1,
        "disconnect_by_time": True,  # 特定时间过后是否从充电站断开连接
        "queue_length": 0,  # EV充电队列的长度，0表示没有队列
        "resolution_preload": "0:00:04",  # 预加载数据的时间分辨率或步长
        "repeat_preload": True,  # 是否在整个模拟中重复某些预加载条件（例如充电需求）
        "scheduling_policy": "Uncontrolled",  # 用于安排EV充电的策略。"Uncontrolled" 表示没有特定的控制算法被应用，充电可能在电动车连接后立即发生。
        "infrastructure": {
            "transformers": [
                {
                    "id": "transformer1",
                    "max_power": -1,  # 默认每个桩的n倍,后面计算会根据充电桩数量重新修改
                    "rated_power": -1,  # 同上
                    "min_power": 1 * 1000,
                    "charging_stations": [
                        {
                            "id": "charging_station1",
                            "max_power": 17.6 * 1000,
                            "rated_power": 5.92 * 1000,
                            "min_power": 1 * 1000,
                            "charging_points": [
                                {"id": "charging_point1", "max_power": 17.6 * 1000, "min_power": 0 * 1000}
                                # ... 其他充电点 ...
                            ]
                        }
                        # ... 其他充电站 ...
                    ]
                }
                # ... 其他变压器 ...
            ]
        },
        "transformer_preload": 0,
        "arrival_distribution": [  # 16个数为1天
            0.107142857, 0.087301587, 0.067460317, 0.047619048, 0.027777778,
            0.064814815, 0.101851852, 0.138888889, 0.291666667, 0.444444444,
            0.509259259, 0.574074074, 0.638888889, 0.759259259, 0.87962963, 1.0,
            0.805555556, 0.611111111, 0.416666667, 0.314814815, 0.212962963,
            0.111111111, 0.099206349, 0.087301587, 0.075396825, 0.063492063,
            0.051587302, 0.03968254, 0.027777778, 0.055555556, 0.083333333,
            0.111111111, 0.277777778, 0.444444444, 0.509259259, 0.574074074,
            0.638888889, 0.759259259, 0.87962963, 1.0, 0.828571429, 0.628571429,
            0.428571429, 0.333333333, 0.238095238, 0.142857143, 0.126530612,
            0.110204082, 0.093877551, 0.07755102, 0.06122449, 0.044897959,
            0.028571429, 0.057142857, 0.085714286, 0.114285714, 0.271428571,
            0.428571429, 0.504761905, 0.580952381, 0.657142857, 0.771428571,
            0.885714286, 1.0, 0.819047619, 0.638095238, 0.457142857, 0.352380952,
            0.247619048, 0.142857143, 0.126530612, 0.110204082, 0.093877551,
            0.07755102, 0.06122449, 0.044897959, 0.028571429, 0.057142857,
            0.085714286, 0.114285714, 0.285714286, 0.457142857, 0.523809524,
            0.59047619, 0.657142857, 0.771428571, 0.885714286, 1.0, 0.868686869,
            0.676767677, 0.484848485, 0.373737374, 0.262626263, 0.151515152,
            0.134199134, 0.116883117, 0.0995671, 0.082251082, 0.064935065,
            0.047619048, 0.03030303, 0.070707071, 0.111111111, 0.151515152,
            0.363636364, 0.575757576, 0.646464646, 0.717171717, 0.787878788,
            0.818181818, 0.848484848, 0.878787879, 0.717171717, 0.555555556,
            0.393939394, 0.333333333, 0.272727273, 0.212121212, 0.19047619,
            0.168831169, 0.147186147, 0.125541126, 0.103896104, 0.082251082,
            0.060606061, 0.111111111, 0.161616162, 0.212121212, 0.439393939,
            0.666666667, 0.646464646, 0.626262626, 0.606060606, 0.656565657,
            0.707070707, 0.757575758, 0.656565657, 0.555555556, 0.454545455,
            0.404040404, 0.353535354, 0.303030303, 0.264069264, 0.225108225
        ],
    
    },


    # Agent properties
    "PPO_prop": {
        # actor网络有两个隐藏层，每层都有100个神经元
        "actor_layers": [100, 100],
        "critic_layers": [100, 100],
        "gamma": 0.99,
        "lr_critic": 3e-3,
        "lr_actor": 1e-3,
        "clip_param": 0.2,
        # 梯度裁剪的最大范数。
        "max_grad_norm": 0.5,
        # 更新次数或更新频率。
        "ppo_update_time": 10,
        "batch_size": 256,
        # 在每个情节结束时将返回值设置为零
        "zero_eoepisode_return": False,
    },
    "MAPPO_prop": {
        "actor_layers": [100, 100],
        "critic_layers": [100, 100],
        "gamma": 0.99,
        "lr_critic": 3e-3,
        "lr_actor": 1e-3,
        "clip_param": 0.2,
        "max_grad_norm": 0.5,
        "ppo_update_time": 10,
        "batch_size": 256,
        "zero_eoepisode_return": False,
    },
    "DDPG_prop": {
        "actor_hidden_dim": 256,
        "critic_hidden_dim": 256,
        "gamma": 0.99,
        "lr_critic": 3e-3,
        "lr_actor": 1e-3,
        # 用于soft更新的参数
        "soft_tau": 0.01,
        "clip_param": 0.2,
        "max_grad_norm": 0.5,
        "ddpg_update_time": 10,
        "batch_size": 64,
        "buffer_capacity": 524288,
        "episode_num": 10000,
        # 学习的间隔
        "learn_interval": 100,
        # 在开始学习之前要采取的随机操作的数量
        "random_steps": 100,
        "gumbel_softmax_tau": 1,
        # 如果为True，则actor和critic共享一些层
        "DDPG_shared": True
    },
    
    "TarMAC_prop": {
        # 是否使用循环神经网络（RNN）作为策略。如果为True，策略将具有记忆性，可以处理序列数据?
	    "recurrent_policy": True, 	# Use RNN
	    "state_size": 128, 			# Size of the RNN state
	    "communication_size": 32, 	# Size of the communication message
        # TarMAC的通信模式。这不同于集群的communication_mode。例如，from_states_rec_att可能表示从状态中使用递归注意力进行通信。
        # 1.'no_comm', 没有通信;
        # 2.'from_states_rec_att', 基于它们的状态通信，并使用递归注意力机制,这允许智能体根据其他智能体的状态和它们之间的关系来加权和汇总信息;
        # 3.'from_states'直接基于它们的状态通信，不使用任何特定的注意力机制。
	    "tarmac_communication_mode": "from_states_rec_att",			# Mode of communication protocole (not the same as communication_mode of the cluster)
	    "comm_num_hops": 1,			# Number of hops during the communication
        #  在损失函数中的值损失和熵损失的系数。
	    "value_loss_coef": 0.5,	# Coefficient of the value loss in the loss function
	    "entropy_coef": 0.01,		# Coefficient of the entropy loss in the loss function
	    "tarmac_lr": 7e-4,					# Learning rate
        # RMSProp 或 Adam 优化器的 Epsilon 设置为 0.00001
	    "tarmac_eps": 1e-5,				# Epsilon for RMSProp or Adam optimizer
	    "tarmac_gamma": 0.99,				# Discount factor
        # RMSProp 优化器的 Alpha，设置为 0.99
	    "tarmac_alpha": 0.99,				# Alpha for RMSProp optimizer
        # 梯度的最大范数。如果为None，则不进行裁剪。
	    "tarmac_max_grad_norm": 0.5,		# Maximal norm of the gradient. If None, no clipping is done.
        # 如果为True，则使用分布式训练
	    "distributed": False,
        # TarMAC算法的更新次数
	    "nb_tarmac_updates": 10,
        # 每次更新使用的样本批次的大小?
	    "tarmac_batch_size": 128,
	},

    # 区别：
    # TarMAC agent和TarMAC PPO agent是两种不同的算法。TarMAC agent是基于RNN的多智能体通信算法，而TarMAC PPO agent结合了TarMAC的通信机制和PPO的更新策略。这意味着TarMAC PPO agent在基于TarMAC的通信机制的基础上，使用了PPO算法进行策略更新。TarMAC PPO代理是基于PPO的TarMAC代理的一个变种。
    # TarMAC PPO代理具有专门为PPO算法设计的参数，例如clip_param和ppo_update_time。
    # TarMAC PPO代理具有与通信相关的特定参数，例如with_gru、with_comm和tarmac_comm_mode。
    # TarMAC代理的配置参数更多地关注于通信和RNN的细节，而TarMAC PPO代理则更多地关注于PPO和通信的细节。
    # 这些差异表明，尽管两者都是基于TarMAC的，但TarMAC PPO代理是专门为PPO算法优化的版本，而TarMAC代理则更多地关注于通信和RNN(递归神经网络)的细节。
    
    # 从rl_controllers.py文件中，我们可以看到TarMAC代理的定义。以下是其主要特点：
    # 使用RNN（递归神经网络）。
    # 有状态大小、通信大小、通信模式等参数。
    # 使用RMSProp或Adam优化器。
    # 有学习率、epsilon、折扣因子等参数。
    
    # 从tarmac_ppo.py文件中，我们可以看到TarMAC PPO代理的定义。以下是其主要特点：
    # 定义了TarMACPPO类，该类继承自PPO类。
    # 使用了TarMAC的通信机制。有与TarMAC相似的参数，如通信大小、通信模式等。
    # 使用了PPO的更新策略。PPO是一种策略优化方法，它通过限制策略更新的大小来避免过大的策略更新，从而提高学习的稳定性。
    # 文章中提到，TarMAC PPO代理在实验中表现出了更好的性能，尤其是在系统规模较大时。
    "TarMAC_PPO_prop": {
        "actor_hidden_state_size": 64,   # Size of the hidden state of the actor
        "critic_hidden_layer_size": 64,         # Size of the hidden layers in the critic
        # 即num_value, 会改变comm_hidden2action网络的层数, 如nn.Linear(num_value+hidden_state_size, hidden_state_size)
	    "communication_size": 16, 	# Size of the communication message
        # 这可能与某种注意力机制或通信协议有关
	    "key_size": 8, 	# Size of the key/query 
        # 通信的跳数。在某些多智能体系统中，消息可能需要通过多个中间智能体传递，这个参数定义了这些跳数。 
	    "comm_num_hops": 1,			# Number of hops during the communication
        "lr_critic": 1e-3,
        "lr_actor": 1e-3,
        # 这是优化器（如RMSProp或Adam）中用于保持数值稳定性的小常数。它防止了除以零的情况。
	    "eps": 1e-5,				# Epsilon for RMSProp or Adam optimizer
	    "gamma": 0.99,				# Discount factor
        # 梯度的最大范数。如果设置了这个值，梯度裁剪将被应用，以防止梯度爆炸问题，确保训练过程的稳定性。
	    "max_grad_norm": 0.5,		# Maximal norm of the gradient. If None, no clipping is done.
        # PPO算法中的裁剪参数。这是一个防止策略更新过大的机制。
        "clip_param": 0.2,
        # 指的是在每个epoch（或数据收集周期）结束时，使用收集到的数据对策略进行更新的次数。这个参数直接影响了策略优化的频率和方式，其中更高的值意味着在进行下一个数据收集周期之前，相同的数据集被用来进行更多次的优化迭代。这有助于更充分地利用已收集的数据，但也可能增加计算负担，并有过拟合的风险。
        # 主要目的是提高同一批数据的利用率，通过重复利用数据来寻求更好的策略更新方向。
        "ppo_update_time": 10,
        "batch_size": 256,  # 我的不是训练慢 而是电动车那些模型跑的慢,模型跑出数据再训练网络 而且刚刚不是调试,快多了.调试就慢 拉屎 很快就updating....
        # 是否在网络中使用GRU（门控循环单元）. GRU代表门控循环单元（Gated Recurrent Unit），是深度学习领域中一种类型的循环神经网络（RNN）。GRU与另一种RNN类型——长短期记忆网络（LSTM）类似，但结构更为简单。它们常用于序列数据任务，如时间序列分析、自然语言处理和语音识别，因为它们能有效捕捉数据中的时间依赖性
        "with_gru": False,
        # 是否在智能体之间使用通信。
        "with_comm": True,
        # 重要 使用TarMAC进行通信的智能体数量。原来10. 训练时为N_ctr, 测试时为N_cde, 值可不同, 但必须小于EV智能体+HVAC智能体总数
        "number_agents_comm_tarmac": 2,
        # TarMAC的通信模式。它可以是以下几种模式之一：
        # neighbours: 只与邻居智能体通信。random_sample: 随机选择一些智能体进行通信。none: 不与任何智能体通信。all: 所有智能体都进行通信。
        "tarmac_comm_mode": 'neighbours',
        # 通信缺陷概率
        "tarmac_comm_defect_prob": 0.0 # Probability of a TarMAC communication defect
    },
    
    "DQN_prop": {
        "network_layers": [100, 100],
        "gamma": 0.99,
        "tau": 0.01,
        "buffer_capacity": 524288,
        "lr": 1e-3,
        "batch_size": 256,
        "epsilon_decay": 0.99998,
        "min_epsilon": 0.01,
    },
    "MPC_prop": {
        "rolling_horizon": 15,
    },


    # Training process properties
    "training_prop": {
        # 在训练过程中actor模型保存的中间次数,会被cli中的替换掉,除非cli.py中的值设为-1
        "nb_inter_saving_actor": 50, # Number of intermediate saving of the actor,原来是9
        # 在训练过程中智能体被测试的次数
        "nb_test_logs": 200, # Number of times the agent is tested during the training
        # 原来21600,训练中的每次测试将持续21600个时间步，相当于一个完整的天
        "nb_time_steps_test": 21600, # Number of time steps during the tests (1 full day)
        # 训练期间的环境重置（或训练回合）的次数
        "nb_tr_episodes": 200, # Number of training episodes (environment resets)
        # 训练期间的策略更新次数
        "nb_tr_epochs": 200, # Number of training epochs (policy updates)
        # 训练过程中性能被平均和记录的次数。
        "nb_tr_logs": 200, # Number of times the performances are averaged and logged during the training
        # 训练期间的总时间步数
        "nb_time_steps": 3276800, # 3276800 论文:3286800 time steps相当于 152 天，分为 200 集episodes。Number of time steps during the training
    },
}
