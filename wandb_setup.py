import wandb
import datetime

def wandb_setup(opt, config_dict):
    current_time = config_dict["default_env_prop"]["start_real_date"]

    log_config = {"opt": vars(opt), "config_file": config_dict}
    if opt.exp == "T":  # 训练就肯定有模型
        actor_name = opt.save_actor_name
        hvac_agent_name = ""
        station_agent_name = ""
    else:  # 测试
        actor_name = opt.actor_name
        hvac_agent_name = opt.agent
        station_agent_name = opt.station_agent
        if config_dict["default_env_prop"]["cluster_prop"]["hvac_nb_agents"] == 0:
            hvac_agent_name = ""
        if config_dict["default_env_prop"]["cluster_prop"]["station_nb_agents"] == 0:
            station_agent_name = ""
        # 检查actor_name是否包含指定的bangbang, 不包含则不需要网络模型, 处理两种时有瑕疵,还能改进
        test_names = ["BangBang", "DeadbandBangBang", "Basic", "AlwaysOn", "EvBangBang", "EvDeadbandBangBang", "EvBasic", "EvAlwaysOn"]
        if config_dict["default_env_prop"]["cluster_prop"]["hvac_nb_agents"] !=0 and any(name in hvac_agent_name for name in test_names):
            actor_name = ""
        if config_dict["default_env_prop"]["cluster_prop"]["station_nb_agents"] !=0 and any(name in station_agent_name for name in test_names):
            actor_name = ""
    lr = 0
    if "Tarmac" in actor_name:  # 确定交流数, 只有这种情况不一样
        comm = config_dict["TarMAC_PPO_prop"]["number_agents_comm_tarmac"]
        lr = config_dict["TarMAC_PPO_prop"]["lr_critic"]
    elif config_dict["default_env_prop"]["cluster_prop"]["hvac_nb_agents"] != 0: # 有hvac. 可交流数目前保持一致
        comm = config_dict["default_env_prop"]["cluster_prop"]["hvac_nb_agents_comm"]
    elif config_dict["default_env_prop"]["cluster_prop"]["station_nb_agents"] != 0:  # 没有hvac有station
        comm = config_dict["default_env_prop"]["cluster_prop"]["station_nb_agents_comm"]

    wandb_run = wandb.init(
        settings=wandb.Settings(start_method="fork"),
        project="ProofConcept",  # 需要在wandb网页上新建一个项目 ProofConcept copy1-ProofConcept 
        entity="effortking",  # Efan 原来marl-dr
        config=log_config,
        name="%s_c%d_%d%s_%d%s_dead%s_Δtar%s_%.2fe%d_rp%.3f_%.2fh_env%d_net%d_%s_S%d_%ds_epi%d_α%.1f_%.1f_%.1f_%.1f_%.1f_%.1f_P%.2f±%.1f_Q%.2f±%.1f_%s_lr%.4f_%s" % (
            opt.exp,
            comm,
            config_dict["default_env_prop"]["cluster_prop"]["hvac_nb_agents"],
            hvac_agent_name[:6],
            config_dict["default_env_prop"]["cluster_prop"]["station_nb_agents"],
            station_agent_name[:6],
            config_dict["default_env_prop"]["cluster_prop"]["charging_deadband"][:4],
            config_dict["default_env_prop"]["cluster_prop"]["charging_mode"][:4],
            config_dict["default_ev_prop"]["alpha_num_events"],
            config_dict["default_ev_prop"]["num_charging_events"],
            config_dict["default_ev_prop"]["ratio_p"],
            config_dict["default_ev_prop"]["mean_park"],
            opt.env_seed,
            opt.net_seed,
            config_dict["default_env_prop"]["power_grid_prop"]["signal_mode"][:3],
            config_dict["default_env_prop"]["power_grid_prop"]["signal_parameters"]["sinusoidals"]["periods"][1],
            config_dict["default_env_prop"]["time_step"],
            config_dict["training_prop"]["nb_tr_episodes"],
            config_dict["default_env_prop"]["reward_prop"]["alpha_temp"],
            config_dict["default_env_prop"]["reward_prop"]["alpha_hvac_active_sig"],
            config_dict["default_env_prop"]["reward_prop"]["alpha_ev_active_sig"],
            config_dict["default_env_prop"]["reward_prop"]["alpha_ev_reactive_sig"],
            config_dict["default_env_prop"]["reward_prop"]["alpha_ev_soc_penalty"],
            config_dict["default_env_prop"]["reward_prop"]["alpha_ev_time_penalty"],
            config_dict["default_env_prop"]["power_grid_prop"]["active_artificial_ratio"],
            config_dict["default_env_prop"]["power_grid_prop"]["artificial_active_signal_ratio_range"],
            config_dict["default_env_prop"]["power_grid_prop"]["reactive_artificial_ratio"],
            config_dict["default_env_prop"]["power_grid_prop"]["artificial_reactive_signal_ratio_range"],
            actor_name,
            lr,
            current_time
        ),
    )

    wandb_run.define_metric(name='Mean train return', step_metric='Training steps')
    wandb_run.define_metric(name='Mean temperature offset', step_metric='Training steps')
    wandb_run.define_metric(name='Mean temperature error', step_metric='Training steps')
    wandb_run.define_metric(name='Mean signal hvac active offset', step_metric='Training steps')
    wandb_run.define_metric(name='Mean signal havc active error', step_metric='Training steps')
    wandb_run.define_metric(name='Mean signal ev active offset', step_metric='Training steps')
    wandb_run.define_metric(name='Mean signal ev active error', step_metric='Training steps')
    wandb_run.define_metric(name='Mean signal ev reactive offset', step_metric='Training steps')
    wandb_run.define_metric(name='Mean signal ev reactive error', step_metric='Training steps')
    wandb_run.define_metric(name='Mean next signal error', step_metric='Training steps')
    wandb_run.define_metric(name='Mean next signal offset', step_metric='Training steps')
    wandb_run.define_metric(name='Mean test return', step_metric='Training steps')
    wandb_run.define_metric(name='Test mean temperature error', step_metric='Training steps')
    wandb_run.define_metric(name='Test mean signal error', step_metric='Training steps')
    wandb_run.define_metric(name='Mean EV queue count', step_metric='Training steps')
    return wandb_run
