import wandb


def wandb_setup(opt, config_dict):

    log_config = {"opt": vars(opt), "config_file": config_dict}
    if opt.exp == "T":  # 训练
        actor_name = opt.save_actor_name
    else:  # 测试
        actor_name = opt.actor_name
    wandb_run = wandb.init(
        settings=wandb.Settings(start_method="fork"),
        project="ProofConcept",  # 需要在wandb网页上新建一个项目
        entity="effortking",  # Efan 原来marl-dr
        config=log_config,
        name="%s_TCLs_%d_envseed%d_netseed%d_%s" % (
            opt.exp, opt.hvac_nb_agents, opt.env_seed, opt.net_seed, actor_name),
    )
    wandb_run.define_metric(name='Mean train return', step_metric='Training steps')
    wandb_run.define_metric(name='Mean temperature offset', step_metric='Training steps')
    wandb_run.define_metric(name='Mean temperature error', step_metric='Training steps')
    wandb_run.define_metric(name='Mean signal offset', step_metric='Training steps')
    wandb_run.define_metric(name='Mean signal error', step_metric='Training steps')
    wandb_run.define_metric(name='Mean next signal error', step_metric='Training steps')
    wandb_run.define_metric(name='Mean next signal offset', step_metric='Training steps')
    wandb_run.define_metric(name='Mean test return', step_metric='Training steps')
    wandb_run.define_metric(name='Test mean temperature error', step_metric='Training steps')
    wandb_run.define_metric(name='Test mean signal error', step_metric='Training steps')
    return wandb_run
