def setup_algo(adapter_config: dict):
    '''
    Set up algorithm config according to different architecture search methods
    '''
    algo_name = adapter_config['switch']['algo']['name']
    algo_config = {}
    if algo_name == 'darts':
        adapter_config['switch']['algo']['soft_train'] = True
        adapter_config['switch']['algo']['soft_switch'] = True
        adapter_config['switch']['algo']['first_order'] = True
        adapter_config['switch']['algo']['second_order'] = True
        adapter_config['switch']['algo']['use_gumbel'] = False
    elif algo_name == 'gdas':
        adapter_config['switch']['algo']['soft_train'] = False
        adapter_config['switch']['algo']['soft_switch'] = False
        adapter_config['switch']['algo']['first_order'] = False
        adapter_config['switch']['algo']['second_order'] = False
        adapter_config['switch']['algo']['use_gumbel'] = True
        # Tau
        adapter_config['switch']['tau']['init_value'] = 10
        adapter_config['switch']['tau']['stop_value'] = 0.1
        adapter_config['switch']['tau']['type'] = 'exp'
    elif algo_name == 'fair_darts':
        adapter_config['switch']['algo']['soft_train'] = True
        adapter_config['switch']['algo']['soft_switch'] = True
        adapter_config['switch']['algo']['first_order'] = True
        adapter_config['switch']['algo']['second_order'] = True
        adapter_config['switch']['algo']['use_gumbel'] = False
        adapter_config['switch']['algo']['aux_loss_ratio'] = 10
    elif algo_name == 'gumbel_darts':
        adapter_config['switch']['algo']['soft_train'] = True
        adapter_config['switch']['algo']['soft_switch'] = True
        adapter_config['switch']['algo']['first_order'] = True
        adapter_config['switch']['algo']['second_order'] = True
        adapter_config['switch']['algo']['use_gumbel'] = True
        # Tau
        adapter_config['switch']['tau']['init_value'] = 10
        adapter_config['switch']['tau']['stop_value'] = 0.1
        adapter_config['switch']['tau']['type'] = 'exp'
    elif algo_name == 's3delta':
        adapter_config['switch']['algo']['soft_train'] = False
        adapter_config['switch']['algo']['soft_switch'] = False
        adapter_config['switch']['algo']['first_order'] = True
        adapter_config['switch']['algo']['second_order'] = True
        adapter_config['switch']['algo']['use_gumbel'] = False

        adapter_config['switch']['tau']['type'] = 'const'
        adapter_config['switch']['tau']['init_value'] = 1
        adapter_config['switch']['algo']['sigmoid_tau'] = 1
        adapter_config['switch']['algo']['para_budget'] = 0.5 # 1.0M
    else:
        raise NotImplementedError(f"search algorithm {algo_name} is not implemented")

    return adapter_config