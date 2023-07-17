def setup_algo(algo: str, adapter_config: dict):
    '''
    Set up algorithm config according to different architecture search methods
    '''
    algo_name = adapter_config['switch']['algo']['name']
    if algo_name == 'darts':
        setattr(adapter_config['switch']['algo'], 'soft_train', True)
        setattr(adapter_config['switch']['algo'], 'soft_switch', True)
        setattr(adapter_config['switch']['algo'], 'first_order_approx', False)
        setattr(adapter_config['switch']['algo'], 'use_gumbel', False)
    elif algo_name == 'gdas':
        setattr(adapter_config['switch']['algo'], 'soft_train', False)
        setattr(adapter_config['switch']['algo'], 'soft_switch', False)
        setattr(adapter_config['switch']['algo'], 'use_gumbel', True)
        # Tau
        setattr(adapter_config['switch']['tau'], 'init_value', 10)
        setattr(adapter_config['switch']['tau'], 'stop_value', 0.1)
        setattr(adapter_config['switch']['tau'], 'type', 'exp')
    elif algo_name == 'fair-darts':
        setattr(adapter_config['switch']['algo'], 'soft_train', True)
        setattr(adapter_config['switch']['algo'], 'soft_switch', True)
        setattr(adapter_config['switch']['algo'], 'first_order_approx', False)
        setattr(adapter_config['switch']['algo'], 'use_gumbel', False)
        setattr(adapter_config['switch']['algo'], 'aux_loss_ratio', 10)
    elif 'gumbel-darts':
        setattr(adapter_config['switch']['algo'], 'soft_train', True)
        setattr(adapter_config['switch']['algo'], 'soft_switch', True)
        setattr(adapter_config['switch']['algo'], 'first_order_approx', False)
        setattr(adapter_config['switch']['algo'], 'use_gumbel', True)
        # Tau
        setattr(adapter_config['switch']['tau'], 'init_value', 10)
        setattr(adapter_config['switch']['tau'], 'stop_value', 0.1)
        setattr(adapter_config['switch']['tau'], 'type', 'exp')
    else:
        raise NotImplementedError(f"search algorithm {algo} is not implemented")

    return adapter_config