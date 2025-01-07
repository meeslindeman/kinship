import math

def sweep_config_init():

    #grid, random or bayesian
    sweep_config = {
        'method': 'random'
    }

    #what sweep optimizes for
    metric = {
        'name': 'metrics/train/loss',
        'goal': 'minimize'
    }

    #hyperparams to explore
    parameters_dict = {
        # Agents
        'embedding_size': {
            'values': [20, 80, 128, 256]
        },
        'hidden_size': {
            'values': [20, 80, 128, 256]
        },
        'vocab_size':  {
            'values': [15, 24, 64, 100]
        },
        # Training
        'mode':  {
            'values': ['gs', 'rf', 'vq']
        },
        'n_epochs':  {
            'values': [30]
        },
        'learning_rate':  {
            'distribution': 'log_uniform',
            'min': -6,
            'max': -2
        },
        'batch_size':  {
            'distribution': 'q_log_uniform',
            'q': 1,
            'min': math.log(16),
            'max': math.log(600)
        },
    }

    sweep_config['metric'] = metric
    sweep_config['parameters'] = parameters_dict

    return sweep_config

