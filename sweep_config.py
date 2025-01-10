import math
import datetime

def sweep_config_init():

    prefix = "wAddedFeatures_"
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


    #grid, random or bayesian
    sweep_config = {
        'name' : f"{prefix}{timestamp}",
        'method': 'random'
    }

    #what sweep optimizes for
    metric = {
        'name': 'metrics/test/loss',
        'goal': 'minimize'
    }

    #hyperparams to explore
    parameters_dict = {
        # Game
        'distractors': {
            'values': [15, 20, 30],
        },
        # Agents
        'embedding_size': {
            'values': [256]#[128, 256, 384]
        },
        'hidden_size': {
            'values': [128] #[20, 40, 64, 128]
        },
        'vocab_size':  {
            'values': [15, 24, 64]
        },
        'gs_tau': {
            'values': [0.85, 0.9, 0.95, 1.0]
        },
        # Training
        'mode':  {
            'values': ['gs']
        },
        'n_epochs':  {
            'values': [11]
        },
        # 'learning_rate': {
        #     'values': [0.01]
        # },
        'learning_rate':  {
            'distribution': 'log_uniform',
            'min': -6,
            'max': -3
        },
        'batch_size':  {
            'distribution': 'q_log_uniform',
            'q': 1,
            'min': math.log(16),
            'max': math.log(100)
        },
        # 'batch_size': {
        #     'values': [40]
        # }
    }

    sweep_config['metric'] = metric
    sweep_config['parameters'] = parameters_dict

    return sweep_config

