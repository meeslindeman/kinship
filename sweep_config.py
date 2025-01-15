import math
import datetime

def sweep_config_init():

    prefix = "wAddedFeatures_"
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


    #grid, random or bayesian
    sweep_config = {
        'name' : f"{prefix}{timestamp}",
        'method': 'grid'
    }

    #what sweep optimizes for
    metric = {
        'name': 'metrics/test/loss',
        'goal': 'minimize'
    }


    #hyperparams to explore
    parameters_dict = {
        #Seeds
        'data_seed' : {
            'values': [100, 101, 102, 103, 104, 105]
        },
        'random_seed' : {
            'values': [41,42,43,44,45],
        },
        # Game
        'prune_graph' : {
            'values': [True],
        },
        'distractors': {
            'values': [35, 5],
        },
        # Agents
        'embedding_size': {
            'values': [200]
        },
        'hidden_size': {
            'values': [200],
        },
        'vocab_size':  {
            'values': [100, 64, 32, 15]
        },
        # Training
        'mode':  {
            'values': ['gs']
        },
        'gs_tau': {
            'values': [1.5]
        },
        'n_epochs':  {
            'values': [200]
        },
        'learning_rate': {
            'values': [1e-3]
        },
        'batch_size': {
            'values': [50]
        },
        'max_len':{
            'values': [1]
        }
    }

    sweep_config['metric'] = metric
    sweep_config['parameters'] = parameters_dict

    return sweep_config

