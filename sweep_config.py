import datetime
import uuid

def sweep_config_init():

    prefix = "testing-time"
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

        #Seed
        'random_seed' : {
            'values': [int(uuid.uuid4().int % (2**32)) for _ in range(10)]
        },
        # Game
        'prune_graph' : {
            'values': [True],
        },
        'distractors': {
            'values': [5],
        },
        # Agents
        'embedding_size': {
            'values': [80]
        },
        'hidden_size': {
            'values': [80],
        },
        'vocab_size':  {
            'values': [64, 32, 16]
        },
        'layer':  {
            'values': ['rgcn']
        },
        # Training
        'mode':  {
            'values': ['gs']
        },
        'gs_tau': {
            'values': [1.5]
        },
        'n_epochs':  {
            'values': [250]
        },
        'learning_rate': {
            'values': [1e-3]
        },
        'batch_size': {
            'values': [50]
        },
        'max_len':{
            'values': [2]
        }

    }

    sweep_config['metric'] = metric
    sweep_config['parameters'] = parameters_dict

    return sweep_config

