import datetime

def sweep_config_init():

    prefix = "deleteme"
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
            'values': [51,52,53,54,55]
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
            'values': [1,2,3]
        }

    }

    sweep_config['metric'] = metric
    sweep_config['parameters'] = parameters_dict

    return sweep_config

