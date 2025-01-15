import wandb, logging, coloredlogs
from pprint import pprint
from options import Options
from init import initialize_dataset_if_needed
from analysis.save import results_to_dataframes
from archs.game import get_game
from archs.dataloader import get_loaders
from archs.train import perform_training
from graph.kemp_dataset import KempGraphDataset
from sweep_config import sweep_config_init

logging.basicConfig(level=logging.INFO)
coloredlogs.install(level='INFO')


def set_options(config):
    '''
    Copies config values (from current wandb run) into Options class.
    Note: the wandb config needs to use exactly the same hyperapameter names required in Options.
    :param config: the wandb config, sampled from sweep agent
    :return: instance of Options class
    '''

    try:
        opts = Options(**config)
    except TypeError:
        print("The wandb config doesn't match our Options class. "
              "Check hyperaparameter names (missing or mispelled). ")

    #Add this one separately, as it's logged even when unused
    setattr(opts, 'codebook_size', config['vocab_size'])

    #Eval opts
    setattr(opts, 'evaluation', True)
    setattr(opts, 'evaluation_interval', 10)

    return opts

def run_sweep_experiment(sweep_config: dict = None, save: bool = True):

    with wandb.init(config=sweep_config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        pprint(config, compact=False)

        #Transfer current config into Options class
        opts = set_options(config)
        target_folder=f'results/{opts.need_probs}'

        #Log
        logging.info(f"Running {str(config)}")

        #Data loading
        initialize_dataset_if_needed(opts)
        dataset = KempGraphDataset(root=opts.root+opts.need_probs, prune=opts.prune_graph)
        print(f"Dataset: {opts.root+opts.need_probs}")

        #Train
        train_loader, valid_loader, eval_loader = get_loaders(opts, dataset)
        game = get_game(opts, dataset.num_node_features)
        results, trainer = perform_training(opts, train_loader, valid_loader, eval_loader, game) #no eval data
        metrics_df, counts_df, evaluation_df = results_to_dataframes(results, opts, target_folder, save)

        #Log to wandb
        for _, row in metrics_df.iterrows():
            log_data = {
                'epoch': row['epoch'],
                f"metrics/{row['mode']}/loss": row['loss'],
                f"metrics/{row['mode']}/accuracy": row['acc'],
            }
            if row['mode'] == 'train':
                log_data['metrics/evaluation/eval_acc']=row.get('eval_acc', None)
            wandb.log(log_data)

    return metrics_df

def delete_failed_runs(project):
    api = wandb.Api()
    runs = api.runs(project)
    for run in runs:
        if run.state == 'failed':
            print(f'Deleting run {run.id} - {run.name}')
            run.delete()

if __name__ == "__main__":

    try:
        sweep_config = sweep_config_init()
        sweep_id = wandb.sweep(sweep_config, project='kinship')
        wandb.agent(sweep_id, run_sweep_experiment)#, count=1)
    except BrokenPipeError:
        print('Failed to run sweep')
    finally:
        delete_failed_runs('kinship')
