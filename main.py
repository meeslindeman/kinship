import argparse
import logging
import coloredlogs
from options import Options
from init import initialize_dataset_if_needed
from archs.run_series import run_experiment, run_series_experiments
from analysis.timer import timer

logging.basicConfig(level=logging.INFO)
coloredlogs.install(level='INFO')

@timer
def run_experiments(options_input):
    if isinstance(options_input, Options):
        results = run_experiment(options_input, f'results/{options_input.need_probs}')
    elif isinstance(options_input, list):
        results, _ = run_series_experiments(options_input, f'results/')
    else:
        raise ValueError("Invalid input for options_input")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiments based on the provided options.')
    parser.add_argument('--mode', type=str, choices=['continuous', 'rf', 'gs', 'vq'], default='continuous', help='Set training mode')
    parser.add_argument('--single', action='store_true', help='Run a single experiment')
    parser.add_argument('--prune_graph', action='store_true', help='Prune graph to bfs tree')
    parser.add_argument('--wandb', action='store_true', help='Log to wandb')

    args = parser.parse_args()

    opts = Options(mode=args.mode, prune_graph=args.prune_graph, log_wandb=args.wandb)

    initialize_dataset_if_needed(opts)

    if args.single:
        # Run a single experiment: set options in command line
        run_experiments(opts)
    else:
        # Run multiple experiments: set __str__ in Options and labels in plot.py accordingly
        multiple_options = [
            Options(mode=args.mode, prune_graph=args.prune_graph, log_wandb=args.wandb),
            Options(mode=args.mode, prune_graph=args.prune_graph, log_wandb=args.wandb),
        ]
        run_experiments(multiple_options)
