import argparse
import logging
import coloredlogs
from options import Options
from archs.run_series import run_experiment, run_series_experiments
from analysis.plot import plot_experiment
from analysis.timer import timer

logging.basicConfig(level=logging.INFO)
coloredlogs.install(level='INFO')

@timer
def run_experiments(options_input):
    if isinstance(options_input, Options):
        results = run_experiment(options_input, f'results/{options_input.need_probs}')
        plot_experiment(results, mode='both', save=False)
    elif isinstance(options_input, list):
        results, _ = run_series_experiments(options_input, f'results/')
    else:
        raise ValueError("Invalid input for options_input")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiments based on the provided options.')
    parser.add_argument('--mode', type=str, choices=['rf', 'gs'], default='gs', help='Set training mode (gs or rf)')
    parser.add_argument('--single', action='store_true', help='Run a single experiment')

    args = parser.parse_args()

    if args.single:
        # Run a single experiment: set options in command line
        single_options = Options(mode=args.mode)
        run_experiments(single_options)
    else:
        # Run multiple experiments: set __str__ in Options and labels in plot.py accordingly
        multiple_options = [
            Options(need_probs='dutch'),
            Options(need_probs='kemp'),
            Options(need_probs='uniform')
        ]
        run_experiments(multiple_options)
