# Language Emergence GNN
Language emergence project using graph neural networks.

## Usage
#### Running Experiments
Use `main.py` to run experiments with various configurations. You can either run a single experiment or a series of experiments.

The dataset will be automatically initialized if it doesn’t exist. This process is managed by main.py and does not require additional steps from the user.

When the dataset is created:

You’ll see a message indicating that the dataset is being initialized.
After the first initialization, the dataset will be reused for subsequent runs, unless manually deleted.

#### Example Commands
Run a Single Experiment:

- `python main.py --single --mode gs --wandb`
This command runs a single experiment in Gumbel-Softmax (gs) mode and logs to W&B.

Run Multiple Experiments:

- `python main.py --mode rf --wandb`
This command runs multiple experiments in Reinforce (rf) mode with W&B logging enabled.
Make sure to set a list of options manually in `main.py`.

## Options
The script accepts several command-line arguments to control the experiment settings:

- `--mode`: Specifies the training mode (continuous, rf for Reinforce, or gs for Gumbel-Softmax). Default is gs.
- `--single`: Runs a single experiment. If not set, the script will run a series of experiments.
- `--prune_graph`: Prunes the input graph to a BFS tree.
- `--wandb`: Enables logging to Weights & Biases.

Experiment configurations, including options such as `with_vq` (enabling the VQ layer), are controlled via the `Options` class.