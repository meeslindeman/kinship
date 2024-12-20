from dataclasses import dataclass
import torch

@dataclass
class Options:
    """
    Class representing the options for the experiment.
    """
    # Data
    root: str = 'data/'
    need_probs: str = 'uniform' # 'uniform', 'kemp', 'dutch -> see need prob plots
    number_of_graphs: int = 5000 # default: 3200
    generations: int = 3 # Depricated: used for random graph generation
    padding_len: int = 80 # Depricated: used for topsim sequence generation
    edges_away: int = 3 # Depricated: used for random graph generation
    prune_graph: bool = False # True for pruning graph to bfs tree (keep only paths from ego to other nodes)

    # Game
    distractors: int = 50 # default: 5
    mode: str = 'continuous' # 'gs', 'rf' 'vq'

    # Agents
    embedding_size: int = 200 # default: 80
    heads: int = 1 # default: 4
    hidden_size: int = 200 # default: 20
    sender_cell: str = 'gru' # 'rnn', 'gru', 'lstm'
    layer: str = 'rgcn' # 'gat', 'transform', 'rgcn'
    max_len: int = 1 # default: 1
    gs_tau: float = 1.0 # default: 1.0

    # Training
    n_epochs: int = 2000
    vocab_size: int = 100
    batch_size: int = 50
    random_seed: int = 43
    learning_rate: float = 1e-3

    # Logging
    compute_topsim: bool = False
    messages: bool = True
    ego_nodes: bool = False
    sequence: bool = False
    log_wandb: bool = False

    # Evaluation
    evaluation: bool = True
    eval_distractors: int = None
    eval_batch_size: int = 1
    evaluation_interval: int = 5
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set this according to parameters in main.py
    def __str__(self):
        return f"{self.need_probs}"
