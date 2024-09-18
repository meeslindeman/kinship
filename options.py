from dataclasses import dataclass

@dataclass
class Options:
    """
    Class representing the options for the experiment.
    """

    # Data
    root: str = 'data/' 
    need_probs: str = 'dutch' # 'uniform', 'kemp', 'dutch
    number_of_graphs: int = 5000 # default: 3200
    generations: int = 3 # default: 3
    padding_len: int = 80 # Only used for old sequence generation
    edges_away: int = 3 # default: 2
    
    # Game
    distractors: int = 1 # default: 5
    set_up: str = 'relationship' # 'single', 'relationship'
    mode: str = 'gs' # 'gs', 'rf' (also set in arguments command line)

    # Agents
    embedding_size: int = 10 # default: 10
    heads: int = 4 # default: 4
    hidden_size: int = 20 # default: 20
    sender_cell: str = 'gru' # 'rnn', 'gru', 'lstm'
    layer: str = 'gat' # 'gat', 'transformer'
    max_len: int = 2 # default: 4
    gs_tau: float = 1.0 # default: 1.0

    # Training
    n_epochs: int = 100
    vocab_size: int = 8
    batch_size: int = 25
    random_seed: int = 42

    # Logging
    compute_topsim: bool = False
    messages: bool = False
    targets: bool = False
    ego_nodes: bool = False
    sequence: bool = False

    # Set this according to parameters in main.py
    def __str__(self):
        return f"{self.need_probs}"
