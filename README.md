# Language Emergence GNN
Language emergence project using graph neural networks.

Initialise dataset (or use pre-generated data folder): 
init.py (Set amount of children in graph/build.py)

Set general hyperparameters for experiments:
options.py

Run experiments by:
main.py for a series of experiments, set in 
multiple_options = [
    Options(agents='dual', generations=2),
    Options(agents='dual', generations=3),
    Options(agents='dual', generations=4)
]

main.py --single for options set in options.py



