# Language Emergence GNN
Language emergence project using graph neural networks.

Some modifications in EGG have to be made. In core/interactions.py:
line 209 -> aux_input[k] = _check_cat([x.aux_input[k] for x in interactions]) to aux_input[k] = None
line 218 -> aux_input=aux_input to aux_input=[x.aux_input for x in interactions]

Set options in options.py.
To run: main.py --single (optional --rf)


