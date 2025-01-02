from graph.kemp_build import get_graph
from graph.kemp_dataset import KempGraphDataset

g, nm = get_graph()

dataset = KempGraphDataset(root='data/uniform/')

for n in nm:
    print(n, dataset[0].x[nm[n]])

# print(dataset[0])
# print(dataset[0].x)