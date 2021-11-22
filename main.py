import networkx as nx
import numpy as np
import time
import matplotlib.pyplot as plt

number_nodes = 10
probability = 1

G_RE = nx.erdos_renyi_graph(n=number_nodes, p=probability, seed=int(time.time()))

edges = list(G_RE.edges())
indices_edges = [i for i in range(len(edges))]
selected_edges = list(np.random.choice(indices_edges, 3))

nodes = list(G_RE.nodes())
indices_nodes = [i for i in range(len(nodes))]
selected_nodes = list(np.random.choice(indices_nodes, 3))

print("The graph:", G_RE)

print("Edges: ", edges)
print("Selected edges:", selected_edges)

print("Nodes: ", nodes)
print("Selected nodes:", selected_nodes)

for i in selected_edges:
    print("Log Edges:", edges[i][0], edges[i][1])
    # G_RE.remove_edge(edges[i][0], edges[i][1])

for i in selected_nodes:
    print("Log Nodes:", nodes[i])