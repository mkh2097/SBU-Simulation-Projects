import networkx as nx
import numpy as np
import time
import matplotlib.pyplot as plt

number_nodes = 10
probability = 0.5

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


connectivity = nx.is_connected(G_RE)
number_of_connected_comps = nx.number_connected_components(G_RE)
num_isolated_nodes = len(list(nx.isolates(G_RE)))

print("connectivity:", connectivity)
print("number_of_connected_comps:", number_of_connected_comps)
print("num_isolated_nodes:", num_isolated_nodes)

pos = nx.spring_layout(G_RE)
nx.draw(G_RE, pos=pos, with_labels=True)
plt.savefig('fig.png',bbox_inches='tight')