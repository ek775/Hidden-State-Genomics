import networkx as nx
from networkx import Graph


# create 4105 nodes from vocabulary size
vocab_size = 4105
G = Graph()
G.add_nodes_from(range(vocab_size))

# create edges between nodes
for i in range(vocab_size):
    for j in range(i + 1, vocab_size):
        G.add_edge(i, j)
        print(f"Edge count: {G.number_of_edges()}")

print(nx.draw(G))
print(nx.info(G))
