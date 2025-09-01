import networkx as nx
from networkx import MultiGraph
import numpy as np
import os, psutil

# construct adjacency matrix
vocab_size = 4105
perfect_adjacency = np.full((vocab_size, vocab_size), 10, dtype=np.int32)
np.fill_diagonal(perfect_adjacency, 0)
G = nx.from_numpy_array(perfect_adjacency, parallel_edges=True, create_using=MultiGraph)

print(G)
print("Is the graph connected?", nx.is_connected(G))
process = psutil.Process(os.getpid())
rss_bytes = process.memory_info().rss
rss_mb = rss_bytes / (1024 * 1024)
print("Memory usage of G:", rss_mb, "MB")
