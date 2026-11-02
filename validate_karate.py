"""Sanity check: our MCL on Zachary's Karate Club vs the known 2-faction split."""
import networkx as nx
import numpy as np
from sklearn.metrics import normalized_mutual_info_score

from mcl import mcl_graph

G = nx.karate_club_graph()
truth = [0 if G.nodes[n]["club"] == "Mr. Hi" else 1 for n in G.nodes]
for r in (1.5, 2.0, 2.5, 3.0):
    clusters, res, nodes = mcl_graph(G, inflation=r, weight=None)
    q = nx.community.modularity(G, clusters)
    nmi = normalized_mutual_info_score(truth, res.labels)
    print(f"inflation={r}: {len(clusters)} clusters, modularity={q:.3f}, "
          f"NMI vs factions={nmi:.3f}, iterations={res.iterations}, converged={res.converged}")
