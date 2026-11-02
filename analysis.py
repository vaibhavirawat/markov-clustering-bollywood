"""Bollywood co-star network: build graph, run MCL, benchmark, sweep inflation.

Outputs: outputs/*.csv / *.json (tables used by the report and figures).
"""
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score

from mcl import mcl_graph

HERE = Path(__file__).parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
MIN_FILMS = 5          # keep actors with at least this many films
INFLATION = 1.8        # chosen after the sweep (see sweep())


def build_graph(min_films=MIN_FILMS, jaccard=True):
    """Co-star graph. Edge weight = shared films, or (jaccard=True) shared films /
    films by either actor. The Jaccard weight stops prolific hub actors from
    swallowing everyone, which is essential for MCL (see ablation in __main__)."""
    films = pd.read_csv(HERE / "data" / "films.csv")
    count, years = Counter(), defaultdict(list)
    for year, cast in zip(films.year, films.cast):
        for a in cast.split("|"):
            count[a] += 1
            years[a].append(year)
    keep = {a for a, c in count.items() if c >= min_films}
    G = nx.Graph()
    for cast in films.cast:
        actors = [a for a in cast.split("|") if a in keep]
        for i in range(len(actors)):
            for j in range(i + 1, len(actors)):
                u, v = actors[i], actors[j]
                w = G[u][v]["weight"] + 1 if G.has_edge(u, v) else 1
                G.add_edge(u, v, weight=w)
    giant = max(nx.connected_components(G), key=len)
    G = G.subgraph(giant).copy()
    for a in G:
        G.nodes[a]["films"] = count[a]
        G.nodes[a]["avg_year"] = float(np.mean(years[a]))
    if jaccard:
        for u, v, d in G.edges(data=True):
            d["shared"] = d["weight"]
            d["weight"] = d["shared"] / (count[u] + count[v] - d["shared"])
    return G


def mod(G, clusters):
    return nx.community.modularity(G, clusters, weight="weight")


def labels_of(G, clusters):
    lab = {n: i for i, c in enumerate(clusters) for n in c}
    return [lab[n] for n in G.nodes]


def kl_k_way(G, k, seed=0):
    """Kernighan-Lin recursive bisection down to k parts (largest split first)."""
    parts = [set(G.nodes)]
    while len(parts) < k:
        parts.sort(key=len, reverse=True)
        big = parts.pop(0)
        a, b = nx.community.kernighan_lin_bisection(G.subgraph(big), weight="weight", seed=seed)
        parts += [set(a), set(b)]
    return [sorted(p) for p in parts]


def spectral(G, k, seed=0):
    nodes = list(G.nodes)
    A = nx.to_numpy_array(G, nodelist=nodes, weight="weight")
    lab = SpectralClustering(k, affinity="precomputed", random_state=seed,
                             assign_labels="discretize").fit_predict(A)
    out = defaultdict(list)
    for n, l in zip(nodes, lab):
        out[l].append(n)
    return list(out.values())


def sweep(G, values=(1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2, 2.5, 3.0)):
    rows = []
    for r in values:
        t = time.perf_counter()
        clusters, res, _ = mcl_graph(G, inflation=r)
        rows.append(dict(inflation=r, clusters=len(clusters),
                         largest=max(map(len, clusters)),
                         singletons=sum(len(c) == 1 for c in clusters),
                         modularity=mod(G, clusters), iterations=res.iterations,
                         seconds=time.perf_counter() - t))
    return pd.DataFrame(rows)


def benchmark(G, mcl_clusters):
    k = len(mcl_clusters)
    methods = {}
    t = time.perf_counter(); mcl_clusters, _, _ = mcl_graph(G, inflation=INFLATION)
    methods["MCL (ours)"] = (mcl_clusters, time.perf_counter() - t)
    t = time.perf_counter(); coarse, _, _ = mcl_graph(G, inflation=1.4)
    methods["MCL (ours, coarse r=1.4)"] = (coarse, time.perf_counter() - t)
    for name, fn in [
        ("Kernighan-Lin (k-way)", lambda: kl_k_way(G, k)),
        ("Agglomerative (greedy modularity)", lambda: [sorted(c) for c in nx.community.greedy_modularity_communities(G, weight="weight")]),
        ("Spectral", lambda: spectral(G, k)),
        ("Louvain", lambda: [sorted(c) for c in nx.community.louvain_communities(G, weight="weight", seed=0)]),
    ]:
        t = time.perf_counter(); c = fn(); methods[name] = (c, time.perf_counter() - t)
    ref = labels_of(G, methods["Louvain"][0])
    rows = []
    for name, (c, secs) in methods.items():
        rows.append(dict(method=name, clusters=len(c), largest=max(map(len, c)),
                         modularity=mod(G, c),
                         coverage=nx.community.partition_quality(G, c)[0],
                         nmi_vs_louvain=normalized_mutual_info_score(ref, labels_of(G, c)),
                         seconds=secs))
    return pd.DataFrame(rows), methods


def describe_clusters(G, clusters, top=6):
    rows = []
    for i, c in enumerate(sorted(clusters, key=len, reverse=True)):
        deg = sorted(c, key=lambda n: G.degree(n, weight="weight"), reverse=True)
        rows.append(dict(cluster=i, size=len(c),
                         avg_year=round(float(np.mean([G.nodes[n]["avg_year"] for n in c])), 1),
                         top_actors=", ".join(deg[:top])))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    G = build_graph()
    print(f"graph: {G.number_of_nodes()} actors, {G.number_of_edges()} co-star edges, "
          f"density={nx.density(G):.4f}, avg clustering={nx.average_clustering(G):.3f}")
    # ablation: same MCL on raw co-appearance counts vs Jaccard weights
    Graw = build_graph(jaccard=False)
    ab = []
    for name, H in [("raw counts", Graw), ("Jaccard", G)]:
        for r in (1.6, 1.8, 2.0, 2.5):
            c, _, _ = mcl_graph(H, inflation=r)
            ab.append(dict(weights=name, inflation=r, clusters=len(c),
                           largest=max(map(len, c)), modularity=mod(H, c)))
    ab = pd.DataFrame(ab); ab.to_csv(OUT / "weight_ablation.csv", index=False)
    print(ab.round(3).to_string(index=False))
    sw = sweep(G); sw.to_csv(OUT / "inflation_sweep.csv", index=False)
    print(sw.round(3).to_string(index=False))
    clusters, res, _ = mcl_graph(G, inflation=INFLATION)
    print(f"\nMCL r={INFLATION}: {len(clusters)} clusters in {res.iterations} iterations")
    bench, methods = benchmark(G, clusters)
    bench.to_csv(OUT / "benchmark.csv", index=False)
    print(bench.round(3).to_string(index=False))
    desc = describe_clusters(G, clusters, top=8); desc.to_csv(OUT / "mcl_clusters.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
    print(desc.head(20).to_string(index=False))
    json.dump({str(i): c for i, c in enumerate(sorted(clusters, key=len, reverse=True))},
              open(OUT / "mcl_clusters.json", "w"), indent=1)
