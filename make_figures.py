"""Generate all figures into outputs/ (run analysis.py first)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from analysis import INFLATION, OUT, benchmark, build_graph
from mcl import mcl, mcl_graph

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
PALETTE = plt.get_cmap("tab20").colors


def fig_network(G, clusters):
    clusters = sorted(clusters, key=len, reverse=True)
    top = 12
    color = {n: "#c8c8c8" for n in G}
    for i, c in enumerate(clusters[:top]):
        for n in c:
            color[n] = PALETTE[i]
    # cluster-aware layout: pull same-cluster actors together, loosen cross links
    cid = {n: i for i, c in enumerate(clusters) for n in c}
    L = nx.Graph()
    L.add_nodes_from(G)
    for u, v, d in G.edges(data=True):
        L.add_edge(u, v, weight=d["weight"] * (25 if cid[u] == cid[v] else 0.4))
    pos = nx.spring_layout(L, weight="weight", k=0.6, iterations=300, seed=7)
    fig, ax = plt.subplots(figsize=(15, 11))
    cut = np.percentile([d["weight"] for *_, d in G.edges(data=True)], 85)
    edges = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] >= cut]
    nx.draw_networkx_edges(G, pos, edgelist=edges, alpha=0.10, width=0.6, ax=ax)
    sizes = [20 + 9 * G.nodes[n]["films"] for n in G]
    nx.draw_networkx_nodes(G, pos, node_color=[color[n] for n in G], node_size=sizes,
                           linewidths=0.4, edgecolors="white", ax=ax)
    for i, c in enumerate(clusters[:top]):
        for n in sorted(c, key=lambda n: G.nodes[n]["films"], reverse=True)[:2]:
            ax.text(*pos[n], n, fontsize=8.5, ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    ax.set_title(f"Bollywood co-star network ({G.number_of_nodes()} actors) - "
                 f"{len(clusters)} MCL clusters (inflation {INFLATION}); top {top} coloured", fontsize=14)
    ax.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "fig_network.png", dpi=170); plt.close(fig)


def fig_sweep():
    sw = pd.read_csv(OUT / "inflation_sweep.csv")
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(sw.inflation, sw.modularity, "o-", color=PALETTE[0], label="modularity")
    ax1.set_xlabel("inflation parameter r"); ax1.set_ylabel("modularity Q", color=PALETTE[0])
    ax2 = ax1.twinx(); ax2.spines["right"].set_visible(True)
    ax2.plot(sw.inflation, sw.clusters, "s--", color=PALETTE[6], label="clusters")
    ax2.set_yscale("log"); ax2.set_ylabel("number of clusters (log)", color=PALETTE[6])
    ax1.axvline(INFLATION, color="grey", ls=":"); ax1.text(INFLATION + 0.03, 0.02, "chosen r", color="grey")
    ax1.set_title("Inflation controls cluster granularity")
    fig.tight_layout(); fig.savefig(OUT / "fig_inflation_sweep.png", dpi=170); plt.close(fig)


def fig_benchmark():
    b = pd.read_csv(OUT / "benchmark.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    cols = [PALETTE[0] if "MCL" in m else "#9aa5b1" for m in b.method]
    axes[0].barh(b.method, b.modularity, color=cols); axes[0].set_xlabel("modularity Q (higher = better)")
    axes[0].invert_yaxis()
    axes[1].barh(b.method, b.seconds, color=cols); axes[1].set_xscale("log")
    axes[1].set_xlabel("runtime, seconds (log)"); axes[1].invert_yaxis(); axes[1].set_yticklabels([])
    fig.suptitle("MCL vs the course's clustering algorithms (same graph, same weights)")
    fig.tight_layout(); fig.savefig(OUT / "fig_benchmark.png", dpi=170); plt.close(fig)


def fig_steps():
    """Show the flow matrix converging on Zachary's karate club (34 nodes)."""
    K = nx.karate_club_graph()
    A = nx.to_numpy_array(K, weight=None)
    # order nodes by final cluster so blocks are visible
    res = mcl(A, inflation=2.0, keep_history=True)
    order = np.argsort(res.labels, kind="stable")
    shots = sorted({0, 1, 3, len(res.history) - 1})
    fig, axes = plt.subplots(1, len(shots) + 1, figsize=(4 * (len(shots) + 1), 4.2))
    for ax, s in zip(axes, shots):
        M = res.history[s][np.ix_(order, order)]
        ax.imshow(M, cmap="magma_r", vmin=0, vmax=min(1, M.max()), interpolation="nearest")
        ax.set_title("start (normalised adjacency)" if s == 0 else
                     f"after iteration {s}" + (" (converged)" if s == len(res.history) - 1 else ""), fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    pos = nx.spring_layout(K, seed=3)
    nx.draw_networkx(K, pos, node_color=[PALETTE[l] for l in res.labels], node_size=140,
                     font_size=6, edge_color="#bbbbbb", ax=axes[-1])
    axes[-1].set_title(f"{res.labels.max() + 1} clusters found", fontsize=10); axes[-1].axis("off")
    fig.suptitle("MCL in action on Zachary's Karate Club: expansion + inflation until flow settles")
    fig.tight_layout(); fig.savefig(OUT / "fig_mcl_steps.png", dpi=170); plt.close(fig)


def fig_eras(G, clusters):
    clusters = sorted(clusters, key=len, reverse=True)[:8]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    data = [[G.nodes[n]["avg_year"] for n in c] for c in clusters]
    ax.boxplot(data, vert=False, patch_artist=True,
               boxprops=dict(facecolor=PALETTE[1]), medianprops=dict(color="black"))
    ax.set_yticklabels([f"#{i} (n={len(c)})" for i, c in enumerate(clusters)])
    ax.invert_yaxis(); ax.set_xlabel("average year of an actor's films")
    ax.set_title("Clusters line up with eras of Hindi cinema")
    fig.tight_layout(); fig.savefig(OUT / "fig_eras.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    G = build_graph()
    clusters, _, _ = mcl_graph(G, inflation=INFLATION)
    fig_network(G, clusters); fig_sweep(); fig_benchmark(); fig_steps(); fig_eras(G, clusters)
    print("figures written to", OUT)
