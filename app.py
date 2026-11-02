"""Streamlit demo: Markov Clustering on the Bollywood co-star network.

Run:  .venv/bin/streamlit run app.py
"""
import time

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import normalized_mutual_info_score

from analysis import build_graph, kl_k_way, labels_of, mod, spectral
from mcl import mcl, mcl_graph

st.set_page_config(page_title="MCL on Bollywood", page_icon="🎬", layout="wide")

BG, PANEL, TEXT, GOLD, PINK = "#1B1228", "#2B1B3D", "#F4EEF8", "#E0A526", "#C2255C"
PALETTE = ["#E0A526", "#C2255C", "#4DABF7", "#51CF66", "#F76707", "#9775FA",
           "#20C997", "#FF8787", "#74C0FC", "#FFD43B", "#E599F7", "#A9E34B"]
OTHER = "#6B6580"

st.markdown(
    f"""<style>
    .block-container {{padding-top: 1.6rem;}}
    h1, h2, h3 {{font-family: Cambria, Georgia, serif;}}
    div[data-testid="stMetric"] {{background:{PANEL}; padding:12px 16px; border-radius:10px;}}
    div[data-testid="stMetricValue"] {{color:{GOLD};}}
    </style>""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- data
@st.cache_data(show_spinner="Building co-star graph...")
def get_graph(min_films, jaccard):
    return build_graph(min_films=min_films, jaccard=jaccard)


@st.cache_data(show_spinner=False)
def run_mcl(min_films, jaccard, inflation, expansion):
    G = get_graph(min_films, jaccard)
    t = time.perf_counter()
    clusters, res, nodes = mcl_graph(G, inflation=inflation, expansion=expansion)
    secs = time.perf_counter() - t
    clusters = sorted(clusters, key=len, reverse=True)
    return clusters, res.iterations, res.converged, secs


@st.cache_data(show_spinner="Laying out the network...")
def get_layout(min_films, jaccard):
    """Fixed layout, built once from a reference clustering so that moving the
    inflation slider only recolours nodes instead of re-drawing the graph."""
    G = get_graph(min_films, jaccard)
    ref, _, _ = mcl_graph(G, inflation=1.8)
    cid = {n: i for i, c in enumerate(ref) for n in c}
    L = nx.Graph()
    L.add_nodes_from(G)
    for u, v, d in G.edges(data=True):
        L.add_edge(u, v, weight=d["weight"] * (25 if cid[u] == cid[v] else 0.4))
    return nx.spring_layout(L, weight="weight", k=0.6, iterations=300, seed=7)


def shared_films(G, u, v):
    d = G[u][v]
    return d.get("shared", d["weight"])


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🎬 MCL controls")
inflation = st.sidebar.slider("Inflation r", 1.3, 3.0, 1.8, 0.05,
                              help="Higher r = smaller, tighter clusters.")
expansion = st.sidebar.selectbox("Expansion power e", [2, 3, 4], index=0)
min_films = st.sidebar.slider("Min films per actor", 3, 15, 5)
weighting = st.sidebar.radio("Edge weights", ["Jaccard overlap", "Raw co-appearance count"])
jaccard = weighting.startswith("Jaccard")
if not jaccard:
    st.sidebar.warning("Raw counts let hub actors swallow everyone. Watch MCL collapse into one blob.")
st.sidebar.caption("Data: Wikipedia, List of Hindi films 2000 to 2024")

G = get_graph(min_films, jaccard)
clusters, iters, converged, secs = run_mcl(min_films, jaccard, inflation, expansion)
Q = mod(G, clusters)
cid = {n: i for i, c in enumerate(clusters) for n in c}

st.title("Who works with whom in Bollywood?")
st.caption("Markov Clustering (MCL), implemented from scratch, on a co-star network of Hindi films. "
           "IIS4140 Social Network Analysis")

m = st.columns(5)
m[0].metric("Actors", G.number_of_nodes())
m[1].metric("Co-star edges", f"{G.number_of_edges():,}")
m[2].metric("Clusters found", len(clusters))
m[3].metric("Modularity Q", f"{Q:.3f}")
m[4].metric(f"MCL time ({iters} iterations)", f"{secs * 1000:.0f} ms")
if not converged:
    st.warning("MCL did not fully converge for these settings.")

tab_net, tab_clu, tab_how, tab_bench, tab_actor = st.tabs(
    ["🌐 Live network", "🗂 Cluster explorer", "🧠 How MCL works", "📊 Benchmark", "🔎 Find an actor"])


# ---------------------------------------------------------------- tab 1: network
def network_figure(G, clusters, highlight=None, max_named=12):
    pos = get_layout(min_films, jaccard)
    fig = go.Figure()
    w = np.array([d["weight"] for *_, d in G.edges(data=True)])
    cut = np.percentile(w, 85)
    ex, ey = [], []
    for u, v, d in G.edges(data=True):
        if d["weight"] >= cut:
            ex += [pos[u][0], pos[v][0], None]
            ey += [pos[u][1], pos[v][1], None]
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", hoverinfo="skip", showlegend=False,
                             line=dict(color="rgba(200,190,220,0.10)", width=0.6)))

    def node_trace(members, color, name, showlegend=True):
        xs = [pos[n][0] for n in members]
        ys = [pos[n][1] for n in members]
        size = [7 + 0.35 * G.nodes[n]["films"] for n in members]
        hover = [f"<b>{n}</b><br>{G.nodes[n]['films']} films, avg year {G.nodes[n]['avg_year']:.0f}"
                 f"<br>cluster #{cid[n]}" for n in members]
        return go.Scatter(x=xs, y=ys, mode="markers", name=name, text=hover, hoverinfo="text",
                          showlegend=showlegend,
                          marker=dict(color=color, size=size, line=dict(color=BG, width=0.8)))

    for i, c in enumerate(clusters[:max_named]):
        top = max(c, key=lambda n: G.degree(n, weight="weight"))
        fig.add_trace(node_trace(c, PALETTE[i % len(PALETTE)], f"#{i} {top} ({len(c)})"))
    rest = [n for c in clusters[max_named:] for n in c]
    if rest:
        fig.add_trace(node_trace(rest, OTHER, f"{len(clusters) - max_named} smaller clusters ({len(rest)})"))
    if highlight is not None:
        fig.add_trace(go.Scatter(x=[pos[highlight][0]], y=[pos[highlight][1]], mode="markers+text",
                                 text=[highlight], textposition="top center", showlegend=False,
                                 hoverinfo="skip", textfont=dict(color="white", size=14),
                                 marker=dict(size=26, color="rgba(0,0,0,0)", line=dict(color="white", width=3))))
    fig.update_layout(height=640, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor=BG, plot_bgcolor=BG,
                      font=dict(color=TEXT), legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
                      xaxis=dict(visible=False), yaxis=dict(visible=False), hoverlabel=dict(font_size=13))
    return fig


with tab_net:
    st.caption("Drag the inflation slider in the sidebar: the layout stays fixed and only the colours change. "
               "Click legend entries to hide or isolate clusters.")
    st.plotly_chart(network_figure(G, clusters))


# ---------------------------------------------------------------- tab 2: cluster explorer
with tab_clu:
    rows = []
    for i, c in enumerate(clusters):
        deg = sorted(c, key=lambda n: G.degree(n, weight="weight"), reverse=True)
        rows.append({"cluster": i, "size": len(c),
                     "avg year": round(float(np.mean([G.nodes[n]["avg_year"] for n in c])), 1),
                     "top actors": ", ".join(deg[:6])})
    df = pd.DataFrame(rows)
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Clusters")
        st.dataframe(df, hide_index=True, width="stretch", height=420,
                     column_config={"size": st.column_config.ProgressColumn(
                         "size", min_value=0, max_value=int(df["size"].max()), format="%d")})
    with right:
        st.subheader("Era of each cluster")
        big = df[df["size"] >= 5].head(15)
        era = go.Figure(go.Bar(x=big["avg year"], y=[f"#{c}" for c in big["cluster"]], orientation="h",
                               marker_color=GOLD, text=big["avg year"], textposition="outside"))
        era.update_layout(height=420, margin=dict(l=0, r=30, t=10, b=0), paper_bgcolor=BG, plot_bgcolor=BG,
                          font=dict(color=TEXT), yaxis=dict(autorange="reversed", type="category"),
                          xaxis=dict(range=[2000, 2025], title="average year of films"))
        st.plotly_chart(era)
    st.subheader("Members of a cluster")
    pick = st.selectbox("Cluster", df["cluster"], format_func=lambda i: f"#{i}  ({df.loc[i, 'size']} actors)  {df.loc[i, 'top actors'].split(',')[0]}")
    members = sorted(clusters[pick], key=lambda n: G.degree(n, weight="weight"), reverse=True)
    st.dataframe(pd.DataFrame({"actor": members,
                               "films": [G.nodes[n]["films"] for n in members],
                               "avg year": [round(G.nodes[n]["avg_year"], 1) for n in members]}),
                 hide_index=True, width="stretch")


# ---------------------------------------------------------------- tab 3: how MCL works
with tab_how:
    st.markdown("A random walk stays inside a dense group and rarely crosses the sparse links between groups. "
                "**Expansion** squares the transition matrix (flow spreads), **inflation** raises entries to power r "
                "(strong flow grows, weak flow dies). Repeat until nothing changes.")
    demo_inf = st.slider("Inflation for this demo", 1.5, 3.0, 2.0, 0.1, key="demo_inf")
    K = nx.karate_club_graph()
    A = nx.to_numpy_array(K, weight=None)
    res = mcl(A, inflation=demo_inf, keep_history=True)
    step = st.slider("Iteration", 0, len(res.history) - 1, 0, key="demo_step",
                     help="0 = normalised adjacency matrix, last = converged")
    order = np.argsort(res.labels, kind="stable")
    M = res.history[step][np.ix_(order, order)]
    labels_now = res.history[step].argmax(axis=0)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Flow matrix")
        heat = go.Figure(go.Heatmap(z=M, colorscale="Magma", reversescale=True, zmin=0, zmax=min(1, float(M.max())),
                                    showscale=False))
        heat.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor=BG,
                           xaxis=dict(visible=False), yaxis=dict(visible=False, autorange="reversed"))
        st.plotly_chart(heat)
    with c2:
        st.subheader("Zachary's Karate Club")
        kp = nx.spring_layout(K, seed=3)
        uniq = {l: i for i, l in enumerate(sorted(set(labels_now)))}
        kf = go.Figure()
        kf.add_trace(go.Scatter(x=[x for u, v in K.edges for x in (kp[u][0], kp[v][0], None)],
                                y=[y for u, v in K.edges for y in (kp[u][1], kp[v][1], None)],
                                mode="lines", line=dict(color="rgba(200,190,220,0.25)", width=1),
                                hoverinfo="skip", showlegend=False))
        kf.add_trace(go.Scatter(x=[kp[n][0] for n in K], y=[kp[n][1] for n in K], mode="markers+text",
                                text=[str(n) for n in K], textfont=dict(size=9, color=BG),
                                marker=dict(size=22, color=[PALETTE[uniq[labels_now[n]] % len(PALETTE)] for n in K]),
                                hoverinfo="skip", showlegend=False))
        kf.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor=BG, plot_bgcolor=BG,
                         xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(kf)
    st.caption(f"Iteration {step} of {len(res.history) - 1}: {len(uniq)} cluster(s) so far. "
               f"Final: {res.labels.max() + 1} clusters, converged={res.converged}.")
    if res.chaos:
        ch = go.Figure(go.Scatter(x=list(range(1, len(res.chaos) + 1)), y=res.chaos, mode="lines+markers",
                                  line=dict(color=GOLD)))
        ch.update_layout(height=220, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor=BG, plot_bgcolor=BG,
                         font=dict(color=TEXT), title="Chaos (distance from converged), log scale",
                         yaxis=dict(type="log"), xaxis=dict(title="iteration"))
        st.plotly_chart(ch)


# ---------------------------------------------------------------- tab 4: benchmark
@st.cache_data(show_spinner="Running all algorithms...")
def run_benchmark(min_films, jaccard, inflation, expansion):
    G = get_graph(min_films, jaccard)
    mc, _, _, secs_mcl = run_mcl(min_films, jaccard, inflation, expansion)
    k = len(mc)
    algos = {"MCL (ours)": (mc, secs_mcl)}
    for name, fn in [
        ("Kernighan-Lin (k-way)", lambda: kl_k_way(G, k)),
        ("Agglomerative (greedy modularity)", lambda: [sorted(c) for c in nx.community.greedy_modularity_communities(G, weight="weight")]),
        ("Spectral", lambda: spectral(G, k)),
        ("Louvain", lambda: [sorted(c) for c in nx.community.louvain_communities(G, weight="weight", seed=0)]),
    ]:
        t = time.perf_counter()
        c = fn()
        algos[name] = (c, time.perf_counter() - t)
    ref = labels_of(G, algos["Louvain"][0])
    out = []
    for name, (c, s) in algos.items():
        out.append({"method": name, "clusters": len(c), "largest": max(map(len, c)),
                    "modularity": round(mod(G, c), 3),
                    "NMI vs Louvain": round(normalized_mutual_info_score(ref, labels_of(G, c)), 3),
                    "seconds": round(s, 3)})
    return pd.DataFrame(out)


with tab_bench:
    st.markdown("All methods run live on the same graph and weights. Kernighan-Lin and spectral are given MCL's "
                "number of clusters, since they cannot choose k themselves.")
    if st.button("Run benchmark", type="primary"):
        st.session_state["bench"] = run_benchmark(min_films, jaccard, inflation, expansion)
    bench = st.session_state.get("bench")
    if bench is not None:
        b1, b2 = st.columns(2)
        colors = [PINK if "MCL" in m_ else "#9AA5B1" for m_ in bench["method"]]
        with b1:
            f = go.Figure(go.Bar(x=bench["modularity"], y=bench["method"], orientation="h",
                                 marker_color=colors, text=bench["modularity"], textposition="outside"))
            f.update_layout(title="Modularity (higher is better)", height=340, paper_bgcolor=BG, plot_bgcolor=BG,
                            font=dict(color=TEXT), yaxis=dict(autorange="reversed"), xaxis=dict(range=[0, max(bench["modularity"]) * 1.3]),
                            margin=dict(l=0, r=30, t=40, b=0))
            st.plotly_chart(f)
        with b2:
            f = go.Figure(go.Bar(x=bench["seconds"], y=bench["method"], orientation="h",
                                 marker_color=colors, text=bench["seconds"], textposition="outside"))
            f.update_layout(title="Runtime, seconds (log scale)", height=340, paper_bgcolor=BG, plot_bgcolor=BG,
                            font=dict(color=TEXT), yaxis=dict(autorange="reversed", showticklabels=False),
                            xaxis=dict(type="log", range=[float(np.log10(bench["seconds"].min() / 2)), float(np.log10(bench["seconds"].max() * 5))]),
                            margin=dict(l=0, r=30, t=40, b=0))
            st.plotly_chart(f)
        st.dataframe(bench, hide_index=True, width="stretch")
        st.caption("Louvain and greedy agglomerative score higher modularity because they optimise it directly. "
                   "MCL needs no preset number of clusters.")
    else:
        st.info("Click **Run benchmark** to compare MCL with Kernighan-Lin, agglomerative, spectral and Louvain.")


# ---------------------------------------------------------------- tab 5: actor lookup
with tab_actor:
    default = "Shah Rukh Khan" if "Shah Rukh Khan" in G else sorted(G.nodes)[0]
    names = sorted(G.nodes)
    actor = st.selectbox("Pick an actor", names, index=names.index(default))
    i = cid[actor]
    mates = [n for n in clusters[i] if n != actor]
    nbrs = sorted(G[actor], key=lambda n: shared_films(G, actor, n), reverse=True)[:10]
    a, b = st.columns(2)
    with a:
        st.subheader(f"{actor}")
        st.write(f"**{G.nodes[actor]['films']} films**, average year {G.nodes[actor]['avg_year']:.0f}.")
        st.write(f"Belongs to cluster **#{i}** with {len(clusters[i])} actors"
                 f"{' (alone)' if not mates else ''}.")
        st.markdown("**Biggest names in this cluster:** " + (", ".join(sorted(mates, key=lambda n: G.degree(n, weight='weight'), reverse=True)[:12]) or "none"))
        st.markdown("**Most shared films with:**")
        st.dataframe(pd.DataFrame({"co-star": nbrs,
                                   "shared films": [int(shared_films(G, actor, n)) for n in nbrs],
                                   "same cluster": ["yes" if cid[n] == i else "no" for n in nbrs]}),
                     hide_index=True, width="stretch")
    with b:
        st.plotly_chart(network_figure(G, clusters, highlight=actor))
