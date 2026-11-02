# Presentation outline (about 8 minutes, 9 slides)

**1. Title.** *Who works with whom in Bollywood? Markov Clustering on a co-star network.*

**2. The question.** Can an algorithm discover the "friend groups" of Hindi cinema from casts alone, with no labels and no preset number of groups?

**3. The data.**
- Scraped Wikipedia *List of Hindi films*, 2000 to 2024: 2,305 films, 2,060 actors.
- Graph: 512 actors with at least 5 films, 10,777 co-star edges.
- Edge weight is Jaccard overlap of films.

**4. How MCL works (the core slide).** Show `outputs/fig_mcl_steps.png`.
- A random walk stays inside a dense group and rarely crosses the sparse links between groups.
- Repeat two steps: **expansion** (matrix squared, flow spreads) and **inflation** (entries raised to power r, strong flow gets stronger, weak flow dies).
- It stops when the matrix stops changing. Rows that keep flow are attractors, and they define the clusters.
- Karate Club check: our implementation finds the 2 known factions.

**5. Result: the network.** Show `outputs/fig_network.png`. 51 clusters, found without being told k. Point at the left-to-right era gradient.

**6. What the clusters mean.** Show the table in the report (section 4) and `outputs/fig_eras.png`.
- Pankaj Tripathi / Sumeet Vyas group: recent OTT and indie cast (avg 2019).
- Ranveer Singh / Alia Bhatt / Arjun Kapoor group: mid-2010s young stars.
- Govinda / Kader Khan group: 90s-style veterans (avg 2009).
- Radhika Apte / Richa Chadha / Nawazuddin Siddiqui: actor-driven indie cinema.
- Say clearly that these labels are my interpretation of top actors.

**7. The inflation knob.** Show `outputs/fig_inflation_sweep.png`. Low r gives 1 blob, r = 1.4 gives 5 clusters, r = 1.8 gives 51, high r gives singletons.

**8. MCL vs the course algorithms.** Show `outputs/fig_benchmark.png`.
- Beats Kernighan-Lin and spectral at the same number of clusters, and is fast.
- Louvain and greedy agglomerative score higher modularity because they optimise it directly.
- MCL's advantage: no need to choose k.

**9. Lessons and limits.**
- Edge weights matter a lot: raw counts gave one 503-actor blob, Jaccard fixed it.
- Hard clusters only, but real actors overlap between circles.
- Wikipedia casts are partial.
- Future work: overlapping variants, and adding link prediction for casting recommendations.

## Likely questions

- **Why not Louvain?** It scores higher modularity, but it is a different course topic. MCL's selling point is flow simulation and no preset k.
- **How did you pick r?** Swept 1.3 to 3.0 and chose by modularity and readability. There is no ground truth, so it is a judgement call.
- **Is MCL guaranteed to converge?** Not formally, but in every run here the chaos measure reached 0 within 31 iterations.
- **Why Jaccard weights?** Hub actors with 70+ films otherwise dominate the random walk.
