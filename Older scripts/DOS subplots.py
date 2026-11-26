import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ---------- Adjacency matrices ----------
# Star graph, N = 10: hub = node 9, leaves = nodes 0-8
matrix_A = np.zeros((10, 10), dtype=int)
matrix_A[9, :9] = 1
matrix_A[:9, 9] = 1

# House of Graphs, Graph 26: C_5 + K_2
matrix_26 = np.array([[0, 1, 1, 0, 0, 1, 1],
                      [1, 0, 0, 1, 0, 1, 1],
                      [1, 0, 0, 0, 1, 1, 1],
                      [0, 1, 0, 0, 1, 1, 1],
                      [0, 0, 1, 1, 0, 1, 1],
                      [1, 1, 1, 1, 1, 0, 1],
                      [1, 1, 1, 1, 1, 1, 0]])

# Chain (path graph) with 11 nodes: node i joined to node i+1
N_chain = 11
matrix_chain = np.zeros((N_chain, N_chain), dtype=int)
for i in range(N_chain - 1):
    matrix_chain[i, i + 1] = matrix_chain[i + 1, i] = 1

# ---------- Layouts ----------
def star_layout(hub, leaves):
    pos = {hub: (0, 0)}
    for i, v in enumerate(leaves):
        angle = np.pi/2 - 2*np.pi*i/len(leaves)   # clockwise from the top
        pos[v] = (np.cos(angle), np.sin(angle))
    return pos

pos_star = star_layout(9, list(range(9)))

cycle = [0, 1, 3, 4, 2]          # 5-cycle of Graph 26 in cycle order (0-based labels)
pos_26 = {v: (np.cos(np.pi/2 - 2*np.pi*i/5), np.sin(np.pi/2 - 2*np.pi*i/5))
          for i, v in enumerate(cycle)}
pos_26[5], pos_26[6] = (-0.3, 0), (0.3, 0)

pos_chain = {i: (i, 0) for i in range(N_chain)}   # nodes in a straight line


# ---------- DOS ----------
def dos(A):
    """Distinct adjacency eigenvalues and their DOS weights m(lambda)/N."""
    lam = np.linalg.eigvalsh(A)                                   # real symmetric -> real, sorted
    mu, m = np.unique(np.round(lam, 8), return_counts=True)       # group degenerate eigenvalues
    return mu, m, m / A.shape[0]

# Last entry of each tuple: nodes drawn in the highlight colour
graphs = [("Star graph, N = 10",            matrix_A,     pos_star,  [9]),
          ("Graph 26: $C_5 + K_2$, N = 7",  matrix_26,    pos_26,    [5, 6]),
          (f"Chain, N = {N_chain}",         matrix_chain, pos_chain, [0])]

# ---------- Figure: graphs on top, DOS underneath ----------
fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained",
                         gridspec_kw={"height_ratios": [1, 1.2]})

for col, (title, A, pos, highlight) in enumerate(graphs):
    G = nx.from_numpy_array(A)
    mu, m, w = dos(A)

    print(title)
    for x, k, p in zip(mu, m, w):
        print(f"  lambda = {x:+.4f}   multiplicity = {k}   DOS = {p:.3f}")
    print("  weights sum to 1:", np.isclose(w.sum(), 1))

    # Top row: graph drawing
    ax = axes[0, col]
    colors = ["tomato" if v in highlight else "lightblue" for v in G.nodes()]
    nx.draw(G, pos, ax=ax, with_labels=True, node_size=400, node_color=colors, font_size=9)
    ax.set_title(title, pad=15)     # pad: gap (in points) between title and panel
    if A is matrix_chain:
        ax.set_ylim(-1, 1)          # a straight line: give it vertical room instead of equal aspect
    else:
        ax.set_aspect("equal")      # keep circles round

    # Bottom row: DOS
    ax = axes[1, col]
    markerline, stemlines, baseline = ax.stem(mu, w, markerfmt=" ")   # markerfmt=" ": no dot on top
    stemlines.set_linewidth(3)                                        # thicker eigenvalue lines
    ax.set(xlabel=r"Eigenvalue $\lambda$", xlim=(-5, 5), ylim=(0, 1))

axes[1, 0].set_ylabel(r"DOS  $m(\lambda)/N$")
fig.suptitle("Eigenvalue Distributions of Selected Graphs", fontsize=14, weight="bold")
fig.get_layout_engine().set(h_pad=0.15)   # extra vertical space between rows and below the main title
fig.savefig("dos_comparison.png", dpi=150)
plt.show()