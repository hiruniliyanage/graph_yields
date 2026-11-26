import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter

# Manually setting up the adjacency matrix for a star graph with 10 nodes
matrix_A = np.array ([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
                      [0, 0, 0, 0, 0, 0, 0, 0, 0 ,1], 
                      [1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,0]
                      ])
# Converting the adjacency matrix to a NetworkX graph and plotting it
G = nx.from_numpy_array(matrix_A, create_using=nx.Graph)
hub = 9
leaves = sorted(v for v in G.nodes() if v != hub)   # [0, 1, ..., 8]
n = len(leaves)

pos = {hub: (0, 0)}
for i, v in enumerate(leaves):
    angle = np.pi/2 - 2*np.pi*i/n        # start at the top (90°), step clockwise
    pos[v] = (np.cos(angle), np.sin(angle))

node_sizes  = [2000 if v == hub else 1000 for v in G.nodes()]
node_colors = ["tomato" if v == hub else "lightblue" for v in G.nodes()]

plt.figure()
nx.draw(G, with_labels=True, node_size=node_sizes, node_color=node_colors)
plt.show()

# Finding the eigenvalues of the adjacency matrix
N = matrix_A.shape[0]
lambda_ = np.linalg.eigvalsh(matrix_A)
print("eigenvalues:", np.round(lambda_, 2))

# Grouping the eigenvalues and finding their multiplicities and DOS
mu, m = np.unique(np.round(lambda_, 2), return_counts=True)
w = m / N                                    # DOS weight for each eigenvalue

for x, k, p in zip(mu, m, w):
    print(f"lambda = {x:.1f}   multiplicity = {k}   DOS = {p:.2f}")

plt.stem(mu, w)
plt.xlabel("All possible eigenvalues")
plt.xlim(-4, 4)
plt.ylim(0, 0.8)
plt.ylabel("Density of States (DOS)")
plt.title("The Eigenvalue Distribution for a Star Graph")
plt.show()

# Evaluate eigenvectors for the star graph
eigenvalues, eigenvectors = np.linalg.eig(matrix_A)
print("Eigenvalues:", np.round(eigenvalues, 2))
print("Eigenvectors:",np.round(eigenvectors, 2))

# Evaluate dipole moments for the star graph
dipole_moments = []
for i in range(len(eigenvalues)):
    dipole_moment = np.sum(eigenvectors[:, i] * np.arange(len(eigenvectors[:, i])))
    dipole_moments.append(dipole_moment)
print("Dipole moments:", np.round(dipole_moments, 2))

# Repeating the process for Graph 26 Name: C_5 + K_2 from House of Graphs
matrix_26 = np.array ([[0, 1, 1, 0, 0, 1, 1],
                      [1, 0, 0, 1, 0, 1, 1],
                      [1, 0, 0, 0, 1, 1, 1],
                      [0, 1, 0, 0, 1, 1, 1],
                      [0, 0, 1, 1, 0, 1, 1],
                      [1, 1, 1, 1, 1, 0, 1],
                      [1, 1, 1, 1, 1, 1, 0]
                      ])
# Converting the adjacency matrix to a NetworkX graph and plotting it
G2 = nx.from_numpy_array(matrix_26, create_using=nx.Graph)
nx.draw(G2, with_labels=True)
plt.show()

# Finding the eigenvalues of the adjacency matrix
N2 = matrix_26.shape[0]
lambda_ = np.linalg.eigvalsh(matrix_26)
print("eigenvalues:", np.round(lambda_, 2))

# Grouping the eigenvalues and finding their multiplicities and DOS
mu2, m2 = np.unique(np.round(lambda_, 2), return_counts=True)
w2 = m2/ N2                                    # DOS weight for each eigenvalue

for x, k, p in zip(mu2, m2, w2):
    print(f"lambda = {x:.1f}   multiplicity = {k}   DOS = {p:.2f}")

plt.stem(mu2, w2)
plt.xlabel("All possible eigenvalues")
plt.xlim(-4, 4)
plt.ylim(0, 0.3)
plt.ylabel("Density of States (DOS)")
plt.title("The Eigenvalue Distribution for Graph 26: C_5 + K_2")
plt.show()

# Repeating the process for a three node chain 

matrix_chain = np.array ([
                      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                      [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
                      [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
                      [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0], 
                      [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0], 
                      [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0], 
                      [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0], 
                      [0, 0, 0, 0, 0, 0, 0, 1, 0 ,1, 0], 
                      [0, 0, 0, 0, 0, 0, 0, 0, 1 ,0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0 ,1, 0]
                      ])
# Converting the adjacency matrix to a NetworkX graph and plotting it
G3 = nx.from_numpy_array(matrix_chain, create_using=nx.Graph)
nx.draw(G3, with_labels=True)
plt.show()

# Finding the eigenvalues of the adjacency matrix
N3 = matrix_chain.shape[0]
lambda_3 = np.linalg.eigvalsh(matrix_chain)
print("eigenvalues:", np.round(lambda_3, 2))

# Grouping the eigenvalues and finding their multiplicities and DOS
mu3, m3 = np.unique(np.round(lambda_3, 2), return_counts=True)
w3 = m3 / N3                                    # DOS weight for each eigenvalue

for x, k, p in zip(mu3, m3, w3):
    print(f"lambda = {x:.1f}   multiplicity = {k}   DOS = {p:.2f}")

plt.stem(mu3, w3)
plt.xlabel("All possible eigenvalues $\lambda$")
plt.xlim(-4, 4)
plt.ylim(0, 0.1)
plt.ylabel("Density of States (DOS)")
plt.title("The Eigenvalue Distribution for a Three-Node Chain")
plt.show()

