import numpy as np
from sklearn.cluster import KMeans
import pyximport
pyximport.install(setup_args={'include_dirs':np.get_include()})
import spectral_aux
import warnings
warnings.simplefilter('ignore') # dont's show warnings when convert eigen_vectors from complex to real

def Mnn_graph(S: np.ndarray) -> np.ndarray:
    """
    S - n x m array containing m neigbors for each point
    return: G - n x n boolean array representing undericted graph of neigbourhood
    """
    n = S.shape[0]
    m = S.shape[1]
    G = np.full(shape=(n, n), fill_value=0, dtype=np.int)
    for i in range(n):
        for j in range(m):
            G[i, S[i, j]] = 1
            G[S[i, j], i] = 1
    
    visited = np.full(shape=n, fill_value=False, dtype=np.bool)
    stack = []
    vis_count = 0
    stack.append(0)
    visited[0] = True
    while len(stack) != 0:
        v = stack.pop()
        vis_count += 1
        for u in range(n):
            if G[v, u]==1:
                if not visited[u]:
                    visited[u] = True
                    stack.append(u)
    if vis_count != n: # incoherent graph -> make it coherent
        listOfFirstVertex = []
        C = np.full(shape=n, fill_value=0, dtype=np.int)
        stack = []
        cn = 0
        for i in range(n):
            if C[i] == 0:
                cn += 1
                listOfFirstVertex.append(i)
                stack.append(i)
                C[i] = cn
                while len(stack) > 0:
                    v = stack.pop()
                    for u in range(n):
                        if G[v, u]==1:
                            if C[u] == 0:
                                stack.append(u)
                                C[u] = cn
        for i in range(cn-1):
            u = listOfFirstVertex[i]
            v = listOfFirstVertex[i+1]
            G[u, v] = 1
            G[v, u] = 1
    return G    

def Laplacian_eigen(G: np.ndarray, k: np.int) -> np.ndarray:
    """
     G - n x n boolean array representing undericted graph of neigbourhood
     return: E - n x k array containing 2, 3, ..., k+1 eigen vectors of matrix L
    """
    n = G.shape[0]
    D = np.full(shape=(n, n), fill_value=0, dtype=np.int)
    L = np.empty(shape=(n, n), dtype=np.int)
    E = np.empty(shape=(n, k), dtype=np.double)
    for i in range(n):
        v_degree = 0
        for j in range(n):
            v_degree += G[i, j]
        D[i, i] = v_degree
    L = D - G
    w, v = np.linalg.eigh(L)
    E = v[:, 1:k+1].real
    return E


def spectral_clustering(X: np.ndarray, k: np.int, M: np.int) -> np.ndarray:
    """
    X - array representing n points
    k - number of clusters
    M - number of neigbours
    return: vector of predicted labels for n points
    """
    S = spectral_aux.Mnn(X, M)
    G = Mnn_graph(S)
    E = Laplacian_eigen(G, k)
    labels_pred = KMeans(n_clusters=k).fit_predict(E)
    return labels_pred
