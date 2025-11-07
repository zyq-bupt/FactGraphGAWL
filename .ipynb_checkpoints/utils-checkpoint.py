import numpy as np
import networkx as nx

def nx_to_arrays(G: nx.Graph, etype_to_id: dict | None):
    """
    返回:
      - indptr, indices: CSR 邻接 (无向图则双向加边)
      - edges_u, edges_v, edges_etype: 边列表（每条边一次，建议 u<v 去重）
      - init_labels: 初始节点标签 (用度或已有标签)
    """
    nodes = np.array(list(G.nodes()), dtype=np.int64)
    node2idx = {n:i for i,n in enumerate(nodes)}
    n = nodes.size

    # CSR 邻接
    neighs = [[] for _ in range(n)]
    for u, v, d in G.edges(data=True):
        iu, iv = node2idx[u], node2idx[v]
        neighs[iu].append(iv)
        neighs[iv].append(iu)  # 无向

    counts = np.array([len(x) for x in neighs], dtype=np.int64)
    indptr = np.zeros(n+1, dtype=np.int64)
    indptr[1:] = np.cumsum(counts)
    indices = np.empty(indptr[-1], dtype=np.int64)
    p = 0
    for i, lst in enumerate(neighs):
        lst = sorted(lst)
        L = len(lst)
        indices[p:p+L] = lst
        p += L

    # 边列表 (去重 u<v)
    edges_u, edges_v, edges_etype = [], [], []
    for u, v, d in G.edges(data=True):
        iu, iv = node2idx[u], node2idx[v]
        if iu == iv:  # 跳过自环 (如果你算法里要保留，自行调整)
            continue
        if iu > iv:   # 只保留 u < v
            iu, iv = iv, iu
        et = d.get('type', None)
        etid = 0 if etype_to_id is None else etype_to_id.get(et, 0)
        edges_u.append(iu); edges_v.append(iv); edges_etype.append(etid)
    if len(edges_u) > 0:
        edges_u = np.array(edges_u, dtype=np.int64)
        edges_v = np.array(edges_v, dtype=np.int64)
        edges_etype = np.array(edges_etype, dtype=np.int64)
    else:
        edges_u = np.zeros(0, dtype=np.int64)
        edges_v = np.zeros(0, dtype=np.int64)
        edges_etype = np.zeros(0, dtype=np.int64)

    # 初始标签（若无就用度）
    init_labels = np.zeros(n, dtype=np.int64)
    if all('label' in G.nodes[n] for n in G.nodes()):
        for u in G.nodes():
            init_labels[node2idx[u]] = int(G.nodes[u]['label'])
    else:
        init_labels = counts.copy()  # 用度作为初始标签

    return indptr, indices, edges_u, edges_v, edges_etype, init_labels
