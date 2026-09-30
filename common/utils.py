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

from numba import njit
from numba.typed import Dict
from numba import types

@njit(cache=True)
def poly_hash(vals):
    """
    简单 64-bit 滚动哈希，vals 为 int64 数组
    """
    h = np.uint64(1469598103934665603)  # FNV64 offset
    prime = np.uint64(1099511628211)
    for x in vals:
        h ^= np.uint64(np.int64(x))
        h *= prime
    return h

@njit(cache=True)
def wl_relabel_csr(indptr, indices, init_labels, h):
    """
    indptr, indices: CSR 邻接
    init_labels: int64, shape (n,)
    返回:
      labels: int64, shape (n, h+1)  每列一轮 (0..h)
    """
    n = init_labels.shape[0]
    labels = np.zeros((n, h+1), dtype=np.int64)
    labels[:, 0] = init_labels

    tmp = np.empty(0, dtype=np.int64)  # 复用缓冲

    for it in range(1, h+1):
        # 映射: hash64 -> new_id
        mp = Dict.empty(key_type=types.uint64, value_type=types.int64)
        next_id = 0

        for u in range(n):
            # 收集邻居上一轮标签
            start, end = indptr[u], indptr[u+1]
            deg = end - start
            if tmp.shape[0] < deg + 1:
                tmp = np.empty(deg + 1, dtype=np.int64)
            # 自身标签放在首位
            tmp[0] = labels[u, it-1]
            k = 1
            for p in range(start, end):
                v = indices[p]
                tmp[k] = labels[v, it-1]
                k += 1
            # 排序邻居部分
            if deg > 0:
                # 只排序 tmp[1:k]
                for i in range(1, k):
                    x = tmp[i]
                    j = i - 1
                    while j >= 1 and tmp[j] > x:
                        tmp[j+1] = tmp[j]
                        j -= 1
                    tmp[j+1] = x

            hv = poly_hash(tmp[:k].astype(np.int64))
            if hv in mp:
                labels[u, it] = mp[hv]
            else:
                mp[hv] = next_id
                labels[u, it] = next_id
                next_id += 1

    return labels
from numba.typed import Dict
from numba import types

@njit(cache=True)
def pack_key(label_u, label_v, etype):
    # 假设 label 值 < 2^20, etype < 2^20
    return (np.int64(etype) << 40) | (np.int64(label_u) << 20) | np.int64(label_v)

@njit(cache=True)
def graph_stats_per_iter(labels, edges_u, edges_v, edges_etype, it, edge_weight_per_type):
    """
    统计单个图在第 it 轮的：
      - label_freq: Dict[label -> count]
      - edge_count: Dict[key -> weighted_count], key=(label_u,label_v,etype)有向无向按需处理
    edge_weight_per_type: int64->float64 的长度 = n_etype 的数组
    """
    label_freq = Dict.empty(key_type=types.int64, value_type=types.int64)
    edge_count = Dict.empty(key_type=types.int64, value_type=types.float64)

    n = labels.shape[0]

    # 节点标签频次
    for u in range(n):
        lu = labels[u, it]
        if lu in label_freq:
            label_freq[lu] += 1
        else:
            label_freq[lu] = 1

    # 边计数（无向：u<v 已保证；若需要有向，去掉该假设）
    m = edges_u.shape[0]
    for e in range(m):
        u = edges_u[e]; v = edges_v[e]; et = edges_etype[e]
        w = edge_weight_per_type[et]  # 权重
        lu = labels[u, it]; lv = labels[v, it]
        # 你原实现里区分自环与否并归一化，这里先只计数量，归一化放到核里做
        key = pack_key(lu, lv, et)
        if key in edge_count:
            edge_count[key] += w
        else:
            edge_count[key] = w

    return label_freq, edge_count
@njit(cache=True)
def dict_get_int(dct, key):
    return dct[key] if key in dct else 0

@njit(cache=True)
def dict_get_float(dct, key):
    return dct[key] if key in dct else 0.0

@njit(cache=True)
def kernel_increment_one_iter(
    label_freq_i, edge_count_i,
    label_freq_j, edge_count_j,
    per_type_count_i, per_type_count_j,
    use_selfloop_rule=True, use_small_graph_dominate=True
):
    """
    返回第 it 轮的 K_ij 增量
    per_type_count_*: Dict[etype->int]  (如果要做“每种类型边的个数归一化”，传这个；否则可以全部设 1)
    """
    kij = 0.0

    # 遍历“交集”的键：我们以 j 的边键为主
    for key_j in edge_count_j:
        val_j = edge_count_j[key_j]
        if key_j in edge_count_i:
            val_i = edge_count_i[key_j]

            # 解包 key_j -> (lu, lv, etype)
            et = (key_j >> 40) & ((1 << 20) - 1)
            lu = (key_j >> 20) & ((1 << 20) - 1)
            lv = key_j & ((1 << 20) - 1)

            # 归一化因子
            # per-type 计数
            ct_i = dict_get_int(per_type_count_i, et)
            ct_j = dict_get_int(per_type_count_j, et)
            if ct_i <= 0: ct_i = 1
            if ct_j <= 0: ct_j = 1

            # 节点标签频次
            fi_lu = dict_get_int(label_freq_i, lu)
            fi_lv = dict_get_int(label_freq_i, lv)
            fj_lu = dict_get_int(label_freq_j, lu)
            fj_lv = dict_get_int(label_freq_j, lv)
            if fi_lu <= 0 or fi_lv <= 0 or fj_lu <= 0 or fj_lv <= 0:
                continue

            # 自环与否的不同归一化
            if use_selfloop_rule and (lu == lv) and (fi_lu > 1) and (fj_lu > 1):
                ei = 2.0 * val_i / (fi_lu * (fi_lv - 1) * ct_i)
                ej = 2.0 * val_j / (fj_lu * (fj_lv - 1) * ct_j)
            else:
                ei = 2.0 * val_i / (fi_lu * fi_lv * ct_i)
                ej = 2.0 * val_j / (fj_lu * fj_lv * ct_j)

            # 小图主导/最小值
            if use_small_graph_dominate:
                m1 = fj_lu
                m2 = fj_lv
            else:
                m1 = fi_lu if fi_lu < fj_lu else fj_lu
                m2 = fi_lv if fi_lv < fj_lv else fj_lv

            # 贡献
            # 如果你还有 emb 相似度因子 sim_u*sim_v，这里可以乘上（那部分建议单独 GPU 化）
            kij += (np.sqrt(ei) * np.sqrt(ej) * m1 * m2)

    return kij
