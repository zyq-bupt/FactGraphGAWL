import argparse
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
from data_loader_saver import load_data, data_save
from math import sqrt
# from edge_weights import edge_minor_type_weights, edge_major_type_weights
import os, json, shutil
from pathlib import Path

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="GAWL Weight Configuration")

    # 定义参数（名称要和命令行一致）
    parser.add_argument("--wTT", type=float, default=1.0, help="weight for token-token")
    parser.add_argument("--wTM", type=float, default=1.0, help="weight for token-mention")
    parser.add_argument("--wME", type=float, default=0.5, help="weight for mention-entity")
    parser.add_argument("--wEE", type=float, default=0, help="weight for entity-entity")
    parser.add_argument('--dataset', default='sample', help='Dataset name')#IMDB-BINARY  
    parser.add_argument('--T', type=int, default=1, help='Iterations of WL algorithm')
    # 如果还要接别的参数，也可以继续 add_argument
    # parser.add_argument("--dataset", type=str, default="sample", help="Dataset name")

    return parser.parse_args()

args = parse_args()
edge_major_type_weights = {
        "token-token": args.wTT,
        "token-mention": args.wTM,
        "mention-entity": args.wME,
        "entity-entity": args.wEE,
    }



def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def load_graph_data(ds_name, use_node_labels, use_emb_labels, use_edge_labels):
    node2graph = {}
    Gs = []
    edge_type_weights = []
    with open(ds_name + "graph_indicator.txt", "r") as f:
        # print("datasets/%s/%s_graph_indicator.txt"%(ds_name,ds_name))
        c = 1
        for line in f:
            # print(len(f)
            node2graph[c] = int(line.strip())
        
            if not node2graph[c] == len(Gs):
                Gs.append(nx.Graph())#先为当前图创建nx.Graph()，然后把节点c一个一个加入到图中/
            Gs[-1].add_node(c)
            c += 1

    
    if use_node_labels:
        with open(ds_name + "node_name_labels.txt", "r") as f:
            c = 1
            for line in f:
                node_label = int(line.strip())
                Gs[node2graph[c]-1].nodes[c]['label'] = node_label
                c += 1
    if use_emb_labels:
        with open(ds_name + "node_emb_labels.json", "r") as f:
            emb_list = json.load(f)

        c = 1
        for emb_label in emb_list:
            Gs[node2graph[c]-1].nodes[c]['emb_label'] = np.array(emb_label)
            c += 1



    if use_edge_labels == 1:
        edge_types = []
        with open(ds_name + "edge_labels.txt", "r") as f:
            edge_types = [line.strip() for line in f]

        idx = 0
        with open(ds_name + "A.txt", "r") as f:
            for line in f:
                u_str, v_str = line.strip().split(",")
                u, v = int(u_str), int(v_str.strip())
                G = Gs[node2graph[u] - 1]
                G.add_edge(u, v)
                # 存边类型属性
                G.edges[u, v]['type'] = edge_types[idx]
                idx += 1
        edge_type_weights = edge_minor_type_weights
    elif use_edge_labels == 2:
        edge_types = []
        with open(ds_name + "edge_label_type.txt", "r") as f:
            edge_types = [line.strip() for line in f]

        idx = 0
        with open(ds_name + "A.txt", "r") as f:
            for line in f:
                u_str, v_str = line.strip().split(",")
                u, v = int(u_str), int(v_str.strip())
                G = Gs[node2graph[u] - 1]
                G.add_edge(u, v)
                # 存边类型属性
                G.edges[u, v]['type'] = edge_types[idx]
                idx += 1

        edge_type_weights =  edge_major_type_weights
    else:
        with open(ds_name + "A.txt", "r") as f:
            for line in f:
                u_str, v_str = line.strip().split(",")
                u, v = int(u_str), int(v_str.strip())
                G = Gs[node2graph[u] - 1]
                G.add_edge(u, v)
              
                
    labels = []
    with open(ds_name + "graph_labels.txt", "r") as f:
        for line in f:
            labels.append(int(line.strip()))

    labels  = np.array(labels, dtype=np.float32)
    return Gs, labels, edge_type_weights


def get_wl_labels(Gs, h, node_labels_exist):
    '''
    这段代码完成了 Weisfeiler–Lehman 算法的核心步骤：
    1.初始标签（度或已有标签）→ 编号
    2.迭代地将每个节点的标签与邻居标签“聚合” → 新标签编号
    3.统计每轮标签分布的最大频数（可用于归一化或图特征比较）

    结果可用于：
    1.图同构测试：对比不同图在每轮标签分布是否一致
    2.图嵌入/特征提取：将节点的多轮标签作为特征用于机器学习
    '''
    N = len(Gs)

    node_labels = list()#node_labels[i][j, t] 最终会存储第 i 个图中第 j 个节点在第 t 轮的标签编号。
    node_to_idx = list()#node_to_idx[i] 映射真实节点 ID → node_labels 中的行号。
    for i,G in enumerate(Gs):
        # 创建一个 zeros 数组，行数=节点数，列数=h+1
        node_labels.append(np.zeros((G.number_of_nodes(), h+1), dtype=np.int64))
        # 建立节点到行号的映射，方便后面快速查
        node_to_idx.append(dict())
        
        for j,node in enumerate(G.nodes()):
            node_to_idx[i][node] = j
            # 如果没有预设标签，就用 degree 作为初始标签
            if not node_labels_exist:
                G.nodes[node]['label'] = G.degree(node)


    max_label_counts = list()

    #label_to_idx 字典，将每一种原始标签（度数或已有标签）映射到连续的整数编号 0,1,2,…
    #遍历所有图中所有节点，给它们的 第 0 列 node_labels[i][j,0] 赋上这个整数编号。
    label_to_idx = dict()
    for i,G in enumerate(Gs):
        for j,node in enumerate(G.nodes()):
            if G.nodes[node]['label'] not in label_to_idx:
                label_to_idx[G.nodes[node]['label']] = len(label_to_idx)

            node_labels[i][j,0] = label_to_idx[G.nodes[node]['label']]

    label_counts = np.zeros(len(label_to_idx), dtype=np.int32)
    for i in range(len(node_labels)):
        unique, counts = np.unique(node_labels[i][:,0], return_counts=True)
        v = np.zeros(len(label_to_idx), dtype=np.int32)
        for j in range(unique.size):
            v[unique[j]] = counts[j]
        label_counts = np.maximum(label_counts, v)

    max_label_counts.append(label_counts)

    for it in range(1,h+1):
        label_to_idx = dict()
        for i,G in enumerate(Gs):
            for j,node in enumerate(G.nodes()):
                new_label = list()
                for neighbor in G.neighbors(node):#取上轮标签 node_labels[i][j, it-1]，以及它所有邻居在上轮的标签列表。
                    new_label.append(node_labels[i][node_to_idx[i][neighbor],it-1])
                new_label = sorted(new_label)
                new_label.insert(0, node_labels[i][j,it-1])#排序后把自身标签插到最前面，得到一个不变序的 tuple，比如 (3,1,1,2,5)。
                new_label = tuple(new_label)## 在邻居标签列表前插入自己上轮的标签，组成新的“多重集”标签; 这个 tuple 就是 WL 的“邻居多重集”＋自身标签。
                if new_label not in label_to_idx:## 将这个 tuple 映射到新的整数编号
                    label_to_idx[new_label] = len(label_to_idx)
                else: 
                    pass    
                node_labels[i][j,it] = label_to_idx[new_label]

        label_counts = np.zeros(len(label_to_idx), dtype=np.int32)
        for i in range(len(node_labels)):
            unique, counts = np.unique(node_labels[i][:,it], return_counts=True)
            v = np.zeros(len(label_to_idx), dtype=np.int32)
            for j in range(unique.size):
                v[unique[j]] = counts[j]
            label_counts = np.maximum(label_counts, v)

        max_label_counts.append(label_counts)

    return node_labels#node_labels：每个图、每个节点、每一轮的标签编号矩阵，用于后续比较或特征提取。max_label_counts：一个列表，长度为 h+1，其中每个元素是一个向量，记录对应轮数下所有图中出现最频繁的标签次数上界。


def compute_gawl_kernel(Gs, h, node_labels):
    '''
    多轮（0…h） 地根据节点在第 t 轮的标签，统计每张图中各标签节点的频次，以及不同标签节点之间的边出现的频次；
    对于每一轮，将两张图在相同“标签对”上的边分布做加权匹配累加，得到它们的相似度贡献；

    最终对所有轮次的贡献求和，得到对称的相似度矩阵 K。
    '''
    K = np.zeros((len(Gs), len(Gs)))
    for it in range(h+1):
        edge_to_idx = dict()
        edges = list()
        node_label_freq = list()
        
        for i,G in enumerate(Gs):
            #统计标签频次 & 边的标签对分布
            node_to_idx = dict()
            # —— 2.1 构建节点编号映射 —— 
            for j,node in enumerate(G.nodes()):
                node_to_idx[node] = j
            # —— 2.2 统计本轮 it 下，每个标签出现的节点数 —— 
            unique, counts = np.unique(node_labels[i][:,it], return_counts=True)
            node_label_freq.append(dict())
            for j in range(unique.size):
                node_label_freq[i][unique[j]] = counts[j]#图i种，每个标签值出现的次数。

            # —— 2.3 对每条边，根据端点的标签编号 (v1, v2) 累加计数 —— 
            edges.append(dict())
            for edge in G.edges():
                # 取两端节点在轮次 it 时的标签
                v1 = node_labels[i][node_to_idx[edge[0]],it]
                v2 = node_labels[i][node_to_idx[edge[1]],it]

                if (v1, v2) not in edge_to_idx:
                    # 建立全局标签对 → 索引（可选）
                    edge_to_idx[(v1, v2)] = len(edge_to_idx)
                if (v1, v2) in edges[i]:
                    edges[i][(v1, v2)] += 1
                else:
                    edges[i][(v1, v2)] = 1
                
        #计算两图间的相似度贡献
        for i in range(len(Gs)):
            for j in range(i,len(Gs)):
                for edge in edges[i]:
                    #如果图 i 和图 j 中都有边 edge
                    if edge in edges[j]:
                        #如果是自环 edge头节点标签 = edge尾节点标签，并且在图 i 中头节点的标签频率大于1，在图 j 中，尾节点标签频率大于1
                        if edge[0] == edge[1] and node_label_freq[i][edge[0]] > 1 and node_label_freq[j][edge[0]] > 1:
                            ei = 2*edges[i][edge]/(node_label_freq[i][edge[0]]*(node_label_freq[i][edge[1]]-1))
                            ej = 2*edges[j][edge]/(node_label_freq[j][edge[0]]*(node_label_freq[j][edge[1]]-1))
                        else:
                            #1 归一化：计算每张图中此标签对的「边密度」ei, ej
                            ei = 2 * edges[i][edge]/(node_label_freq[i][edge[0]]*node_label_freq[i][edge[1]])
                            ej = 2 * edges[j][edge]/(node_label_freq[j][edge[0]]*node_label_freq[j][edge[1]])
                      
                        #2 权重：取两图中该标签对应节点数的最小值
                        # m1 = min(node_label_freq[i][edge[0]], node_label_freq[j][edge[0]])
                        # m2 = min(node_label_freq[i][edge[1]], node_label_freq[j][edge[1]])
                        # 改进3
                        m1 = node_label_freq[j][edge[0]]
                        m2 = node_label_freq[j][edge[1]]
                        # 3 累加相似度增量： 
                        K[i,j] += sqrt(ei)*sqrt(ej)*m1*m2
                K[j,i] = K[i,j]

    return K

def cos_sim(emb_uv_i, emb_uv_j):
    total_sim_u = 0.0
    total_sim_v = 0.0
    total_count = 0

    for (u_emb_i, v_emb_i) in emb_uv_i:
        for (u_emb_j, v_emb_j) in emb_uv_j:
            u_sim = cosine_similarity(u_emb_i.reshape(1, -1), u_emb_j.reshape(1, -1))[0][0]
            v_sim = cosine_similarity(v_emb_i.reshape(1, -1), v_emb_j.reshape(1, -1))[0][0]

            # 避免负相似度
            u_sim = max(u_sim, 0.0)
            v_sim = max(v_sim, 0.0)

            total_sim_u += u_sim
            total_sim_v += v_sim
            total_count += 1

    # 防止 total_count=0，保证返回 1.0 不影响整体
    if total_count == 0:
        return 1.0, 1.0
    else:
        avg_sim_u = total_sim_u / total_count
        avg_sim_v = total_sim_v / total_count
        return avg_sim_u, avg_sim_v

# pip install torch  (确保CUDA版)
import torch
import torch.nn.functional as F

def gpu_cos_sim(emb_uv_i, emb_uv_j, device="cuda"):
    """
    emb_uv_i / emb_uv_j: List[Tuple[np.ndarray, np.ndarray]]
    返回: (avg_sim_u, avg_sim_v)
    """
    if len(emb_uv_i) == 0 or len(emb_uv_j) == 0:
        return 1.0, 1.0

    # 组装为矩阵: [Ni, D], [Nj, D]
    Ui = torch.from_numpy(
        np.stack([u for (u, v) in emb_uv_i], axis=0)
    ).to(device=device, dtype=torch.float32)
    Vi = torch.from_numpy(
        np.stack([v for (u, v) in emb_uv_i], axis=0)
    ).to(device=device, dtype=torch.float32)

    Uj = torch.from_numpy(
        np.stack([u for (u, v) in emb_uv_j], axis=0)
    ).to(device=device, dtype=torch.float32)
    Vj = torch.from_numpy(
        np.stack([v for (u, v) in emb_uv_j], axis=0)
    ).to(device=device, dtype=torch.float32)

    # 归一化
    Ui = F.normalize(Ui, dim=1)
    Vi = F.normalize(Vi, dim=1)
    Uj = F.normalize(Uj, dim=1)
    Vj = F.normalize(Vj, dim=1)

    # 两两相似度 = 矩阵乘： [Ni, D] @ [D, Nj] -> [Ni, Nj]
    # 取平均即可（等价于你原先的双重循环取平均）
    sim_u = torch.clamp(Ui @ Uj.T, min=0).mean()  # 截断负值为0
    sim_v = torch.clamp(Vi @ Vj.T, min=0).mean()

    return float(sim_u), float(sim_v)


def compute_gawl_kernel_v2(Gs, h, node_labels, edge_type_weights, use_emb_labels):
    '''
    GAWL kernel 版本 2 ：考虑了边
    - 方案 1 ：每种 edge 小类type 乘以 edge_type_weights（默认为1）
    - 方案 2 ：按 edge 小类type 做 per-type 归一化
    '''
    K = np.zeros((len(Gs), len(Gs)))

    for it in range(h+1):
        edge_to_idx = dict()
        edges = list()
        edges_node_emb = list()
        node_label_freq = list()

        # 【新增】统计 per 图 per edge type count
        edge_type_count = [dict() for _ in range(len(Gs))]

        for i, G in enumerate(Gs):
            node_to_idx = dict()
            for j, node in enumerate(G.nodes()):
                node_to_idx[node] = j

            unique, counts = np.unique(node_labels[i][:, it], return_counts=True)
            node_label_freq.append(dict())
            for j in range(unique.size):
                node_label_freq[i][unique[j]] = counts[j]

            edges.append(dict())
            edges_node_emb.append(dict())
            for edge in G.edges():
                v1 = node_labels[i][node_to_idx[edge[0]], it]
                v2 = node_labels[i][node_to_idx[edge[1]], it]
                
                
                etype = G.edges[edge].get('type', None)
                w = edge_type_weights.get(etype, 1.0)

                # 【新增】统计 edge type count
                if etype in edge_type_count[i]:
                    edge_type_count[i][etype] += 1
                else:
                    edge_type_count[i][etype] = 1

                # 【改】 edges key 改成 (v1,v2,etype)
                key = (v1, v2, etype)

                if key not in edge_to_idx:
                    edge_to_idx[key] = len(edge_to_idx)
                if key in edges[i]:
                    edges[i][key] += w
                else:
                    edges[i][key] = w
                    edges_node_emb[i][key] = list()

                #为了计算边相似性，需要把相应的节点emb存进去。
                if use_emb_labels:
                    emb_v1 = G.nodes[edge[0]]['emb_label']
                    emb_v2 = G.nodes[edge[1]]['emb_label']

                    edges_node_emb[i][key].append((emb_v1,emb_v2))    
        
                

        # === 相似度计算 ===
        edge_contributions_j = dict()
        for i in range(len(Gs)):
            for j in range(i+1, len(Gs)):
                for edge in edges[j]:
                    if edge in edges[i]:
                        etype = edge[2]
                        count_i = edge_type_count[i].get(etype, 1)
                        count_j = edge_type_count[j].get(etype, 1)

                        # 自环特殊处理
                        if edge[0] == edge[1] and node_label_freq[i][edge[0]] > 1 and node_label_freq[j][edge[0]] > 1:
                            ei = 2 * edges[i][edge] / (node_label_freq[i][edge[0]] * (node_label_freq[i][edge[1]] - 1) * count_i)
                            ej = 2 * edges[j][edge] / (node_label_freq[j][edge[0]] * (node_label_freq[j][edge[1]] - 1) * count_j)
                        else:
                            ei = 2 * edges[i][edge] / (node_label_freq[i][edge[0]] * node_label_freq[i][edge[1]] * count_i)
                            ej = 2 * edges[j][edge] / (node_label_freq[j][edge[0]] * node_label_freq[j][edge[1]] * count_j)

                        # 【这里我默认用你的小图主导写法】
                        m1 = node_label_freq[j][edge[0]]
                        m2 = node_label_freq[j][edge[1]]
                        
                        if use_emb_labels: 
                            emb_uv_i = edges_node_emb[i][edge]
                            emb_uv_j = edges_node_emb[j][edge]
                            
                            sim_u, sim_v = gpu_cos_sim(emb_uv_i, emb_uv_j) 
                            kij = sqrt(ei) * sqrt(ej) * m1 * m2 * sim_u * sim_v
                            # if etype == 'mention-entity':
                            #     if kij > 0:
                            #         print(etype,edge)
                                
                            # if etype == etype == 'entity-entity':
                            #     if kij > 0:
                            #         print(edge)
                                
                        else:
                            kij = sqrt(ei) * sqrt(ej) * m1 * m2
                        edge_contributions_j[edge] = kij
                        K[i, j] += kij
                    else:
                        edge_contributions_j[edge] = -1
                K[j, i] = K[i, j]

                # —— 你原本的 K 计算完毕之后 —— #

                # 1. 先取出对角线，得到每个样本的自相似度
                diag = np.sqrt(np.diag(K))            # shape (n,)

                # 2. 构造归一化分母矩阵：diag[i] * diag[j]
                denom = np.outer(diag, diag)          # shape (n, n)

                # 3. 防止除零（如果某些 diag 为零，可以在它们对应位置上加个微小常数）
                denom[denom == 0] = 1e-12

                # 4. 最终归一化
                K_norm = K / denom
    
    # return K
    return K_norm


import os
import json
from pathlib import Path
from dfc2 import convert_json_graph
from tqdm import tqdm

def is_graph_file_empty(input_path: Path, graph_keys: list[str]) -> bool:
    with input_path.open('r', encoding='utf-8') as fin:
        data = json.load(fin)
    for key in graph_keys:
        if key in data and 'graph_without_emb' in data[key]:
            G = data[key]['graph_without_emb']
            nodes = G.get('nodes', [])
            links = G.get('links', [])
            if len(nodes) == 0 and len(links) == 0:
                return True

    return False
if __name__ == '__main__':
    # 
    
    use_node_labels = True
    use_emb_labels = True
    use_edge_labels = 2  #0:不使用边权重；1:edge_minor_type_weights, 2:edge_major_type_weights
    dataset = 'DeFacto'#  UniSumEval

    if dataset== 'DeFacto':
        keys = ["article", "candidate", "humman_summary"]
        data_path = 'defacto_data_gawl'
        flist=['test','val','train']
    else:
        keys = ["article", "candidate"] 
        data_path = 'unisumeval_data_gawl'
        flist=['test']
    
    if use_emb_labels:
        outd = 'factgraph_result_withemb'
    else:
        outd = 'factgraph_result'
    prefix = ""
    undirected = False
    outdir = "/root/autodl-fs/zyq/%s/%s/"%(data_path,outd)

    right_in = 0
    equ_in = 0
    allcount_in = 0
    right_ex = 0
    equ_ex = 0
    allcount_ex = 0
    
    kongc = 0
    # cc = 0
    for f in flist:
        # 读文件，然后转化
        # input_path = "/root/autodl-fs/zyq/%s/hgmaedata/factgraph_result_HGMAE_3_1_4/%s/"%(data_path,f)
        # input_path = "/root/autodl-fs/zyq/data_gawl/factgraph_result_withemb/"
        input_path = os.path.join(outdir,f)
        
        tmp_root = Path(outdir)  # 原 outdir 只是作为根目录
        tmp_root.mkdir(parents=True, exist_ok=True)
        for filename in tqdm(os.listdir(input_path)):
            if filename.endswith('.json'):
                # print(filename)
                # if filename != '1244.json':
                #     continue
                input_file = os.path.join(input_path, filename)

                if is_graph_file_empty(Path(input_file), keys):
                    kongc+=1
                    continue


                # cc+=1
                # if cc>10:break
                    
                # 1) 为当前样本建立唯一化工作区：样本名 + 进程ID
                stem = Path(filename).stem
                work_dir = tmp_root / f"{stem}__pid{os.getpid()}"
                if work_dir.exists():
                    shutil.rmtree(work_dir)
                work_dir.mkdir(parents=True, exist_ok=True)
                
                convert_json_graph(
                    input_path=Path(input_file),
                    graph_keys=keys,
                    output_prefix=prefix,
                    undirected=undirected,
                    use_emb_labels=use_emb_labels,
                    output_dir=work_dir
                )
           
                # print(f"代码11执行时间：{end_time - start_time:.6f} 秒")
                
                # start_time = time.time()  # 记录开始时间
                #调用gawl
                Gs, y, edge_type_weights= load_graph_data(str(work_dir) + '/', use_node_labels, use_emb_labels, use_edge_labels)
                
                # 如果是defacto，那么是要比较正确摘要和错误摘要的事实性评估得分
                if dataset== 'DeFacto':
                    K_all = []
                    for gidx in range(1,3):
                        compare_Gs= [Gs[0],Gs[gidx]]
                        node_labels = get_wl_labels(compare_Gs, args.T, use_node_labels)

                        if use_edge_labels == 0:  #
                            K = compute_gawl_kernel(compare_Gs, args.T, node_labels)
                        else:
                            K = compute_gawl_kernel_v2(compare_Gs, args.T, node_labels, edge_type_weights, use_emb_labels)
                        K_all.append(K[0,1])

                    with open(input_file, 'r', encoding='utf-8') as file:
                        try:
                            input_data = json.load(file)
                        except json.JSONDecodeError as e:
                            print(f"读取文件 {input_file} 时出错: {e}")

                    if input_data['intrinsic_error'] == True and input_data['extrinsic_error'] == False:
                        if K_all[1] > K_all[0]:
                            right_in += 1
                            # with open('log_in_lager.txt', 'a', encoding='utf-8') as f:
                            #     f.write('in：'+filename+'\n')
                            
                        if K_all[1] == K_all[0]:
                            equ_in += 1
                            # with open('log_in_equl.txt', 'a', encoding='utf-8') as f:
                            #     f.write('in：'+filename+'\n')
                            
                        allcount_in +=1
                    elif input_data['intrinsic_error'] == False and input_data['extrinsic_error'] == True:
                        if K_all[1] > K_all[0]:
                            right_ex += 1
                            # with open('log_ex_lager.txt', 'a', encoding='utf-8') as f:
                            #     f.write('ex：'+filename+'\n')
                        if K_all[1] == K_all[0]:
                            equ_ex += 1
                            # with open('log_ex_equl.txt', 'a', encoding='utf-8') as f:
                            #     f.write('ex：'+filename+'\n')
                        allcount_ex +=1
                    
                else:
                    compare_Gs= Gs
                    node_labels = get_wl_labels(compare_Gs, args.T, use_node_labels)

                    if use_edge_labels == 0:  #
                        K = compute_gawl_kernel(compare_Gs, args.T, node_labels)
                    else:
                        K = compute_gawl_kernel_v2(compare_Gs, args.T, node_labels, edge_type_weights, use_emb_labels)
                    # print(K[0,1])

                    orig_data = load_data(os.path.join('/root/autodl-fs/zyq/unisumeval_data_gawl/dreeam_result/test/', filename))

                    new_data = orig_data.copy()

                    new_data['graph_sim']={'WL-GAWL':K[0,1]}

                    saveSim_path = "/root/autodl-fs/zyq/unisumeval_data_gawl/factgraphGAWL_WL_PT/%s/"%(f)
                    # saveSim_file = os.path.join(saveSim_path, filename)
                    data_save(new_data, saveSim_path)
                # 计算完可以选择清理
                shutil.rmtree(work_dir, ignore_errors=True)    

    # print(dataset)
    print(right_in,equ_in)
    print(right_ex,equ_ex)
    print(right_in+right_ex)
    params=[edge_major_type_weights["token-token"],edge_major_type_weights["token-mention"],edge_major_type_weights["mention-entity"],edge_major_type_weights["entity-entity"]]
    with open('PTresult_3.txt', 'a', encoding='utf-8') as f:
        f.write(f"GAWL_PT_{params[0]}_{params[1]}_{params[2]}_{params[3]}"+ '\n' +'in: ' + str(right_in) + ', ' + str(equ_in) + '\n' + 'in: ' + str(right_ex) + ', ' + str(equ_ex) + '\n'+str(right_in+right_ex)+ '\n')   





