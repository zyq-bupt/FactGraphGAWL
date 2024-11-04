import networkx as nx
import sys
sys.path.append("/root/autodl-tmp/shortest-path-graph/graph")
from graph.words_of_graph import WordsOfGraph
from graph.shortest_path_graph import ShortestPathGraph
from k_score.k_counter import KCounter

import json
import networkx as nx
import numpy as np

def load_graph_with_embeddings(input_file_path):
    # 从 JSON 文件读取数据
    with open(input_file_path, 'r') as f:
        data = json.load(f)
    
    # 将 JSON 数据转换为 NetworkX 图
    G = nx.node_link_graph(data)
    
    # 提取节点嵌入并将其恢复为 numpy 数组格式
    embeddings = {}
    for node_id, node_data in G.nodes(data=True):
        if 'embedding' in node_data:
            embeddings[node_id] = np.array(node_data['embedding'])  # 将列表转换回 numpy 数组
    
    return G, embeddings



# 现在 G 是一个包含嵌入信息的 NetworkX 图，embeddings 是一个包含节点嵌入的字典



if __name__ == '__main__':
    s1 = 'e1 e2 period'
    s2 = 'e1 e2'
    s1 = "e1 was in e2 dated summer of 1938 along other dresses from this same period."
    s2 = "e1 was in e2 dated summer of 1938 along with several other dresses from this same period."
    # G1 = WordsOfGraph(s1, 2).get_graph()
    # G2 = WordsOfGraph(s2, 2).get_graph()
    # 使用示例
    input_file_path1 = '/root/autodl-tmp/graph_embedding/xsum/graph_with_embeddings1.json'
    G1, embeddings1 = load_graph_with_embeddings(input_file_path1)
    input_file_path2 = '/root/autodl-tmp/graph_embedding/xsum/graph_with_embeddings2.json'
    G2, embeddings2 = load_graph_with_embeddings(input_file_path2)

    C1 = ShortestPathGraph(G1, 50).get_graph()
    C2 = ShortestPathGraph(G2, 50).get_graph()
    score = KCounter(C1, C2).get_k_score()
    print(score)
