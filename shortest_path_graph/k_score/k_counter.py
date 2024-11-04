import networkx as nx
from networkx.classes.graph import Graph
import numpy as np

class KCounter:
    C1 = nx.Graph()
    C2 = nx.Graph()
    k_node_sum = int()
    k_walk_sum = int()
    norm = int()
    k_score = int()

    def __init__(self, C1:Graph, C2:Graph):
        self.C1 = C1
        self.C2 = C2
        self.k_node_sum = self.count_k_node_sum(self.C1, self.C2)
        self.k_walk_sum = self.count_k_walk_sum(self.C1, self.C2)
        self.norm = self.count_norm(self.C1, self.C2)
        self.k_score = self.count_k_score(self.k_node_sum, self.k_walk_sum, self.norm)

    def count_k_node_sum(self, C1:Graph, C2:Graph):
        k_node = 0
        for node1, data1 in C1.nodes(data=True):
            name1 = data1.get('name')
            # 遍历第二个数据集中的所有节点
            for node2, data2 in C2.nodes(data=True):
                name2 = data2.get('name')   
                # 如果 name 相同，则记录这个匹配
                if name1 ==  name2:
                    k_node += 1
        # V1 = C1.nodes
        # V2 = C2.nodes
        # k_node = 0
        # for v1 in V1:
        #     if v1 in V2:
        #         k_node += 1
        return k_node

    def count_k_walk_sum(self, C1:Graph, C2:Graph):
        k_walk = 0
        # 遍历 C1 中的所有边
        for u1, v1, data1 in C1.edges(data=True):
            # 获取节点 u1 和 v1 的 name 属性
            name_u1 = C1.nodes[u1].get('name')
            name_v1 = C1.nodes[v1].get('name')
            
            # 在 C2 中找到具有相同 name 的节点
            u2 = None
            v2 = None
            for node2, data2 in C2.nodes(data=True):
                if data2.get('name') == name_u1:
                    u2 = node2
                if data2.get('name') == name_v1:
                    v2 = node2
                if u2 and v2:
                    break

            # 检查在 C2 中是否存在 u2 和 v2 之间的边
            if u2 is not None and v2 is not None and C2.has_edge(u2, v2):
                # 获取边的权重
                l1 = data1.get('weight')  # 默认权重为 1
                l2 = C2[u2][v2].get('weight')  # 默认权重为 1

                k_walk += l1 * l2

        # E1 = C1.edges
        # E2 = C2.edges
        # common_edge = list()
        # k_walk = 0
        # common_edge=list(set(E1) & set(E2))
        # # for e in E1:
        # #     if e in E2:
        # #         common_edge.append(e)
        
        # for ce in common_edge:
        #     if ce[0]==ce[1]:
        #         continue
        #     l1 = E1[ce]['weight']
        #     l2 = E2[ce]['weight']
        #     k_walk += l1 * l2 * 2
        return k_walk

    def count_norm(self, C1:Graph, C2:Graph):
        A1 = nx.to_numpy_array(C1)
        A2 = nx.to_numpy_array(C2)
        D1 = np.eye(N=A1.shape[0], M=A1.shape[1])
        D2 = np.eye(N=A2.shape[0], M=A2.shape[1])
        M1 = A1 + D1
        M2 = A2 + D2
        norm = np.linalg.norm(M1) * np.linalg.norm(M2)
        return norm

    def count_k_score(self,k_node, k_walk, norm):
        k_score = (k_node + k_walk) / norm
        return k_score
    
    def get_k_score(self):
        return self.k_score