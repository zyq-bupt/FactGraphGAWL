import networkx as nx
from typing import Callable, Optional

class ShortestPathGraph:
    """
    将最短路径<=d的节点对补成边：
    - 为自环(i==i)设置固定权重 init_w
    - 为不同点(i!=j)按 weight_fn(dist) 赋权（默认 1/dist）
    - 新增的边统一打上 type=edge_type_name
    - 如已有边且不想覆盖，only_new_edges=True 时只对“原图中不存在的边”进行补充
    """
    def __init__(
        self,
        words_of_graph: nx.Graph,
        d: int,
        init_w: float = 1.0,
        weight_fn: Optional[Callable[[float], float]] = None,
        include_self_loops: bool = True,
        only_new_edges: bool = True,
        edge_type_name: str = "shortest-path",
    ):
        self.G = words_of_graph
        self.d = d
        self.init_w = init_w
        self.weight_fn = weight_fn if weight_fn is not None else (lambda dist: 1.0 / dist)
        self.include_self_loops = include_self_loops
        self.only_new_edges = only_new_edges
        self.edge_type_name = edge_type_name

    def construct_graph(self) -> nx.Graph:
        # Dijkstra 最短路，支持已有边权（若原图无权重默认为1）
        path_length = dict(nx.all_pairs_dijkstra_path_length(self.G, weight="weight"))

        # 逐对检查并补边
        for src, dist_map in path_length.items():
            for dst, dist in dist_map.items():
                if dist < 0:
                    continue
                if dist > self.d:
                    continue
                if src == dst and not self.include_self_loops:
                    continue

                # 仅补充“原本不存在”的边（避免覆盖已有 type）
                if self.only_new_edges and self.G.has_edge(src, dst):
                    continue

                if src == dst:
                    w = self.init_w
                else:
                    # 防止除0（虽然 dijkstra 不会给出0的非自环）
                    w = self.weight_fn(dist) if dist > 0 else self.init_w

                # 添加/更新边
                self.G.add_edge(src, dst)
                # 不覆盖已有类型：若已经有 type 就保留；否则标注 shortest-path
                if "type" not in self.G.edges[src, dst]:
                    self.G.edges[src, dst]["type"] = self.edge_type_name
                # 写入/叠加权重：将 sp 权重写入 edge 属性 'weight'（如已有 weight 可叠加）
                prev_w = self.G.edges[src, dst].get("weight", 0.0)
                self.G.edges[src, dst]["weight"] = prev_w + w

        return self.G

    def get_graph(self) -> nx.Graph:
        return self.construct_graph()
