import networkx as nx
import json
from py2neo import Graph, Node, Relationship
from graph_embedding.graph_builder import GraphBuilder


graph_builder= GraphBuilder()

text_list=['article','abstract','candidate','humman_summary']

with open('/root/autodl-fs/zyq/DeFacto/data/factgraph_result/test/139.json', 'r') as f:
    input_data = json.load(f)

new_data = input_data.copy()
print(input_data['doc_id'])
G1=graph_builder.build_graph(input_data["candidate"])
graph_builder= GraphBuilder()
G2=graph_builder.build_graph(input_data["humman_summary"])
    

                    


# 2. 连接到 Neo4j 数据库
graph = Graph("bolt://localhost:7687", auth=("neo4j", "zhuyingqi"))
graph.run("MATCH (n) DETACH DELETE n")
# 3. 将 NetworkX 图导入 Neo4j
def import_networkx_to_neo4j(G, graph):
    # 导入节点
    for node_id, node_data in G.nodes(data=True):
        # 创建带有标签的节点
        label = node_data.get("label", "Node")
        node_properties = {k: v for k, v in node_data.items() if k != "label"}  # 不包括标签
        neo4j_node = Node(label, **node_properties)
        graph.merge(neo4j_node, label, "name")  # 使用 `name` 属性作为唯一标识符，避免重复创建

    # 导入关系
    for source, target, edge_data in G.edges(data=True):
        relationship_type = edge_data.get("relationship", "RELATED_TO")
        
        # 查找源节点和目标节点
        source_node = graph.nodes.match(name=G.nodes[source]["name"]).first()
        target_node = graph.nodes.match(name=G.nodes[target]["name"]).first()
        
        # 创建关系
        if source_node and target_node:
            rel = Relationship(source_node, relationship_type, target_node)
            graph.merge(rel)

# 5. 执行导入
import_networkx_to_neo4j(G1, graph)
# import_networkx_to_neo4j(G2, graph)
print("NetworkX 图已成功导入到 Neo4j 数据库中。")

