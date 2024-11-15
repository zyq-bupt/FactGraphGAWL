import networkx as nx
import json
from py2neo import Graph, Node, Relationship
from graph_embedding.graph_builder import GraphBuilder



# 3. 将 NetworkX 图导入 Neo4j
def import_networkx_to_neo4j(G, graph):
    # 导入节点
    for node_id, node_data in G.nodes(data=True):
        # 创建带有标签的节点
        label = node_data.get("label", "Node")
        node_properties = {k: v for k, v in node_data.items() if k != "label"}  # 不包括标签
        neo4j_node = Node(label, **node_properties)
        graph.merge(neo4j_node, label,'node_id')  # 使用 `name` 属性作为唯一标识符，避免重复创建

    # 导入关系
    for source, target, edge_data in G.edges(data=True):
        relationship_type = edge_data.get("relationship", "RELATED_TO")
        
        # 查找源节点和目标节点
        source_node = graph.nodes.match(node_id=G.nodes[source]["node_id"]).first()
        target_node = graph.nodes.match(node_id=G.nodes[target]["node_id"]).first()
        
        # 创建关系
        if source_node and target_node:
            rel = Relationship(source_node, relationship_type, target_node)
            graph.merge(rel)


if __name__ == '__main__':
    #################
    #因为免费版neo4j只能有一个数据库，所以一次只能展示一个摘要图，所以分开两次在数据库中展示。
    #################

    # 连接到 Neo4j 数据库
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "zhuyingqi"))
    graph.run("MATCH (n) DETACH DELETE n")
    with open('/root/autodl-fs/zyq/DeFacto/data/factgraph_tt_result/test/2611.json', 'r') as f:
        input_data = json.load(f)
    print(input_data['doc_id'])

    graph_builder= GraphBuilder()
    G1=graph_builder.build_graph(input_data["candidate"])
    import_networkx_to_neo4j(G1, graph)

    ####第二次把这个G2解除屏蔽，把上面G1的屏蔽

    # graph_builder= GraphBuilder()
    # G2=graph_builder.build_graph(input_data["humman_summary"])
    # import_networkx_to_neo4j(G2, graph)
    
    print("NetworkX 图已成功导入到 Neo4j 数据库中。")