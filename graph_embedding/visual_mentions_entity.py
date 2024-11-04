from py2neo import Graph, Node, Relationship, NodeMatcher
import matplotlib.pyplot as plt
import networkx as nx
import json

# 连接到Neo4j数据库
graph = Graph("bolt://localhost:7687", auth=("neo4j", "zhuyingqi"))

# # 加载JSON文件
with open('/root/autodl-tmp/xsum/little_data/xsum_test_dreeam_small.json', 'r') as f:
    blink_data = json.load(f)

with open('/root/autodl-tmp/xsum/little_data/dreeam_results.json', 'r') as f:
    dreeam_data = json.load(f)

# 创建实体节点和mention节点
entity_nodes = []
mention_nodes = []
matcher = NodeMatcher(graph)
for i, entity in enumerate(blink_data[1]['vertexSet']):
    # 创建实体节点
    entity_node = Node("Entity", name=f"Entity_{i}")
    graph.create(entity_node)
    entity_nodes.append(entity_node)
    
    # 创建mention节点并与实体节点建立关系
    for mention in entity:
        mention_node = matcher.match("Mention", id=mention['mention_id']).first()
        if not mention_node:
            mention_node = Node("Mention", id=mention['mention_id'], name=mention['name'])
            graph.create(mention_node)
            mention_nodes.append(mention_node)
       
        
        # 创建mention属于实体的关系
        belongs_rel = Relationship(mention_node, "BELONGS_TO", entity_node)
        graph.create(belongs_rel)
related_entities = set()  # 用于跟踪已有关系的实体
# 创建实体与实体之间的关系
for relation in dreeam_data:
    if relation['title']=='1':
        head_entity_idx = relation['h_idx']
        tail_entity_idx = relation['t_idx']
        relation_type = relation['r']
        
        # 获取头实体和尾实体节点
        head_entity_node = entity_nodes[head_entity_idx]
        tail_entity_node = entity_nodes[tail_entity_idx]
        
        # 创建实体之间的关系
        entity_rel = Relationship(head_entity_node, relation_type, tail_entity_node)
        graph.create(entity_rel)

        # 添加到已有关系的实体集合
        related_entities.add(head_entity_idx)
        related_entities.add(tail_entity_idx)

print("数据已成功导入到Neo4j数据库中！")

# # 使用NetworkX从Neo4j提取图数据
# matcher = NodeMatcher(graph)
# G = nx.DiGraph()

# # 获取所有实体节点和mention节点
# entity_nodes = list(matcher.match("Entity"))
# mention_nodes = list(matcher.match("Mention"))

# # 添加实体节点到图中
# for node in entity_nodes:
#     G.add_node(node["name"], label="Entity")

# # 添加mention节点到图中
# for node in mention_nodes:
#     G.add_node(node["name"], label="Mention")

# # 获取mention到实体的关系
# belongs_rels = graph.match(r_type="BELONGS_TO")
# for rel in belongs_rels:
#     G.add_edge(rel.start_node["name"], rel.end_node["name"], label="BELONGS_TO")

# # 获取实体之间的关系
# entity_rels = graph.match()
# for rel in entity_rels:
#     if rel.__class__.__name__!= 'BELONGS_TO':  # 忽略BELONGS_TO关系，因其已被处理
#         G.add_edge(rel.start_node["name"], rel.end_node["name"], label=rel.__class__.__name__)
# # 画图
# plt.figure(figsize=(15, 10))
# pos = nx.spring_layout(G, k=1.5, iterations=50)  # 调整参数以减少线交叉

# # 绘制节点和边，使用不同颜色和形状来区分实体和mention
# entity_nodes_labels = {node: node for node, attr in G.nodes(data=True) if attr["label"] == "Entity"}
# mention_nodes_labels = {node: node for node, attr in G.nodes(data=True) if attr["label"] == "Mention"}

# nx.draw_networkx_nodes(G, pos, nodelist=entity_nodes_labels.keys(), node_color="lightblue", node_shape="o", label="Entity Nodes")
# nx.draw_networkx_nodes(G, pos, nodelist=mention_nodes_labels.keys(), node_color="orange", node_shape="s", label="Mention Nodes")
# nx.draw_networkx_edges(G, pos, edgelist=G.edges(data=True), arrows=True)
# nx.draw_networkx_labels(G, pos, labels={**entity_nodes_labels, **mention_nodes_labels}, font_size=8)

# # 绘制边的标签
# # edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
# # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

# plt.title("Graph Visualization of Entities and Mentions in Neo4j")
# plt.legend()
# plt.show()
# plt.savefig('/root/autodl-tmp/xsum/little_data/fig_ent_ent.jpg')