import json
from py2neo import Graph, Node, Relationship
import networkx as nx
import matplotlib.pyplot as plt


# 连接到 Neo4j 数据库
graph = Graph("bolt://localhost:7687", auth=("neo4j", "zhuyingqi"))

with open('/root/autodl-tmp/xsum/little_data/token_dep_mentions.json', 'r') as f:
    data = json.load(f)
 
# 遍历 JSON 数据
for entry in data:
    title = entry['title']  # 获取文章编号
    if title!='1':
        continue
    else:
        article_text = entry['article']  # 获取文章内容
        
        # 创建 Article 节点
        article_node = Node("Article", id=title, name=article_text)
        graph.create(article_node)
        
        # 创建 token 节点
        token_nodes = {}
        for parse_tree in entry['parse_trees']:
            head_idx = parse_tree['h_idx']
            child_idx = parse_tree['t_idx']
            head_word = parse_tree['head']
            child_word = parse_tree['child']
            relation_type = parse_tree['r']
            
            # 如果头 token 不在字典中，则创建并保存
            if head_idx not in token_nodes:
                head_token = Node("Token", id=head_idx, name=head_word)
                graph.create(head_token)
                token_nodes[head_idx] = head_token
            else:
                head_token = token_nodes[head_idx]

            # 如果子 token 不在字典中，则创建并保存
            if child_idx not in token_nodes:
                child_token = Node("Token", id=child_idx, name=child_word)
                graph.create(child_token)
                token_nodes[child_idx] = child_token
            else:
                child_token = token_nodes[child_idx]
            
            # 创建依存关系
            dep_rel = Relationship(head_token, relation_type, child_token)
            graph.create(dep_rel)
        
        # 创建 mentions 节点并关联到对应的 token 节点
        for mention in entry['mentions']:
            mention_id = mention['mention_id']
            mention_name = mention['name']
            mention_type = mention['type']
            token_index = mention['token_index']  # 获取关联的 token 索引
            
            # 创建一个 Mention 节点
            mention_node = Node("Mention", id=mention_id, name=mention_name, type=mention_type)
            graph.create(mention_node)

            # 将 Mention 节点与相应的 Token 节点关联
            for t_i in token_index:
                if t_i in token_nodes:
                    token_node = token_nodes[t_i]
                    mention_rel = Relationship(mention_node, "RELATED_TO", token_node)
                    graph.create(mention_rel)
     
print("数据已成功导入到 Neo4j 中。")





# # 查询所有节点和关系，排除 Article 节点
# query = """
# MATCH (n)-[r]->(m)
# WHERE NOT 'Article' IN labels(n) AND NOT 'Article' IN labels(m)
# RETURN n, r, m
# LIMIT 250
# """

# # 执行查询并返回结果
# results = graph.run(query).data()

# # 初始化 NetworkX 图
# G = nx.DiGraph()

# # 创建字典来存储不同类型的节点
# token_nodes = []
# mention_nodes = []

# # 遍历结果并添加节点和边
# for result in results:
#     # 根据节点类型来获取属性和区分节点
#     node1_type = result['n'].labels  # 获取节点类型
#     node2_type = result['m'].labels
#     print(node1_type,node1_type)
#     # 对节点 1 进行处理，只处理非 Article 节点
#     node1 = result['n'].get('word', result['n'].get('name', 'Unknown'))
#     if 'Token' in node1_type:
#         token_nodes.append(node1)
#     elif 'Mention' in node1_type:
#         mention_nodes.append(node1)

#     # 对节点 2 进行处理，只处理非 Article 节点
#     node2 = result['m'].get('word', result['m'].get('name', 'Unknown'))
#     if 'Token' in node2_type:
#         token_nodes.append(node2)
#     elif 'Mention' in node2_type:
#         mention_nodes.append(node2)

#     # 添加节点和边到图中
#     G.add_node(node1)
#     G.add_node(node2)
#     relation = result['r'].type  # 获取关系类型
#     G.add_edge(node1, node2, label=relation)

# # 绘制图形
# plt.figure(figsize=(15, 15))  # 调整图像大小

# # 使用 spring_layout，并调整 k 参数来控制节点之间的距离
# pos = nx.spring_layout(G, k=0.5, iterations=50)

# # 根据不同类型的节点设置颜色
# nx.draw_networkx_nodes(G, pos, nodelist=token_nodes, node_color='skyblue', node_size=1500, label='Token')
# nx.draw_networkx_nodes(G, pos, nodelist=mention_nodes, node_color='lightgreen', node_size=1500, label='Mention')

# # 绘制边，减少节点大小，增加边宽度
# nx.draw_networkx_edges(G, pos, edge_color='gray', width=2)

# # 绘制节点标签
# nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

# # # 绘制边的标签（关系类型），并旋转以避免重叠
# # edge_labels = nx.get_edge_attributes(G, 'label')
# # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, rotate=True)

# # 显示图形
# plt.legend(scatterpoints=1)  # 添加图例
# plt.show()

# # 保存图形
# # plt.savefig('/root/autodl-tmp/xsum/little_data/fig_colored_without_article.jpg')

# plt.savefig('/root/autodl-tmp/xsum/little_data/fig.jpg')

