from graph_embedding.graph_builder import GraphBuilder
from graph_embedding.get_node_embedding import GraphEmbeddingProcessor
from shortest_path_graph.graph.shortest_path_graph import ShortestPathGraph
from shortest_path_graph.k_score.k_counter import KCounter
from data_loader_saver import load_data, data_save

from config import parse_arguments
import os
import json
import networkx as nx

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
def caculate_graphSim():
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


def main(args):
    graph_builder= GraphBuilder()
    graph_processor = GraphEmbeddingProcessor(args)
    text_list=['article','abstract','candidate','humman_summary']
    for filename in os.listdir(args.dreeam_output_file_path):
        if filename.endswith('.json'):
            file_path = os.path.join(args.dreeam_output_file_path, filename)
            input_data = load_data(file_path)

            new_data = input_data.copy()
            print(input_data['doc_id'])
            for key,item in input_data.items():
                if key in text_list:
                    graph=graph_builder.build_graph(item)
                    graph_with_emb = graph_processor.process_graph_embeddings(graph)
                    new_data[key]['graph_with_emb']=graph_with_emb
                
            data_save(new_data, args.factgraph_output_file_path)

            #下面可以接着算两两之间的sp相似度，先跑完上面的，然后把存储屏蔽了，save放在最后，把相似度也存进去。
            data_with_graphSim = caculate_graphSim(new_data)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
   
    