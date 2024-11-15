from graph_embedding.graph_builder import GraphBuilder
from graph_embedding.get_node_embedding import GraphEmbeddingProcessor
from shortest_path_graph.graph.shortest_path_graph import ShortestPathGraph
from shortest_path_graph.k_score.k_counter import KCounter
from data_loader_saver import load_data, data_save

from config import parse_arguments
import os
import json
import networkx as nx
import numpy as np
from tqdm import tqdm

def caculate_graphSim(g1, g2):
    d = 50
    init_w=1/100000
    c1 = ShortestPathGraph(g1, d, init_w).get_graph()
    c2 = ShortestPathGraph(g2, d, init_w).get_graph()
    similarity = KCounter(c1, c2).get_k_score()

    return similarity

def main_graphSim(args):
    # right_list = ['522','266','1218','1712','2042','2611']##'1448',139, '1196'
    text_list=['article','abstract','candidate','humman_summary']
    for filename in tqdm(os.listdir(args.factgraph_output_file_path)):
        
        # docid = filename.split('.')[0]
        # if docid in right_list:
        #     print(filename)
        if filename.endswith('.json'):
            file_path = os.path.join(args.factgraph_output_file_path, filename)
            graph_data = load_data(file_path)
            new_data = graph_data.copy()
            new_data['graph_sim'] = {}

            
            for k1 in range(len(text_list)):
                key1 = text_list[k1]
                for k2 in range(k1+1,len(text_list)):
                    key2 = text_list[k2]
                    g1=nx.node_link_graph(graph_data[key1]['graph_with_emb'])
                    g2=nx.node_link_graph(graph_data[key2]['graph_with_emb'])
                    new_data['graph_sim']['%s_%s'%(key1,key2)] = caculate_graphSim(g1,g2)
                    
                        # print('%s_%s'%(key1,key2),new_data['graph_sim']['%s_%s'%(key1,key2)])
                        # print('****',new_data[key2]['text'])
            data_save(new_data, args.graph_similarity_output_file_path)


def main(args):
    graph_builder= GraphBuilder()

    text_list=['article','abstract','candidate','humman_summary']
    for filename in tqdm(os.listdir(args.dreeam_output_file_path)):
        if filename.endswith('.json'):

            # docid = filename.split('.')[0]
            # if docid in right_list:

            file_path = os.path.join(args.dreeam_output_file_path, filename)
            input_data = load_data(file_path)

            new_data = input_data.copy()
            # print(input_data['doc_id'])
            for key,item in input_data.items():
                if key in text_list:
                    graph=graph_builder.build_graph(item)
                    gdata = nx.node_link_data(graph)
                    new_data[key]['graph_with_emb'] = gdata

                    # graph_with_emb = graph_processor.process_graph_embeddings(graph)
                    # new_data[key]['graph_with_emb'] = graph_with_emb

            data_save(new_data, args.factgraph_output_file_path)

            #下面可以接着算两两之间的sp相似度.
            
if __name__ == '__main__':
    flist=['test','val','train']
    for f in flist:
        args = parse_arguments(f)
        main(args)
        main_graphSim(args)


    