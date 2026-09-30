from build.graph_embedding.graph_builder import GraphBuilder
# from build.graph_embedding.get_node_embedding import GraphEmbeddingProcessor
from common.data_loader_saver import load_data, data_save

from config import parse_arguments
import os
import json
import networkx as nx
import numpy as np
from tqdm import tqdm


def main(args):
    graph_builder= GraphBuilder()
    # graph_processor = GraphEmbeddingProcessor(args)
    
    # text_list=['article','abstract','candidate','humman_summary']#defacto
    text_list=['article','abstract','candidate']#unisumeval
    for filename in tqdm(os.listdir(args.dreeam_output_file_path)):
        if filename.endswith('.json'):
            # filename = '2214.json'
            # docid = filename.split('.')[0]
            # if docid in right_list:

            file_path = os.path.join(args.dreeam_output_file_path, filename)
            input_data = load_data(file_path)

            new_data = input_data.copy()
            # print(input_data['doc_id'])
            for key,item in input_data.items():
                if key in text_list:
                    graph=graph_builder.build_graph(item)
                    ##1 如果不要节点embedding，直接用下面两行代码，屏蔽掉2
                    gdata = nx.node_link_data(graph)
                    new_data[key]['graph_without_emb'] = gdata


                    ##2 如果要节点embedding，那么就运行下面代码且屏蔽掉1
                    # graph_with_emb = graph_processor.process_graph_embeddings(graph)
                    # new_data[key]['graph_with_emb'] = graph_with_emb

            data_save(new_data, args.factgraph_output_file_path)

            #下面可以接着算两两之间的sp相似度.
            
if __name__ == '__main__':
    # flist=['test','val','train']
    flist = ['test']
    for f in flist:
        args = parse_arguments(f)
        main(args)



    