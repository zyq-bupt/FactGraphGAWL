from graph_embedding.graph_builder import GraphBuilder
# from graph_embedding.get_node_embedding import GraphEmbeddingProcessor
from data_loader_saver import load_data, data_save

from config import parse_arguments
import os
import json
import networkx as nx
import numpy as np
from tqdm import tqdm


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
                    new_data[key]['graph_without_emb'] = gdata

                    # graph_with_emb = graph_processor.process_graph_embeddings(graph)
                    # new_data[key]['graph_with_emb'] = graph_with_emb

            data_save(new_data, args.factgraph_output_file_path)

            #下面可以接着算两两之间的sp相似度.
            
if __name__ == '__main__':
    flist=['test','val','train']
    for f in flist:
        args = parse_arguments(f)
        main(args)



    