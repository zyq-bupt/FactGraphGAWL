# config.py
import argparse

def parse_arguments(file):
    datafile = '/root/autodl-fs/zyq/data_gawl'
    # blink_models_path = '/root/autodl-fs/zyq/BLINK/models'
    # kb_data_path = '/root/autodl-fs/zyq/BLINK/data/KB_data'

    config = {
        "spacy_model_name": 'en_core_web_trf',
        # "dreeam_output_file_path":datafile+'/dreeam_result/%s/'%(file),
        "dreeam_output_file_path":datafile+'/dreeam_result/%s/'%(file),
        "factgraph_output_file_path": datafile+'/factgraph_result/%s/'%(file),

        "pretrain_model_path": '/root/autodl-fs/zyq/pegasus-xsum',
        "wikidata5m_path": '/root/autodl-fs/zyq/rotate_wikidata5m.pkl',
        "wikidata5m_entity_path": '/root/autodl-fs/zyq/wikidata5m_entity.txt',
        "wikidata5m_relation_path": '/root/autodl-fs/zyq/wikidata5m_relation.txt',
        "graph_similarity_output_file_path": datafile+'/graph_similarity_result_1/%s/'%(file),
        # "": '/root/',

    }
    # 返回 argparse.Namespace 对象
    return argparse.Namespace(**config)
