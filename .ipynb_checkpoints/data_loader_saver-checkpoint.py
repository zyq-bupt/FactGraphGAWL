# import pandas as pd
import json
import networkx as nx

def load_data(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"读取文件 {dataset_path} 时出错: {e}")
    return data

def data_save(data: dict, outputfile: str):
    if 'doc_name' in data.keys():
        doc_id = data['doc_id']
    else:
        doc_id = data['doc_id']
    with open(outputfile+str(doc_id)+'.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

