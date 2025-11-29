import os
import json
from tqdm import tqdm
import numpy as np

def load_data(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def process_all_splits(input_base_dir
                     ):
    
    splits = ['test']#, 'train', 'val'
    sections = ['article', 'candidate']#, 'humman_summary'
    need_tme_emb = [True, True, True]

    count = 0
    
    ##******************unisumeval专用，开始******************
    _data = []
    with open('/root/autodl-fs/zyq/UniSumEval/merged_file2.jsonl', 'r') as f:
        for line in f:
            _data.append(json.loads(line))

    haserror_list = [item['doc_id'] for item in _data if item['summary_success_state'] == 'success' and item['faithfulness_score'] !=1]
    print(haserror_list)
    ##******************unisumeval专用，结束******************
    
    for split in splits:
        in_dir = os.path.join(input_base_dir, split)
     
        for fname in tqdm(os.listdir(in_dir)):
            # count +=1
            # if count >10:
            #     break
            if not fname.lower().endswith('.json'):
                continue
                
            ##******************unisumeval专用，开始******************
            if fname.strip('.json') not in haserror_list:
                continue
            ##******************unisumeval专用，结束******************
            in_path = os.path.join(in_dir, fname)

            input_data = load_data(in_path)
            text = input_data['article']['text'].split(' ')

            # if 2000<len(text)<4000:
            print(fname, input_data['article']['ent_relation'])

if __name__ == '__main__':
    BASE_INPUT = '/root/autodl-fs/zyq/UniSumEval/factgraph_result'#unisumeval_data_gawl
   
    process_all_splits(BASE_INPUT)  
    print('All graphs processed.')
