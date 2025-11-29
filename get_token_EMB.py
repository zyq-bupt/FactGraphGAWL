import os
import json
from tqdm import tqdm
import numpy as np
import torch
from transformers import PegasusTokenizerFast, PegasusModel
import pickle

def load_token_model(pegasus_model_path):
    tokenizer = PegasusTokenizerFast.from_pretrained(pegasus_model_path, use_fast=True)
    model = PegasusModel.from_pretrained(pegasus_model_path)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    embed_dim = model.config.d_model

    return tokenizer, model, embed_dim, device

def load_wikidata5m_model(wikidata5m_path):
        
        with open(wikidata5m_path, "rb") as fin:
            model = pickle.load(fin)
        entity2id = model.graph.entity2id
        entity_embeddings = model.solver.entity_embeddings
        return entity2id,entity_embeddings  

def load_data(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_data(data, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def add_graph_embeddings(data,sections,need_tme_emb, tokenizer, model, embed_dim, device, entity2id,entity_embeddings ):
    """
    Load JSON graph, compute embeddings for Token, Mention, and Entity nodes
    for article, candidate, and human_summary sections,
    and write out augmented JSON.
    """
    # helper functions
    def mention_embedding(text: str) -> torch.Tensor:
        enc = tokenizer(text, return_tensors='pt', truncation=True,
                        padding='longest').to(device)
        with torch.no_grad():
            out = model.get_encoder()(**enc)
        return out.last_hidden_state.mean(dim=1).squeeze().cpu()
        
    def entity_embedding(wikidata_id: str,entity2id,entity_embeddings) -> np.ndarray:
        if wikidata_id in entity2id.keys():
            # print('searchid:',entity,wkdataid)
            entity_emb=entity_embeddings[entity2id[wikidata_id]]
            return entity_emb
        else:
            return np.zeros(model.config.d_model)

    def get_token_emb(sents):
        encoding = tokenizer(
            sents,
            is_split_into_words=True,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        with torch.no_grad():
            encoder_outputs = model.get_encoder()(input_ids=input_ids,
                                                  attention_mask=attention_mask)
        hidden_states = encoder_outputs.last_hidden_state.cpu()

        word_id_seqs = [encoding.word_ids(batch_index=i) for i in range(len(sents))]
        all_word_embs = []
        for sent_idx, word_ids in enumerate(word_id_seqs):
            num_words = len(sents[sent_idx])
            embs = []
            for wid in range(num_words):
                positions = [i for i, w in enumerate(word_ids) if w == wid]
                if positions:
                    sub_embs = hidden_states[sent_idx, positions, :]
                    embs.append(sub_embs.mean(dim=0))
                else:
                    # No subtoken positions, use zero vector
                    embs.append(torch.zeros(embed_dim))
            all_word_embs.append(embs)
        flat_word_embs = [emb for sent_emb in all_word_embs for emb in sent_emb]
        return flat_word_embs

    # 3. Compute and inject embeddings for each section
    for sec in sections:
        sents = data[sec]['sents']        
        if need_tme_emb[0]==True:
            flat_word_embs = get_token_emb(sents)
        for node in data[sec]['graph_without_emb']['nodes']:
            label = node['label']

            if label == 'Token' and need_tme_emb[0]==True:
                idx = int(node['node_id'].split('_')[1])
                if 0 <= idx < len(flat_word_embs):
                    node['embedding'] = flat_word_embs[idx].tolist()
                else:
                #     # print(f"Warning: Token index {idx} out of range (0 to {len(flat_word_embs)-1}). Using zero vector.")
                #     wrong_json_path = '/root/autodl-fs/zyq/data_gawl/factgraph_result/wrong.json'

                #     # 如果文件不存在，创建空 list
                #     if os.path.exists(wrong_json_path):
                #         with open(wrong_json_path, 'r', encoding='utf-8') as f:
                #             wrongdata = json.load(f)
                #     else:
                #         wrongdata = []

                    # # 加入新路径
                    # wrongdata.append(input_json_path.replace('/root/autodl-fs/zyq/data_gawl/factgraph_result', ''))

                    # # 写回 json
                    # with open(wrong_json_path, 'w', encoding='utf-8') as f:
                    #     json.dump(wrongdata, f, ensure_ascii=False, indent=2)
                 
                    node['embedding'] = [0.0] * embed_dim
            elif label == 'Mention' and need_tme_emb[1]==True:
                text = node.get('name', '')
                node['embedding'] = mention_embedding(text).tolist()
            elif label == 'Entity' and need_tme_emb[2]==True:
                wid = node.get('wikidata_id')
                node['embedding'] = entity_embedding(wid,entity2id,entity_embeddings).tolist()
    return data
def process_all_splits(input_base_dir,
                       output_base_dir,
                       pegasus_model_path='/root/autodl-fs/zyq/pegasus-xsum'
                     ):
    """
    Process all .json files under test, train, val splits.
    """
    wikidata5m_path = '/root/autodl-fs/zyq/rotate_wikidata5m.pkl'

    tokenizer, model, embed_dim, device = load_token_model(pegasus_model_path)
    entity2id,entity_embeddings = load_wikidata5m_model(wikidata5m_path)
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
        out_dir = os.path.join(output_base_dir, split)
        os.makedirs(out_dir, exist_ok=True)
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
            out_path = os.path.join(out_dir, fname)

            input_data = load_data(in_path)
            out_data = add_graph_embeddings(input_data,sections, need_tme_emb, tokenizer, model, embed_dim, device, entity2id,entity_embeddings )
            save_data(out_data, out_path)

if __name__ == '__main__':
    BASE_INPUT = '/root/autodl-fs/zyq/UniSumEval/factgraph_result'#unisumeval_data_gawl
    BASE_OUTPUT = '/root/autodl-fs/zyq/UniSumEval/factgraph_result_withemb'
    process_all_splits(BASE_INPUT, BASE_OUTPUT)  
    print('All graphs processed.')
