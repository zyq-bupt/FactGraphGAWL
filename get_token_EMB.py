import os
import json
from tqdm import tqdm
import numpy as np
import torch
from transformers import PegasusTokenizerFast, PegasusModel


def add_graph_embeddings(input_json_path, output_json_path,
                         pegasus_model_path='/root/autodl-fs/zyq/pegasus-xsum'):
    """
    Load JSON graph, compute embeddings for Token, Mention, and Entity nodes
    for article, candidate, and human_summary sections,
    and write out augmented JSON.
    """
    # 1. Load input data
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. Load Pegasus tokenizer & model once per process (cache globally)
    global _tokenizer, _model, _device, _entity2id, _entity_embeddings
    if '_tokenizer' not in globals():
        _tokenizer = PegasusTokenizerFast.from_pretrained(pegasus_model_path, use_fast=True)
        _model = PegasusModel.from_pretrained(pegasus_model_path)
        _model.eval()
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _model.to(_device)

    tokenizer = _tokenizer
    model = _model
    device = _device
    embed_dim = model.config.d_model
    # helper functions
    def mention_embedding(text: str) -> torch.Tensor:
        enc = tokenizer(text, return_tensors='pt', truncation=True,
                        padding='longest').to(device)
        with torch.no_grad():
            out = model.get_encoder()(**enc)
        return out.last_hidden_state.mean(dim=1).squeeze().cpu()

    def entity_embedding(wikidata_id: str) -> np.ndarray:
        
        return np.zeros(model.config.d_model)

    # 3. Compute and inject embeddings for each section
    sections = ['article', 'candidate', 'humman_summary']
    for sec in sections:
        sents = data[sec]['sents']
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

        for node in data[sec]['graph_without_emb']['nodes']:
            label = node['label']
            
            if label == 'Token':
                idx = int(node['node_id'].split('_')[1])
                if 0 <= idx < len(flat_word_embs):
                    node['embedding'] = flat_word_embs[idx].tolist()
                else:
                    # print(f"Warning: Token index {idx} out of range (0 to {len(flat_word_embs)-1}). Using zero vector.")
                    wrong_json_path = '/root/autodl-fs/zyq/data_gawl/factgraph_result/wrong.json'

                    # 如果文件不存在，创建空 list
                    if os.path.exists(wrong_json_path):
                        with open(wrong_json_path, 'r', encoding='utf-8') as f:
                            wrongdata = json.load(f)
                    else:
                        wrongdata = []

                    # 加入新路径
                    wrongdata.append(input_json_path.replace('/root/autodl-fs/zyq/data_gawl/factgraph_result', ''))

                    # 写回 json
                    with open(wrong_json_path, 'w', encoding='utf-8') as f:
                        json.dump(wrongdata, f, ensure_ascii=False, indent=2)
                 
                    node['embedding'] = [0.0] * embed_dim
            elif label == 'Mention':
                text = node.get('name', '')
                node['embedding'] = mention_embedding(text).tolist()
            elif label == 'Entity':
                wid = node.get('wikidata_id')
                node['embedding'] = entity_embedding(wid).tolist()

    # 4. Write output
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process_all_splits(input_base_dir,
                       output_base_dir,
                       pegasus_model_path='/root/autodl-fs/zyq/pegasus-xsum'
                     ):
    """
    Process all .json files under test, train, val splits.
    """
    splits = ['test', 'train', 'val']
    # splits = ['tt']
    for split in splits:
        in_dir = os.path.join(input_base_dir, split)
        out_dir = os.path.join(output_base_dir, split)
        os.makedirs(out_dir, exist_ok=True)
        for fname in tqdm(os.listdir(in_dir)):
            # fname = '1968.json'
            if not fname.lower().endswith('.json'):
                continue
            in_path = os.path.join(in_dir, fname)
            out_path = os.path.join(out_dir, fname)
            # print(f'Processing {in_path} -> {out_path}')
            add_graph_embeddings(in_path, out_path,
                                 pegasus_model_path=pegasus_model_path)


if __name__ == '__main__':
    BASE_INPUT = '/root/autodl-fs/zyq/data_gawl/factgraph_result'
    BASE_OUTPUT = '/root/autodl-fs/zyq/data_gawl/factgraph_result_withemb'
    process_all_splits(BASE_INPUT, BASE_OUTPUT)  
    print('All graphs processed.')
