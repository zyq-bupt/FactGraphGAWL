
import networkx as nx
import json
import matplotlib.pyplot as plt
import torch
# from transformers import BartModel, BartTokenizer
from transformers import BertTokenizer, BertModel
import pickle
import numpy as np


class GraphEmbeddingProcessor:
    def __init__(self,args):
        # device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = BertModel.from_pretrained(args.pretrain_model_path)
        self.tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)
        self.wikidata5m_model=self.load_wikidata5m_model(args.wikidata5m_path)
        self.entity2id = self.wikidata5m_model.graph.entity2id
        self.entity_embeddings = self.wikidata5m_model.solver.entity_embeddings
        # self.alias2entity=self.load_alias2entity(args.wikidata5m_entity_path)
        # self.alias2relation=self.load_alias2relation(args.wikidata5m_relation_path)

    def load_wikidata5m_model(self,model_path):
        with open(model_path, "rb") as fin:
            model = pickle.load(fin)
        return model    
    def load_alias2entity(self,entity_file):
         # 创建 alias2entity 字典5918 'Q5465387'
        alias2entity = {}
        # 读取实体别名文件，构建映射字典
        with open(entity_file, 'r', encoding='utf-8') as f:
            for line in f:
                # 分割行数据，第一个是实体ID，后续是别名
                parts = line.strip().split('\t')
                entity = parts[0]  # 实体ID
                aliases = parts[1:]  # 实体别名

                # 为每个别名创建映射
                for alias in aliases:
                    alias2entity[alias] = entity
        return alias2entity
    
    def get_token_bart_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs) 
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()  # 获取token嵌入
        return embedding
    
    def get_mention_bart_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs) 
        embedding = outputs.pooler_output.squeeze()  # 获取句子嵌入
        return embedding
    
    def get_wikidata5m_embedding(self,ent_name,wikidata_id):
        
        if wikidata_id in self.entity2id.keys():
            # print('searchid:',entity,wkdataid)
            entity_emb=self.entity_embeddings[self.entity2id[wikidata_id]]
        else:
            entity_emb=np.zeros(512)
            print(f"No id found for embeding '{ent_name}'.")
        
        return entity_emb


    def load_alias2relation(self,relation_file):
        # 创建 alias2relation 字典
        alias2relation = {}
        # 读取关系别名文件，构建映射字典
        with open(relation_file, 'r', encoding='utf-8') as f:
            for line in f:
                # 分割行数据，第一个是主键，后续是别名
                parts = line.strip().split('\t')
                relation = parts[0]  # 关系ID
                aliases = parts[1:]  # 关系别名

                # 为每个别名创建映射
                for alias in aliases:
                    alias2relation[alias] = relation

        # 输出字典大小
        print(f"Loaded {len(alias2relation)} alias-relation pairs.")
        return alias2relation
    
    def get_relation_embedding(self,relation):
        relation2id = self.wikidata5m_model.graph.relation2id
        relation_embeddings = self.wikidata5m_model.solver.relation_embeddings
        # 示例: 检查某个别名的映射关系
        relation = 'field of work'  # 可替换为你感兴趣的别名
        if relation in self.alias2relation:
            print(self.alias2relation[relation])
            print(relation_embeddings[relation2id[self.alias2relation[relation]]])
        else:
            print(f"No relation found for alias '{relation}'.")

    def process_graph_embeddings(self, graph):
        self.G = graph
        self.embeddings = {}
        for node, attr in self.G.nodes(data=True):
            label = attr.get('label')
            if label == "Token":
                text = attr.get('name', '')
                self.embeddings[node] = self.get_token_bart_embedding(text)
            elif label == "Mention":
                text = attr.get('name', '')
                self.embeddings[node] = self.get_mention_bart_embedding(text)
            elif label == "Entity":
                ent_name=attr.get('name')
                wikidata_id = attr.get('wikidata_id')
                self.embeddings[node] = self.get_wikidata5m_embedding(ent_name,wikidata_id)
        
        data = nx.node_link_data(self.G)
        # 添加节点嵌入信息到节点属性中
        for node in data['nodes']:
            node_id = node['id']
            if node_id in self.embeddings:
                node['embedding'] = self.embeddings[node_id].tolist()  # 将 tensor 转化为 list
            else:
                pass

        return data
    