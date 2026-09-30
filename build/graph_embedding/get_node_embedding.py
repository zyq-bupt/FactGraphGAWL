
import networkx as nx
import json
import matplotlib.pyplot as plt
import torch

# from transformers import pipeline
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import pickle
import numpy as np
# device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GraphEmbeddingProcessor:
    def __init__(self,args):
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # self.feature_extractor = pipeline("feature-extraction", framework="pt", model=args.pretrain_model_path, device=device)
        
        self.tokenizer = PegasusTokenizer.from_pretrained(args.pretrain_model_path)
        self.model = PegasusForConditionalGeneration.from_pretrained(args.pretrain_model_path)  # Note: Use PegasusModel, not PegasusForConditionalGeneration
        self.model.to(device)


        self.wikidata5m_model=self.load_wikidata5m_model(args.wikidata5m_path)
        self.entity2id = self.wikidata5m_model.graph.entity2id


    def load_wikidata5m_model(self,model_path):
        with open(model_path, "rb") as fin:
            model = pickle.load(fin)
        return model    
    
    
    def get_token_pegasus_embedding(self, text):#,pegasus, 
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding="longest").to(device)
        with torch.no_grad():  # 禁用梯度计算（推理时用）
            encoder_outputs = self.model.model.encoder(**inputs)  # 访问模型的编码器部分

        embedding = encoder_outputs.last_hidden_state.mean(dim=1).squeeze().cpu()  # 获取token嵌入
        return embedding
    
    def get_mention_pegasus_embedding(self, text):#pegasus
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding="longest").to(device)
        with torch.no_grad():  # 禁用梯度计算（推理时用）
            encoder_outputs = self.model.model.encoder(**inputs)  # 访问模型的编码器部分
       
        embedding = encoder_outputs.last_hidden_state.mean(dim=1).squeeze().cpu()  
        return embedding
    
    
    def get_wikidata5m_embedding(self,ent_name,wikidata_id):
        
        if wikidata_id in self.entity2id.keys():
            # print('searchid:',entity,wkdataid)
            entity_emb=self.entity_embeddings[self.entity2id[wikidata_id]]
        else:
            entity_emb=np.zeros(512)
            print(f"No id found for embeding '{ent_name}'.")
        
        return entity_emb
    

    def process_graph_embeddings(self, graph):
        self.G = graph
        self.embeddings = {}
        for node, attr in self.G.nodes(data=True):
            label = attr.get('label')
            if label == "Token":
                text = attr.get('name', '')
                self.embeddings[node] = self.get_token_pegasus_embedding(text)
            elif label == "Mention":
                text = attr.get('name', '')
                self.embeddings[node] = self.get_mention_pegasus_embedding(text)
                
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
    