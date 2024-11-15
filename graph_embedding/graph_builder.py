
import networkx as nx
import json
import matplotlib.pyplot as plt

class GraphBuilder:
    def __init__(self):
        self.G = None  # 初始化时不创建图
        

    def build_graph(self, data):
        self.G = nx.DiGraph()  # 有向图，如果关系是双向的，可以改用 nx.Graph()
        self.entity_nodes = []
        self.mention_nodes = []
        self.token_nodes=[]
        self.article_node=[]

        self.data = data
        self.create_token_mention_entity_nodes_and_relations()
        self.create_entity_relationships()
        
        return self.G
    
    def create_node(self,label,node_id,name,type=None):
        if type==None:
            self.G.add_node(node_id,node_id=node_id, label=label, name=name)
        else:
            self.G.add_node(node_id,node_id=node_id, label=label, name=name,type=type)

    def create_tokenNode(self,id,name):
        token_id = f"Token_{id}"
        if token_id not in self.token_nodes:
            self.create_node("Token",token_id,name)
            self.token_nodes.append(token_id) 
        return token_id
    
    def create_mentionNode(self,id,name,type):
        mention_id = f"Mention_{id}"
        if mention_id not in self.mention_nodes:
            self.create_node("Mention",mention_id,name,type)
            self.mention_nodes.append(mention_id) 
        else: 
            print("mention node %s already exist."%(id))
        return mention_id
    
    def create_entityNode(self,id,name, wikipedia_id,wikidata_id):
        entity_id = f"Entity_{id}"
        if entity_id not in self.entity_nodes:
            self.G.add_node(entity_id, node_id=entity_id,label="Entity",name=wikipedia_id, wikidata_id=wikidata_id)
            self.entity_nodes.append(entity_id) 
        return entity_id
    
    def create_token_mention_entity_nodes_and_relations(self):
        # 创建 token 节点
        for parse_tree in self.data['parse_trees']:
            head_idx = parse_tree['h_idx']
            child_idx = parse_tree['t_idx']
            head_word = parse_tree['head']
            child_word = parse_tree['child']
            relation_type = parse_tree['r']
            
            # 如果头 token 不在字典中，则创建并保存
            head_token_id=self.create_tokenNode(head_idx,head_word)
            # 如果子 token 不在字典中，则创建并保存
            child_token_id=self.create_tokenNode(child_idx,child_word)
            
            # 创建依存关系
            self.G.add_edge(head_token_id, child_token_id, relationship=relation_type)
        
        # 创建 mentions 节点并关联到对应的 token 节点
        for mention in self.data['mentions']:
            mention_id = mention['mention_id']
            mention_name = mention['name']
            mention_type = mention['type']
            token_index = mention['token_index']  # 获取关联的 token 索引
            
            # 创建一个 Mention 节点
            mention_node_id = self.create_mentionNode(mention_id,mention_name,mention_type)

            # 将 Mention 节点与相应的 Token 节点关联
            for t_i in token_index:
                token_node_id=f"Token_{t_i}"
                if token_node_id in self.token_nodes:
                    self.G.add_edge(mention_node_id, token_node_id, relationship="RELATED_TO")

        for ent_id, entity in enumerate(self.data['vertexSet']):
            # 创建实体节点
            ent_name=entity[0]['name']
            wikipedia_id=entity[0]['wikipedia_id']
            wikidata_id=entity[0]['wikidata_id']
            entity_node_id = self.create_entityNode(ent_id,ent_name, wikipedia_id,wikidata_id)
            
            # mention 节点与实体节点建立关系
            for mention in entity:
                mention_id = mention['mention_id']
                mention_node_id=f"Mention_{mention_id}"
                # 创建 mention 属于实体的关系
                self.G.add_edge(mention_node_id, entity_node_id, relationship="BELONGS_TO")

    def create_entity_relationships(self):
        related_entities = set()  # 用于跟踪已有关系的实体
        entity_relationships_data=self.data['ent_relation']
        for relation in entity_relationships_data:

            head_entity_idx = relation['h_idx']
            tail_entity_idx = relation['t_idx']
            relation_type = relation['r']
            
            # 获取头实体和尾实体节点
            head_entity_id=f"Entity_{head_entity_idx}"
            tail_entity_id=f"Entity_{tail_entity_idx}"

            
            # 创建实体之间的关系
            self.G.add_edge(head_entity_id, tail_entity_id, relationship=relation_type)
            
            # 添加到已有关系的实体集合
            related_entities.add(head_entity_id)
            related_entities.add(tail_entity_id)

    def display_graph_info(self):
        print(nx.info(self.G))




