import json
import os
from tqdm import tqdm

def extract_relevant_links(graph):
    """提取 Mention-Entity 和 Entity-Entity 两类关系"""
    label_map = {n["id"]: n["label"].lower() for n in graph["nodes"]}
    name_map = {n["id"]: n["name"] for n in graph["nodes"]}
    mention_entity_links = set()
    entity_entity_links = set()

    for link in graph["links"]:
        s, t = link["source"], link["target"]
        if s not in label_map or t not in label_map:
            continue
        types = {label_map[s], label_map[t]}
        s_name = name_map[s]
        t_name = name_map[t]
        # Mention ↔ Entity
        if types == {"mention", "entity"}:
            mention_entity_links.add(frozenset([s_name, t_name]))

        # Entity ↔ Entity
        elif types == {"entity"}:
            entity_entity_links.add(frozenset([s_name, t_name]))

    return mention_entity_links, entity_entity_links


def compare_graph_links(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    # 修正字段名
    cand_graph = data["candidate"]["graph_without_emb"]
    human_graph = data["humman_summary"]["graph_without_emb"]

    cand_me, cand_ee = extract_relevant_links(cand_graph)
    human_me, human_ee = extract_relevant_links(human_graph)

    # 各类边的交集
    
    common_me = cand_me & human_me
    common_ee = cand_ee & human_ee
    if cand_me == human_me:
        com_me=1
    else:
        com_me=0
    if cand_ee == human_ee:
        com_ee=1
    else:
        com_ee=0
    return com_me, com_ee


def main():
    path = '/root/autodl-fs/zyq/defacto_data_gawl/factgraph_result_withemb/'
    total_files = 0
    total_me_common = 0
    total_ee_common = 0

    for split in ['test', 'train', 'val']:
        input_path = os.path.join(path, split)
        for filename in tqdm(os.listdir(input_path), desc=f"Processing {split}"):
            if filename.endswith('.json'):
                print(filename)
                json_path = os.path.join(input_path, filename)
                me_count, ee_count = compare_graph_links(json_path)
                
                total_me_common += me_count
                total_ee_common += ee_count
            

    print(f"总文件数: {total_files}")
    print(f"✅ Mention↔Entity 相同的总数: {total_me_common}")
    print(f"✅ Entity↔Entity 相同的总数: {total_ee_common}")


if __name__ == "__main__":
    main()
