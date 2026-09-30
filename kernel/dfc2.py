#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import sys
from pathlib import Path

def is_graph_file_empty(input_path: Path, graph_keys: list[str]) -> bool:
    with input_path.open('r', encoding='utf-8') as fin:
        data = json.load(fin)
    for key in graph_keys:
        if key in data and 'graph_without_emb' in data[key]:
            G = data[key]['graph_without_emb']
            if G.get('nodes') or G.get('links'):
                return False
    return True

def convert_json_graph(
    input_path: Path | None = None,
    graph_keys: list[str] | None = None,
    output_prefix: str = "",
    undirected: bool = False,
    use_emb_labels: bool = False,
    output_dir: Path = Path("."),
    data: dict | None = None,
):
    if graph_keys is None:
        graph_keys = ["article", "candidate", "humman_summary"]
    # 1. 读取 JSON（也可直接传入内存中的 dict，供扰动实验使用）
    if data is None:
        if input_path is None:
            raise ValueError("convert_json_graph 需要 input_path 或 data")
        with input_path.open('r', encoding='utf-8') as fin:
            data = json.load(fin)

    # 2. 初始化
    current_node_id = 1
    graph_indicator: list[str] = []
    edges: list[str] = []
    edge_labels: list[str] = []
    edge_types: list[str] = []    # 新增：记录每条边的大类
    graph_labels = ['0'] * len(graph_keys)
    node_emb_labels_list = []
  

    # 统一全局名字到标签（可选）
    name2label: dict[str, int] = {}
    node_name_labels: list[str] = []

    # 3. 依次处理各子图
    for gid, key in enumerate(graph_keys, start=1):
        if key not in data or 'graph_without_emb' not in data[key]:
            print(f"⚠️ 跳过不存在的子图 `{key}`", file=sys.stderr)
            continue
        
        ###
        # ent_relation = data[key]['ent_relation']
        # if len(ent_relation)!=0:
        #     for item in ent_relation:
        #         print(item['r'])
        ###


        G = data[key]['graph_without_emb']
       
        # 本地节点->name 映射
        local_name_map = {n['node_id']: n.get('name', '') for n in G.get('nodes', [])}

        # 3.1 本地节点 ID 列表
        local_ids = list(local_name_map.keys())
        # 3.2 本地->全局映射
        local2global: dict[int, int] = {}
        for lid in local_ids:
            local2global[lid] = current_node_id
            graph_indicator.append(str(gid))

            # #记录name和id的映射
            # node_name = local_name_map.get(lid, "")
            # node_id_name_map.append((current_node_id, node_name))

            # 记录名字标签
            name = local_name_map[lid]
            if name not in name2label:
                name2label[name] = len(name2label)
         
            node_name_labels.append(str(name2label[name]))

            if use_emb_labels:
                # --- 提取 embedding ---
                node = next((n for n in G['nodes'] if n['node_id'] == lid), None)
                if node is None:
                    print(f"⚠️ 没找到 node_id={lid} 的 embedding", file=sys.stderr)
                    emb = [0.0] * 1024   # 也可以改成 raise error
                else:
                    emb = node['embedding']
                
                node_emb_labels_list.append(emb)
            

            current_node_id += 1

        


        # 辅助函数：根据节点 ID 前缀判断节点类型
        def node_category(nid: str) -> str:
            if nid.startswith('Token_'):
                return 'token'
            elif nid.startswith('Mention_'):
                return 'mention'
            elif nid.startswith('Entity_'):
                return 'entity'
            else:
                return 'unknown'
            
        # 3.3 遍历 links，输出边和边标签
        for link in G.get('links', []):
            src, tgt = link['source'], link['target']
            if src not in local2global or tgt not in local2global:
                print(f"⚠️ 链接 {src}->{tgt} 中有未映射节点，已跳过", file=sys.stderr)
                continue
            s_glob = local2global[src]
            t_glob = local2global[tgt]
            rel = link.get('relationship', '')

            # 计算大类
            cat1 = node_category(src)
            cat2 = node_category(tgt)
            if cat1 == 'token' and cat2 == 'token':
                et = 'token-token'
            elif {cat1, cat2} == {'token', 'mention'}:
                et = 'token-mention'
            elif {cat1, cat2} == {'mention', 'entity'}:
                et = 'mention-entity'
            elif cat1 == 'entity' and cat2 == 'entity':
                et = 'entity-entity'
            else:
                et = f'{cat1}-{cat2}'

            # 添加有向边及其标签
            edges.append(f"{s_glob}, {t_glob}")
            edge_labels.append(str(rel))
            edge_types.append(et)

            if undirected:
                # 反向边也使用相同类型
                edges.append(f"{t_glob}, {s_glob}")
                edge_labels.append(str(rel))
                edge_types.append(et)


    # 4. 写文件
    out_dir = output_dir                # ← 使用传入的目录
    prefix = output_prefix

    (out_dir / f"{prefix}graph_indicator.txt").write_text(
        '\n'.join(graph_indicator), encoding='utf-8')
    (out_dir / f"{prefix}A.txt").write_text(
        '\n'.join(edges), encoding='utf-8')
    (out_dir / f"{prefix}graph_labels.txt").write_text(
        '\n'.join(graph_labels), encoding='utf-8')
    (out_dir / f"{prefix}node_name_labels.txt").write_text(
        '\n'.join(node_name_labels), encoding='utf-8')
   
    (out_dir / f"{prefix}edge_labels.txt").write_text(
        '\n'.join(edge_labels), encoding='utf-8')
    # 新增：写入每条边的粗分类（大类）
    (out_dir / f"{prefix}edge_label_type.txt").write_text(
        '\n'.join(edge_types), encoding='utf-8')


    if use_emb_labels:
        with (output_dir / f"{output_prefix}node_emb_labels.json").open("w", encoding="utf-8") as fout:
            json.dump(node_emb_labels_list, fout, indent=2)
    
    # 写入节点编号和名称映射
    node_map_path = output_dir / f"{output_prefix}node_id_name_mapping.json"
    with node_map_path.open("w", encoding="utf-8") as fout:
        json.dump(name2label, fout, indent=2)
    # print("✔ 输出完成：",
    #       f"{prefix}graph_indicator.txt,",
    #       f"{prefix}A.txt,",
    #       f"{prefix}graph_labels.txt,",
    #       f"{prefix}node_name_labels.txt,",
    #       f"{prefix}edge_labels.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="转换 JSON 子图为全局节点/边，输出统一节点名标签和边类型标签"
    )
    parser.add_argument("-i", "--input", default='/root/autodl-fs/zyq/data_gawl/factgraph_result_withemb/tt/100.json',
                        help="输入 JSON 文件路径")
    parser.add_argument("-k", "--keys", nargs="+",
                        default=["article", "candidate", "humman_summary"],
                        help="子图 key 列表，顺序决定 graph_indicator 编号")
    parser.add_argument("-p", "--prefix", default="",
                        help="输出文件名前缀（可选）")
    parser.add_argument("-u", "--undirected", action="store_true",
                        help="写入反向边，将有向图当作无向图处理")
    parser.add_argument("-o", "--outdir", default='/root/autodl-fs/zyq/data_gawl/factgraph_result_withemb/tt/',           # ← 新增：输出目录参数
                        help="输出文件所在目录 (默认当前目录)")
    args = parser.parse_args()

    convert_json_graph(
        input_path=Path(args.input),
        graph_keys=args.keys,
        output_prefix=args.prefix,
        undirected=args.undirected
    )
