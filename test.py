import json
import os 
from tqdm import tqdm
from data_loader_saver import load_data

import json

# 假设你的 JSON 数据已经被加载到 data 字典中
# 例如: data = json.load(open("path_to_file.json"))

def remove_embedding(data):
    # 遍历 JSON 结构中的节点
    if isinstance(data, dict):
        # 如果找到键为 "embedding"，则删除它
        if "embedding" in data:
            del data["embedding"]
        # 递归处理嵌套的字典
        for key in data:
            remove_embedding(data[key])
    elif isinstance(data, list):
        # 如果是列表，递归处理每个元素
        for item in data:
            remove_embedding(item)


# 设置文件夹路径
# folder_path = "/root/autodl-fs/zyq/DeFacto/data/graph_similarity_result/"  # 请将此路径替换为你的文件夹路径
output_file = "/root/autodl-fs/zyq/DeFacto/data/graph_similarity_result/merged_data1.json"

# 存储所有 JSON 数据的列表
merged_data = []

# 遍历文件夹中的所有 JSON 文件
file_list = ['test','val','train']
for file in file_list:
    factgraph_output_file_path = '/root/autodl-fs/zyq/DeFacto/data/graph_similarity_result/%s/'%(file)
    filenames = os.listdir(factgraph_output_file_path)
    for filename in tqdm(filenames):
        if filename.endswith('.json'):
            file_path = os.path.join(factgraph_output_file_path, filename)
            graph_data = load_data(file_path)
            remove_embedding(graph_data)
            merged_data.append(graph_data)

# 将合并后的数据写入一个新的 JSON 文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)

print(f"所有 JSON 文件已成功合并到 {output_file} 文件中。")
