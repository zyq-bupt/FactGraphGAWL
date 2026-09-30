import json
import os
from tqdm import tqdm
# 读取并处理merged file
def process_files(merged_path, docs_dir, output_path):
    # 读取merged file的每一行
    with open(merged_path, 'r', encoding='utf-8') as f:
        merged_lines = f.readlines()
    
    processed_lines = []
    
    for line in tqdm(merged_lines):
        # 解析每行的JSON
        data = json.loads(line)
        doc_id = data['doc_id']
        
        # 构建对应的文件路径
        doc_file = os.path.join(docs_dir, f"{doc_id}.json")
        
        # 如果文件存在
        if os.path.exists(doc_file):
            with open(doc_file, 'r', encoding='utf-8') as f:
                doc_data = json.loads(f.read())
                # print(doc_file)
                # 获取graph_sim.article_abstract值
                graph_sim_value = doc_data.get('graph_sim', {}).get('WL-GAWL')
                
                # 添加到machine_evaluation_results_faithfulness
                if 'machine_evaluation_results_faithfulness' not in data:
                    data['machine_evaluation_results_faithfulness'] = {}
                    
                data['machine_evaluation_results_faithfulness']['WL-GAWL-norm'] = graph_sim_value
        
        # 将处理后的数据添加到结果中
        processed_lines.append(json.dumps(data, ensure_ascii=False))
    
    # 写回文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in processed_lines:
            f.write(line + '\n')

# 使用示例
# merged_file = "/root/autodl-fs/zyq/unisumeval_data_gawl/merged_file2.jsonl"
# docs_directory = "/root/autodl-fs/zyq/unisumeval_data_gawl/factgraph_result/test/"
# output_file = "/root/autodl-fs/zyq/unisumeval_data_gawl/merged_file2_processed.jsonl"
# process_files(merged_file, docs_directory, output_file)

import pandas as pd
import json
import pandas as pd
import json

def add_CoKGLM_to_jsonl():
    # 路径
    csv_path = "/root/autodl-tmp/cokglm/verification/nest_cross_val/test_unisumeval_with_pred.csv"
    jsonl_path = "/root/autodl-fs/zyq/unisumeval_data_gawl/merged_file2.jsonl"
    output_path = "/root/autodl-fs/zyq/unisumeval_data_gawl/merged_file2_with_CoKGLM.jsonl"

    # 读取 CSV 文件
    csv_df = pd.read_csv(csv_path)

    # 兼容 doc_name / doc_id 两种情况
    if "doc_name" in csv_df.columns:
        csv_df["doc_id"] = csv_df["doc_name"].astype(str)
    elif "doc_id" in csv_df.columns:
        csv_df["doc_id"] = csv_df["doc_id"].astype(str)
    else:
        raise ValueError("❌ CSV 中必须包含 doc_name 或 doc_id 列")

    # 构建 doc_id → hallucination_prob 的映射
    halueval_map = dict(zip(csv_df["doc_id"], csv_df["hallucination_prob"]))

    total_count = 0
    matched_count = 0

    # 逐行读取 JSONL 并更新
    with open(jsonl_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            data = json.loads(line)
            doc_id = str(data.get("doc_id"))
            total_count += 1

            # 如果存在匹配
            if doc_id in halueval_map:
                score = halueval_map[doc_id]
                matched_count += 1
            else:
                score = 1  # 未匹配的设置为 0

            # 确保嵌套字典存在
            if "machine_evaluation_results_faithfulness" not in data:
                data["machine_evaluation_results_faithfulness"] = {}

            # 写入 halueval 分数
            data["machine_evaluation_results_faithfulness"]["CoKGLM"] = score

            # 写入新文件
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"✅ 处理完成！共处理 {total_count} 行，其中 {matched_count} 行匹配成功。")
    print(f"➡️ 输出文件：{output_path}")


if __name__ == '__main__':
    add_CoKGLM_to_jsonl()

