# FactGraphGAWL

## 摘要
事实图构建与 GAWL 方法改进。

---

## 🧩 `construc_graph.py`
**输入：**
```
autodl-fs/zyq/data_gawl/dreeam_result/
```

**输出：**
```
autodl-fs/zyq/data_gawl/factgraph_result/
```

**功能说明：**  
该脚本用于将 `dreeam_result` 转换为图结构 `factgraph_result`。  
在 dreeam 阶段，虽然已经得到了 token、mention、entity 及其间的 relation，但尚未形成完整的图结构。  
本脚本重新组织这些信息，生成 `graph_without_emb`。

---

## 🔠 `get_token_EMB.py`
**输入：**
```
autodl-fs/zyq/data_gawl/factgraph_result/
```

**输出：**
```
autodl-fs/zyq/data_gawl/factgraph_result_withemb/
```

**功能说明：**  
为节点添加 embedding，加载 `pegasus-xsum` 预训练模型。  
节点 embedding 取自其所在句子的上下文嵌入。  
同一词汇在原文与摘要中可能有不同的 embedding，可用于 PT 图核相似度方法。

**小技巧：**  
学习到一个新的映射方法，可将自定义分词与模型 tokenizer 的分词对应起来（见代码 63–77 行）：

```python
sents = [["I", "love", "programming"]]

# 假设 tokenizer 将 "programming" 分成 "program" + "ming"
# [["I", "love", "program", "ming"]]
encoding['input_ids'] = [101, 146, 1568, 30767, 2561, 102]
encoding.word_ids(batch_index=0)
# 输出: [None, 0, 1, 2, 3, 3, None]
# 表示编码出的第4和第5个 embedding 映射到 sents 的第3个单词。
```

---

## 🧮 `gawl.py`
主函数，用于计算两个图的相似性。

### 一、GAWL 改进内容
在原始 GAWL 方法基础上，进行了三项改进：

1. **节点标签改进**  
   将节点 label 由度数替换为单词。  
   （注意：未剔除停用词，因为方法对结构敏感，涉及多层邻居聚合，若删除中间关系或单词可能影响性能。）  
   → 对应函数：`compute_gawl_kernel`

2. **小图标签频率改进**  
   直接使用小图 *j* 的 label 频率。  
   虽然提升不大，但该修改更符合摘要任务逻辑。  
   原因：原本的 `min()` 函数已默认以小图为基准。

3. **边类型加权改进（关键）**  
   → 对应函数：`compute_gawl_kernel_v2`

#### 边类型设计
| 类型 | 描述 |
|------|------|
| **大类** | T-T = 1，T-M = 2，M-E = 3，E-E = 1.5。<br>其中：<br>nsubj、obj 属于 token-token；<br>RELATED_TO 属于 token-mention；<br>BELONGS_TO 属于 mention-entity；<br>P17、P27 属于 entity-entity。<br>文件：`edge_label_type.txt` 记录每条边的大类。 |
| **小类** | `edge_label.txt` 中存储了具体子类型（如 nsubj、obj、RELATED_TO、BELONGS_TO、P17、P27 等）。目前每个小类出现一次计权重为 1。 |

> 📎 方法细节见论文附录或飞书文档：[FactGraphGAWL文档](https://mu85k14jge.feishu.cn/docx/LDNvdEKL7oh3itxnC4WcK87OnId?from=from_copylink)  
> 参考原论文：[Nikolentzos & Vazirgiannis, 2023](https://github.com/giannisnik/gawl)(Nikolentzos G, Vazirgiannis M. Graph alignment kernels using weisfeiler and leman hierarchies[C]//International Conference on Artificial Intelligence and Statistics. PMLR, 2023: 2019-2034.)

---

### 二、函数流程与参数说明
```python
use_node_labels = True  # True 表示使用单词作为节点 label。后续可继续改进，用embedding作为label。
use_edge_labels = 1     # 0: 不使用边权重；1: 使用小类权重(edge_minor_type_weights)；2: 使用大类权重(edge_major_type_weights)
```

---

## 📁 输入文件说明
**Defacto 数据集：**
```
data_gawl
```

下载链接（百度网盘）：
> 链接：[https://pan.baidu.com/s/1j7Z4mjbtc3Me1BKXxxSekw](https://pan.baidu.com/s/1j7Z4mjbtc3Me1BKXxxSekw)  
> 提取码：`2tns`

---

## ⚙️ 其他注意事项
```python
keys = ["article", "candidate", "humman_summary"]
# 注意！UniSum 的键名不同！
```

---

## 📊 UniSumEval上与人类得分的相关性评估

### `gawl.py`
**输出路径：**
```
/root/autodl-fs/zyq/unisumeval_data_gawl/factgraphGAWL/test
```

---

## 🧾 `edit2.py`
**输入：**
```python
merged_file = "/root/autodl-fs/zyq/unisumeval_data_gawl/merged_file2.jsonl"
docs_directory = "/root/autodl-fs/zyq/unisumeval_data_gawl/factgraphGAWL/test/"
```

**输出：**
```
/root/autodl-fs/zyq/unisumeval_data_gawl/merged_file2_processed.jsonl
```

**功能：**  
将上一步得到的得分与原始文件进行合并。

---

## 📈 `evaluators_benchmark.py`
评估脚本，用于验证模型在不同任务上的表现。

---