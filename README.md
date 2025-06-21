# FactGraphGAWL
摘要，事实图，GAWL改进

# construc_graph.py 
输入：autodl-fs/zyq/data_gawl/dreeam_result/
输出：autodl-fs/zyq/data_gawl/factgraph_result/
功能：
这个函数主要用于将dreeam_result变成图结构factgraph_result。
dreeam结束后，token、mention、entity、以及他们之间的relation，都有了，但是不是图结构，所以要重新组织一下，得到graph_without_emb。
```

```
# get_token_EMB.py
输入：autodl-fs/zyq/data_gawl/factgraph_result/
输出：autodl-fs/zyq/data_gawl/factgraph_result_withemb/
功能：
给节点增加embedding，加载了pegasus-xsum预训练模型，节点emb是所在句子的上下文嵌入。例如：原文和摘要中包含同一个词，但是其emb却可能不一样。可用于PT图核相似度方法。

ps：学习到了一个新的方法，可以将自己的分词和模型tokenizer的分词map起来。见代码63行-77行。

sents = [["I", "love", "programming"]]

假设 tokenizer 把 "programming" 分成 "program" + "ming"，[["I", "love", "program", "ming"]]

encoding['input_ids'] = [101, 146, 1568, 30767， 2561, 102]

encoding.word_ids(batch_index=0) [None, 0, 1, 2, 3, 3, None]，意思是编码出的低4和第5个embedding，映射到我自己分割的sents的第3个。

# gawl.py
主函数，计算两个图的相似性。

一、在GAWL方法的基础上进行了3种改进：

1、将度变为单词形式，即把节点label变成单词。（注意，这里使用了全部单词和关系，没有剔除停用词，因为本方法对结构敏感，而且涉及多层邻居聚合，如果失去了某些中间关系和单词，方法性能会下降。）【对应compute_gawl_kernel函数】

2、直接用小图 j 的 label 频率。（这个方法提升效果小，但是改进目前完全符合摘要任务，提升效果小的原因是，原本的min函数就是以小图为基准，因此大多数情况下，都会以摘要为基准。）

3、【关键改进】增加2种边类型加权：【对应compute_gawl_kernel_v2函数】

    3.1 两种边类型：大类是：

    | 边类型   | 方法描述   | 
    | ------- | :-----: |
    | 大类  |T-T = 1，T-M=2，M-E=3，E-E=1.5。nsubj、obj属于token-token大类、RELATED_TO属于token-mention大类、BELONGS_TO属于mention-entity大类、P17、P27属于entity-entity大类，文件edge_label_type.txt，记录每条边的大类。| 
    | 小类  |  edge_label.txt中存储的是nsubj、obj、RELATED_TO、BELONGS_TO、P17、P27等等这种每条边的子类型。目前每个小类，出现一次就算权重是1，| 

4、方法描述见论文附录or飞书文档（https://mu85k14jge.feishu.cn/docx/LDNvdEKL7oh3itxnC4WcK87OnId?from=from_copylink）

注：（https://github.com/giannisnik/gawl） Nikolentzos G, Vazirgiannis M. Graph alignment kernels using weisfeiler and leman hierarchies[C]//International Conference on Artificial Intelligence and Statistics. PMLR, 2023: 2019-2034.

二、函数流程和关键参数解释

1、use_node_labels = True，true表示用单词作为节点label。后续可继续改进，用embedding作为label。

2、use_edge_labels = 1  #0:不使用边权重；1:edge_minor_type_weights, 2:edge_major_type_weights


# 输入文件

defacto数据集：data_gawl (通过网盘分享的文件：Defacto_data_gawl.zip  链接: https://pan.baidu.com/s/1j7Z4mjbtc3Me1BKXxxSekw 提取码: 2tns )