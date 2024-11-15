# FactGraphSP
摘要，事实图，short path改进

# main.py
主函数，三个功能：从dreeam结果构建图结构（GraphBuilder）、为token、mention、entity节点得到embedding（GraphEmbeddingProcessor）、计算两个图的相似性（最短路径图核方法ShortestPathGraph、KCounter）

# main_without_embedding.py ！！！！
没有embedding功能的主函数。这个运行的快一点，因为后期才用到embedding，这里为了试验各种参数的效果，可以不得到embedding。
在下面的函数里修改参数：
```
def caculate_graphSim(g1, g2):
    d = 50
    init_w=1/100000
```
# merge_file.py
把train、test、val都合并到一个文件中，为了运行速度快。

# evaluation.py
评估factgraph和其他factscore的结果。比较三组数据：abstract、candidate、humman与原文的相似度结果。
内部错误和外部错误分开评估。
评估项目：
1.计算每组数据的中位数、下四分位数、上四分位数。
2.画出箱线图。
3.计算每一组数据是否符合正态分布假设。
4.如果不符合正态分布，那么采用friedmanchisquare计算三组之间的显著性。
5.用wilcoxon评估两两之间的显著性。

注意：friedmanchisquare和wilcoxon都适用于评估成对比较的样本。

# visual_graph.py
可以在neo4j中展示出文本构建出的图结构。
