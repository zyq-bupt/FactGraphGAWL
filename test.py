import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 生成示例数据
np.random.seed(10)
data = {
    '样本1': np.random.normal(loc=0.8, scale=0.1, size=100),
    '样本2': np.random.normal(loc=0.6, scale=0.1, size=100),
    '样本3': np.random.normal(loc=0.75, scale=0.1, size=100),
    '样本4': np.random.normal(loc=0.9, scale=0.1, size=100)
}

# 将数据转换为 DataFrame
df = pd.DataFrame(data)

# 设置 Seaborn 样式
sns.set(style="whitegrid")

# 绘制箱形图
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, palette='viridis')

# 添加标题和标签
plt.title('样本分布箱形图')
plt.xlabel('样本')
plt.ylabel('值')

# 显示图形
plt.tight_layout()
plt.show()
