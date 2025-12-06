import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV
df = pd.read_csv('./data_analysis/GenVideo_quality_report.csv')

# 清理列名（去掉空格）
df.columns = df.columns.str.strip()

# 清理Model列的值
df['Model'] = df['Model'].str.strip()

# 排序数据（按NIQE_Mean升序）
df = df.sort_values('NIQE_Mean')

# 创建图形
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# 设置位置
x = np.arange(len(df))
width = 0.35

# 第一个子图：NIQE分数
bars1 = ax1.bar(x - width/2, df['NIQE_Mean'], width, yerr=df['NIQE_Std'], 
                label='NIQE_Mean', capsize=5, color='skyblue')
ax1.set_ylabel('NIQE Score')
ax1.set_title('NIQE Scores by Model')
ax1.set_xticks(x)
ax1.set_xticklabels(df['Model'], rotation=45, ha='right')
ax1.legend()

df = df.sort_values('MUSIQ_Mean')

# 第二个子图：MUSIQ分数
bars2 = ax2.bar(x + width/2, df['MUSIQ_Mean'], width, yerr=df['MUSIQ_Std'], 
                label='MUSIQ_Mean', capsize=5, color='lightcoral')
ax2.set_ylabel('MUSIQ Score')
ax2.set_title('MUSIQ Scores by Model')
ax2.set_xticks(x)
ax2.set_xticklabels(df['Model'], rotation=45, ha='right')
ax2.legend()

plt.tight_layout()
# plt.show()
plt.savefig('./data_analysis/GenVideo_quality_report.png')