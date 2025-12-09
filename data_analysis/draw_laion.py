import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df2 = pd.read_csv('./data_analysis/aesthetic_scores_genvideo.csv')
df2.columns = df2.columns.str.strip()
df2['Model'] = df2['Model'].str.strip()

df2_sorted = df2.sort_values('Aesthetic_Mean', ascending=False)
# 方法3：水平柱状图
# 版本2：用不同颜色标记被排除的模型
exclude_models = ['Kinetics-400', 'MSR-VTT', 'RealVSR', 'Youku']
df_aigv = df2_sorted[~df2_sorted['Model'].isin(exclude_models)]
avg_aigv = df_aigv['Aesthetic_Mean'].mean()
plt.figure(figsize=(12, 10))
y_pos = np.arange(len(df2_sorted))

# 为不同模型设置不同颜色
colors = []
for model in df2_sorted['Model']:
    if model in exclude_models:
        colors.append('#e74c3c')  # 红色，表示被排除的模型
    else:
        colors.append('#3498db')  # 蓝色，表示AI生成视频模型

bars = plt.barh(y_pos, df2_sorted['Aesthetic_Mean'], 
                xerr=df2_sorted['Aesthetic_Std'], 
                capsize=3, alpha=0.7,
                color=colors,
                edgecolor=['darkred' if c=='#e74c3c' else 'darkblue' for c in colors], 
                linewidth=1)

plt.xlabel('Aesthetic Score (higher is better)', fontsize=12)
plt.ylabel('Model', fontsize=12)
plt.title('Aesthetic Scores by Model', 
          fontsize=14, fontweight='bold', pad=20)

# 显示模型名字
plt.yticks(y_pos, df2_sorted['Model'], fontsize=10)
plt.gca().invert_yaxis()  # 让最高的在最上面

# 只在条末端显示分数
for i, (bar, mean_val) in enumerate(zip(bars, df2_sorted['Aesthetic_Mean'])):
    width = bar.get_width()
    model_name = df2_sorted.iloc[i]['Model']
    text_color = 'black'
    plt.text(width + 0.05, bar.get_y() + bar.get_height()/2., 
             f'{mean_val:.2f}', 
             ha='left', va='center', fontsize=12, color=text_color)

# 添加avg_aigv线
plt.axvline(x=avg_aigv, color='red', linestyle='--', 
            alpha=0.7, linewidth=2, label=f'avg_aigv: {avg_aigv:.2f}')

# 添加图例
import matplotlib.patches as mpatches
legend_patches = [
    mpatches.Patch(color='#3498db', label='AI-Generated Video Models'),
    mpatches.Patch(color='#e74c3c', label='Excluded (non-AIGV)'),
    plt.Line2D([0], [0], color='red', linestyle='--', label=f'avg_aigv: {avg_aigv:.2f}')
]
plt.legend(handles=legend_patches, fontsize=9, loc='lower right')

# 在avg_aigv线上添加文字标注
# plt.text(avg_aigv + 0.05, len(df2_sorted)/2, 
#          f'avg_aigv = {avg_aigv:.2f}\n(excluding Kinetics-400 & MSR-VTT)', 
#          rotation=90, va='center', ha='left',
#          fontsize=9, color='red', fontweight='bold',
#          bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.3))

plt.grid(True, alpha=0.1, axis='x', linestyle='--')

x_max = max(df2_sorted['Aesthetic_Mean']) + df2_sorted['Aesthetic_Std'].max() + 0.3
plt.xlim(left=3.0, right=x_max)

plt.tight_layout()
plt.savefig('./data_analysis/results/aesthetic_scores_with_exclusion.png', dpi=300, bbox_inches='tight')
# plt.show()