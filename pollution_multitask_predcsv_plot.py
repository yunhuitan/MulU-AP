import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

# ---------- 1. 读取含预测值的 CSV ----------
df_all = pd.read_csv("./Sichuan_2023all_meteo_aod_all1_pred.csv")

# ---------- 2. 确定输入特征列 ----------
feature_columns = ['lon', 'lat', 'doy', 'hour', 'rh', 'blh', 'tcw', 'tp',
                   'sp', 't2m', 'u10', 'v10', 'aod', 'dem', 'slope', 'aspect']

# ---------- 3. 划分测试集 ----------
# 使用相同 random_state 保证划分一致
X = df_all[feature_columns]
_, X_test, _, df_test = train_test_split(X, df_all, test_size=0.2, random_state=42)

# ---------- 4. 设置真实值和预测值列 ----------
output_vars = ["PM10", "PM2.5", "O3", "CO", "NO2", "SO2"]
pred_vars = [f"pred_{name}" for name in output_vars]

# ---------- 5. 创建图像保存文件夹 ----------
os.makedirs("scatter_plots_testset", exist_ok=True)

# ---------- 6. 绘制散点图 ----------
for true_col, pred_col in zip(output_vars, pred_vars):
    plt.figure(figsize=(6, 6))
    plt.scatter(df_test[true_col], df_test[pred_col], alpha=0.5)

    plt.xlabel(f"Observed {true_col} ")
    plt.ylabel(f"Predicted {true_col}")

    min_val = min(df_test[true_col].min(), df_test[pred_col].min())
    max_val = max(df_test[true_col].max(), df_test[pred_col].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.tight_layout()
    save_path = f"./scatter_plots/scatter_test_{true_col}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"保存测试集散点图：{save_path}")
