import shap
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm
from matplotlib import cm, rcParams
from sklearn.model_selection import train_test_split

# ---------- 0. 设置设备 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ---------- 1. 模型结构和参数 ----------
num_features = 16
num_outputs = 6
feature_columns = ['lon', 'lat', 'doy', 'hour', 'rh', 'blh', 'tcw', 'tp',
                   'sp', 't2m', 'u10', 'v10', 'aod', 'dem', 'slope', 'aspect']
output_names = ["PM10", "PM2.5", "O3", "CO", "NO2", "SO2"]

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    def forward(self, x):
        return x + self.block(x)

class MultiTaskNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultiTaskNet, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            nn.Linear(800, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            ResidualBlock(100),
            ResidualBlock(100),
            ResidualBlock(100)
        )
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.Linear(50, 25),
                nn.ReLU(),
                nn.Linear(25, 1)
            )
            for _ in range(output_dim)
        ])

    def forward(self, x):
        shared_feat = self.shared(x)
        outputs = [head(shared_feat) for head in self.output_heads]
        return torch.cat(outputs, dim=1)

# 加载模型
model = MultiTaskNet(input_dim=num_features, output_dim=num_outputs)
model.load_state_dict(torch.load(r"./test/multitask_model-best.pth", map_location=device))
model.to(device)
model.eval()

# ---------- 2. 加载数据 ----------
df = pd.read_csv('./test/Sichuan_2023all_meteo_aod_all1.csv')
df = df[~(df[["PM10", "PM2.5", "O3", "CO", "NO2", "SO2"]] < 0).any(axis=1)]

df_x = df[feature_columns]
scaler = StandardScaler()
df_x_scaled = scaler.fit_transform(df_x)
df_y = df[output_names]

dfx_train, dfx_val, y_train, y_val = train_test_split(df_x_scaled, df_y, test_size=0.2, random_state=42)

X_train = torch.tensor(dfx_train, dtype=torch.float32).to(device)
X_val = torch.tensor(dfx_val, dtype=torch.float32).to(device)

# ---------- 3. 准备 SHAP 解释器 ----------
background = X_train[:100]  # 背景样本
test_samples = X_val[:100]        # 要解释的样本
explainer = shap.DeepExplainer(model, background)

# ---------- 4. 计算 SHAP 值 ----------
shap_values = explainer.shap_values(test_samples)

# ---------- 5. 可视化 ----------
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 18
rcParams['font.weight'] = 'bold'
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.titlesize'] = 20

os.makedirs("shap_summary_plots_importance", exist_ok=True)

for i in range(num_outputs):
    cmap = cm.get_cmap('RdBu')
    colors = [cmap(x) for x in np.linspace(0, 1, 16)]
    colors = colors[::-1]
    shap_vals_for_output_i = shap_values[:, :, i]

    print(f"绘制 SHAP Summary 重要性图：{output_names[i]}")
    fig = plt.gcf()
    shap.summary_plot(shap_vals_for_output_i, test_samples.cpu().numpy(), feature_names=feature_columns,
                      show=False, plot_type="bar", color=colors)
    plt.xticks(fontname='Times New Roman', fontweight='bold', fontsize=18)
    plt.yticks(fontname='Times New Roman', fontweight='bold', fontsize=24)
    ax = plt.gca()
    ax.set_xlabel(ax.get_xlabel(), fontsize=18, fontweight='bold', fontname='Times New Roman')

    bar_save_path = f"./shap_summary_plots_importance/shap_summary_importance_{output_names[i]}.png"
    fig.savefig(bar_save_path, dpi=600, bbox_inches='tight')

    print(f"保存重要性图至 {bar_save_path}")

    ##保存为csv：
    # 获取 SHAP 值和特征值（都为 numpy）
    shap_array = shap_vals_for_output_i  # shape: (N_samples, N_features)
    feature_array = test_samples.cpu().numpy()  # shape: (N_samples, N_features)

    # 创建 DataFrame
    df_shap = pd.DataFrame(shap_array, columns=[f"SHAP_{col}" for col in feature_columns])
    df_feat = pd.DataFrame(feature_array, columns=feature_columns)

    # 拼接两个表格
    df_combined = pd.concat([df_feat, df_shap], axis=1)

    # 保存为 CSV
    csv_path = f"./shap_summary_plots_importance/shap_values_{output_names[i]}.csv"
    df_combined.to_csv(csv_path, index=False)
    print(f"✅ SHAP 值和特征保存为 CSV：{csv_path}")



