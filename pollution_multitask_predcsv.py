import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ---------- 1. 定义模型结构 ----------
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
            ) for _ in range(6)  # 输出个数
        ])

    def forward(self, x):
        shared_feat = self.shared(x)
        outputs = [head(shared_feat) for head in self.output_heads]
        return torch.cat(outputs, dim=1)


# ---------- 2. 加载数据 ----------
csv_path = r'F:\pollutions\Sichuan_study\Sichuan_data\a_method\b_pollutions_site_based\method_1\result_1_method_multitask/Sichuan_2023all_meteo_aod_all1.csv'
df = pd.read_csv(csv_path)

# 过滤掉负值
df = df[~(df[["PM10", "PM2.5", "O3", "CO", "NO2", "SO2"]] < 0).any(axis=1)]

# 特征列
feature_columns = ['lon', 'lat', 'doy', 'hour', 'rh', 'blh', 'tcw', 'tp',
                   'sp', 't2m', 'u10', 'v10', 'aod', 'dem', 'slope', 'aspect']

# 提取并标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_columns])
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# 数据划分
train_index, test_index = train_test_split(df.index, test_size=0.2, random_state=42)

# 添加 train_test-set 列
df['train_test-set'] = 0
df.loc[test_index, 'train_test-set'] = 1

# ---------- 3. 加载模型 ----------
model = MultiTaskNet(input_dim=16, output_dim=6)
model.load_state_dict(torch.load(
    r"F:\pollutions\Sichuan_study\Sichuan_data\a_method\b_pollutions_site_based\method_1\result_1_method_multitask\multitask_model.pth",
    map_location=torch.device('cpu')  # 加上这个参数
))
model.eval()

# ---------- 4. 进行预测 ----------
with torch.no_grad():
    preds = model(X_tensor).numpy()

# ---------- 5. 将预测结果添加到 DataFrame ----------
output_names = ["PM10", "PM2.5", "O3", "CO", "NO2", "SO2"]
for i, name in enumerate(output_names):
    df[f'pred_{name}'] = preds[:, i]

# ---------- 6. 保存带预测结果的新 CSV ----------
output_csv_path = "./Sichuan_2023all_meteo_aod_all1_pred.csv"
df.to_csv(output_csv_path, index=False)
print(f"预测结果已保存到：{output_csv_path}")
