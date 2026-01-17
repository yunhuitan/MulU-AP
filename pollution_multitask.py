import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# ---------- 设置设备 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ---------- 1. 读取和处理数据 ----------
num_features = 13
num_outputs = 6
df = pd.read_csv('./Sichuan_2023all_meteo_aod_all1.csv')
df = df[(df[["PM10", "PM2.5", "O3", "CO", "NO2", "SO2"]] > 0).all(axis=1)]

feature_columns = ['lon', 'lat', 'doy', 'hour', 'rh', 'blh', 'tcw', 'tp',
                   'sp', 't2m', 'u10', 'v10', 'aod']
assert len(feature_columns) == num_features

df_x = df[feature_columns]
scaler = StandardScaler()
df_x_scaled = scaler.fit_transform(df_x)

output_names = ["PM10", "PM2.5", "O3", "CO", "NO2", "SO2"]
df_y = df[output_names]

dfx_train, dfx_val, y_train, y_val = train_test_split(df_x_scaled, df_y, test_size=0.2, random_state=42)

X_train = torch.tensor(dfx_train, dtype=torch.float32).to(device)
Y_train = torch.tensor(y_train.values, dtype=torch.float32).to(device)
X_val = torch.tensor(dfx_val, dtype=torch.float32).to(device)
Y_val = torch.tensor(y_val.values, dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# ---------- 2. 定义模型 ----------
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

# ---------- 3. 不确定性加权损失函数 ----------
class UncertaintyLoss(nn.Module):
    def __init__(self, num_tasks):
        super(UncertaintyLoss, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))  # 每个任务一个 log(sigma^2)

    def forward(self, preds, targets):
        losses = []
        for i in range(preds.shape[1]):
            precision = torch.exp(-self.log_vars[i])
            task_loss = precision * (preds[:, i] - targets[:, i])**2 + self.log_vars[i]
            losses.append(task_loss.mean())
        return sum(losses)

# ---------- 4. 初始化模型 ----------
model = MultiTaskNet(input_dim=num_features, output_dim=num_outputs).to(device)
criterion = UncertaintyLoss(num_outputs).to(device)
optimizer = optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=0.001)
epoch_num = 300

# ---------- 5. 模型训练 ----------
best_val_loss = float('inf')
best_model_path = "./multitask_model-best.pth"

for epoch in range(epoch_num + 1):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 验证阶段
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"🔺 Epoch {epoch+1}: 验证损失改善为 {val_loss:.4f}，模型已保存为 {best_model_path}")
    else:
        print(f"Epoch {epoch+1}: 验证损失 {val_loss:.4f}（未改善）")

    print(f"训练损失: {total_loss:.4f}，当前 log_vars: {criterion.log_vars.data.cpu().numpy()}")

# ---------- 6. 加载最优模型 ----------
model.load_state_dict(torch.load(best_model_path))
print(f"✅ 加载最优模型用于评估：{best_model_path}")

# ---------- 7. 模型验证 ----------
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for x_batch, y_batch in val_loader:
        pred = model(x_batch)
        all_preds.append(pred.cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())

all_preds = np.vstack(all_preds)
all_labels = np.vstack(all_labels)

# ---------- 8. 指标计算 ----------
print("\n--- 多任务预测评估指标 ---")
for i in range(num_outputs):
    r2 = r2_score(all_labels[:, i], all_preds[:, i])
    mae = mean_absolute_error(all_labels[:, i], all_preds[:, i])
    rmse = mean_squared_error(all_labels[:, i], all_preds[:, i], squared=False)
    print(f"{output_names[i]} - R2: {r2:.4f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# ---------- 9. 保存每个输出变量的散点图 ----------
os.makedirs("scatter_plots_test", exist_ok=True)
for i in range(num_outputs):
    plt.figure(figsize=(6, 6))
    plt.scatter(all_labels[:, i], all_preds[:, i], alpha=0.5)
    plt.title(f"{output_names[i]} Prediction")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    min_val = min(all_labels[:, i].min(), all_preds[:, i].min())
    max_val = max(all_labels[:, i].max(), all_preds[:, i].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.tight_layout()
    save_path = f"./scatter_plots_test/scatter_{output_names[i]}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"保存散点图：{save_path}")
