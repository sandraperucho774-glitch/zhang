import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import psycopg2
from sklearn.preprocessing import MinMaxScaler
import joblib  # 用于保存归一化工具 scaler
import os

# ================= 配置区域 =================
# 数据库连接
DB_CONF = {
    "host": "localhost",
    "database": "weather",
    "user": "postgres",
    "password": "postgres"
}

# 超参数配置
INPUT_SIZE = 2  # 特征数 (温度, 降水)
HIDDEN_SIZE = 64  # LSTM 隐藏层神经元数量
NUM_LAYERS = 2  # LSTM 层数
OUTPUT_SIZE = 2  # 输出数 (预测未来的温度, 降水)
SEQ_LENGTH = 7  # 过去 7 天 -> 预测第 8 天
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50  # 训练轮数 (可根据电脑性能调整)
MODEL_PATH = 'model_lstm.pth'
SCALER_PATH = 'scaler.pkl'

# 检查是否有 GPU (有 N 卡就用 GPU 跑，没有就用 CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Training on device: {device}")


# ================= 1. 数据准备 =================

def get_training_data():
    """从数据库读取 ERA5 数据，并聚合为【日】数据"""
    print("⏳ 正在从数据库读取历史数据...")
    conn = psycopg2.connect(**DB_CONF)

    # SQL 逻辑：将格点数据按【天】聚合，取全区域的平均温度和最大降水
    # 这样代表了合肥市当天的整体气候特征
    sql = """
        SELECT 
            DATE(record_time) as day, 
            AVG(temperature) as avg_temp, 
            MAX(precipitation) as max_prcp
        FROM era5_data
        GROUP BY day
        ORDER BY day ASC;
    """
    df = pd.read_sql(sql, conn)
    conn.close()

    print(f"✅ 读取完成，共获取 {len(df)} 天的历史气象记录。")
    return df


class WeatherDataset(Dataset):
    """自定义 PyTorch 数据集，用于生成滑动窗口样本"""

    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        # 取连续 7 天的数据作为输入 x
        x = self.data[index: index + self.seq_length]
        # 取第 8 天的数据作为标签 y
        y = self.data[index + self.seq_length]
        return torch.FloatTensor(x), torch.FloatTensor(y)


# ================= 2. 模型定义 =================

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 层
        # batch_first=True 意味着输入数据的维度是 (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 全连接层 (将 LSTM 的输出映射到最终的温度/降水数值)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态 h0 和细胞状态 c0
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # 前向传播
        # out 形状: (batch, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))

        # 我们只需要最后一个时间步的输出用于预测
        out = out[:, -1, :]

        # 通过全连接层
        out = self.fc(out)
        return out


# ================= 3. 训练流程 =================

def train():
    # --- A. 数据清洗与归一化 ---
    df = get_training_data()

    # 提取特征列 (温度, 降水)
    dataset_raw = df[['avg_temp', 'max_prcp']].values.astype('float32')

    # 归一化 (将数据缩放到 0~1 之间，这对 LSTM 收敛至关重要)
    scaler = MinMaxScaler()
    dataset_scaled = scaler.fit_transform(dataset_raw)

    # 保存归一化工具，以后预测时要用来反归一化
    joblib.dump(scaler, SCALER_PATH)
    print(f"💾 Scaler 已保存至 {SCALER_PATH}")

    # --- B. 划分训练集和测试集 ---
    train_size = int(len(dataset_scaled) * 0.8)  # 80% 训练
    train_data = dataset_scaled[:train_size]
    test_data = dataset_scaled[train_size:]

    # 创建 DataLoader
    train_dataset = WeatherDataset(train_data, SEQ_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- C. 初始化模型 ---
    model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)
    criterion = nn.MSELoss()  # 损失函数：均方误差
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # 优化器：Adam

    print("🔥 开始训练 LSTM 模型...")

    # --- D. 训练循环 ---
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for i, (sequences, labels) in enumerate(train_loader):
            sequences = sequences.to(device)
            labels = labels.to(device)

            # 1. 清空梯度
            optimizer.zero_grad()

            # 2. 前向传播
            outputs = model(sequences)

            # 3. 计算损失
            loss = criterion(outputs, labels)

            # 4. 反向传播与优化
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {total_loss / len(train_loader):.6f}')

    # --- E. 保存模型 ---
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"🎉 模型训练完成！已保存至 {MODEL_PATH}")


if __name__ == "__main__":
    train()