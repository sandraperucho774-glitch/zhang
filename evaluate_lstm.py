import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import psycopg2
import joblib
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(2, 64, 2, batch_first=True)
        self.fc = nn.Linear(64, 2)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

scaler = joblib.load('scaler.pkl')
model = LSTMModel()
model.load_state_dict(torch.load('model_lstm.pth', map_location='cpu'))
model.eval()

DB_CONF = {"host": "localhost", "database": "weather", "user": "postgres", "password": "postgres"}

print("⏳ 正在从数据库获取历史数据，寻找合适的验证区间...")
conn = psycopg2.connect(**DB_CONF)
sql = """
    SELECT DATE(record_time) as day, AVG(temperature) as temp, MAX(precipitation) as prcp
    FROM era5_data
    GROUP BY day 
    ORDER BY day ASC;
"""
df = pd.read_sql(sql, conn)
conn.close()

if len(df) < 14:
    print("❌ 数据库里的总天数不足 14 天，无法完成验证！")
    exit()

start_idx = len(df) - 50
end_idx = start_idx + 14
test_df = df.iloc[start_idx:end_idx].copy()

print(f"✅ 成功截取 14 天验证数据: 从 {test_df['day'].iloc[0]} 到 {test_df['day'].iloc[-1]}")

input_raw = test_df[['temp', 'prcp']].values[:7]
input_dates = pd.to_datetime(test_df['day']).dt.strftime('%m-%d').values[:7]

truth_raw = test_df[['temp', 'prcp']].values[7:14]
truth_dates = pd.to_datetime(test_df['day']).dt.strftime('%m-%d').values[7:14]

input_scaled = scaler.transform(input_raw)
current_seq = torch.FloatTensor(input_scaled).unsqueeze(0)

preds = []
for i in range(7):
    with torch.no_grad():
        pred = model(current_seq)
        preds.append(pred.numpy()[0])
        current_seq = torch.cat((current_seq[:, 1:, :], pred.unsqueeze(1)), dim=1)

preds_real = scaler.inverse_transform(np.array(preds))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(truth_dates, truth_raw[:, 0], marker='o', label='真实温度 (Ground Truth)', color='green')
plt.plot(truth_dates, preds_real[:, 0], marker='x', linestyle='--', label='LSTM 预测温度', color='red')
plt.title('连续 7 天温度预测效果对比 (盲测集)')
plt.ylabel('温度 (℃)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.subplot(1, 2, 2)
plt.bar(truth_dates, truth_raw[:, 1], alpha=0.5, label='真实降水', color='blue')
plt.plot(truth_dates, preds_real[:, 1], marker='s', linestyle='--', label='LSTM 预测降水趋势', color='orange')
plt.title('连续 7 天降水预测效果对比 (盲测集)')
plt.ylabel('降水量 (mm)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('LSTM_Validation_Result.png', dpi=300)
print("🎉 验证对比图已成功生成：请在文件夹中查看 LSTM_Validation_Result.png ！")
plt.show()