import xarray as xr
import pandas as pd

# 1. 读取裁剪好的合肥市 ERA5 数据
file_path = r"E:\ERA5_Data\hefei_era5_clipped_final.nc"
ds = xr.open_dataset(file_path)

print("正在将空间网格数据转换为二维表格...")
# 2. 一键转换为 Pandas DataFrame！
df = ds.to_dataframe()

# 3. 数据清洗：剔除边界外的无效空值 (NaN)
df_clean = df.dropna().reset_index()

# 4. 单位换算与特征工程
print("正在进行气象单位换算...")
# 将温度从开尔文(K)转换为摄氏度(℃)
if 't2m' in df_clean.columns:
    df_clean['t2m_celsius'] = df_clean['t2m'] - 273.15
    # 保留两位小数
    df_clean['t2m_celsius'] = df_clean['t2m_celsius'].round(2)

# 将降水从米(m)转换为毫米(mm)
if 'tp' in df_clean.columns:
    df_clean['tp_mm'] = df_clean['tp'] * 1000
    df_clean['tp_mm'] = df_clean['tp_mm'].round(2)

# 5. 预览并保存结果
print("\n转换成功！数据前 5 行如下：")
# 只挑选我们关心的核心列进行展示
columns_to_show = ['valid_time', 'latitude', 'longitude', 't2m_celsius', 'tp_mm']
# 如果你的数据里没有 tp_mm，就只展示有的列
existing_columns = [col for col in columns_to_show if col in df_clean.columns]
print(df_clean[existing_columns].head())

# 保存为 CSV 文件，这就是深度学习模型的标准“口粮”
csv_path = r"E:\ERA5_Data\hefei_era5_timeseries.csv"
# encoding='utf-8-sig' 确保用 Excel 打开时不会乱码
df_clean[existing_columns].to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\n🎉 完美！时序数据已成功保存至：{csv_path}")

# 释放内存
ds.close()