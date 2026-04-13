import xarray as xr
import matplotlib.pyplot as plt

# 1. 打开我们刚刚历经千辛万苦裁剪好的终极文件
file_path = r"E:\ERA5_Data\hefei_era5_clipped_final.nc"
ds = xr.open_dataset(file_path)

# 2. 提取第一个时间点的 2 米温度数据（注意这里换成了 valid_time）
temp_data = ds['t2m'].isel(valid_time=0)

# 3. 画图展示合肥市的温度空间分布
plt.figure(figsize=(8, 6))
# 绘制温度的空间分布，cmap='jet' 是气象常用的伪彩色带
temp_data.plot(cmap='jet')
plt.title("Hefei Temperature Spatial Distribution (Clipped)")
plt.show()

# 释放内存
ds.close()