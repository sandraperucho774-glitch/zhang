import cdsapi
import os

c = cdsapi.Client()

hefei_area = [32.6, 116.6, 31.0, 117.9]
save_dir = r"E:\ERA5_Data"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 循环 10 年，每年 12 个月
years = [str(y) for y in range(2021, 2026)]
months = [f"{m:02d}" for m in range(1, 13)]

print(f"🚀 开始精细化挂机下载 2016-2025 年数据...")

for year in years:
    for month in months:
        # 给每个月的文件起个唯一名，防止覆盖
        target_file = os.path.join(save_dir, f"era5_{year}_{month}.nc")

        # 断点续传：如果文件已经存在且大于 1MB，就认为是下好了的
        if os.path.exists(target_file) and os.path.getsize(target_file) > 1024 * 1024:
            continue

        print(f"📡 正在下载 {year} 年 {month} 月数据...")

        try:
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': [
                        '2m_temperature',
                        'total_precipitation',
                        '10m_u_component_of_wind',
                        '10m_v_component_of_wind',
                    ],
                    'year': year,
                    'month': month,
                    'day': [f"{d:02d}" for d in range(1, 32)],
                    'time': [f"{h:02d}:00" for h in range(0, 24)],
                    'area': hefei_area,
                },
                target_file
            )
            print(f"✅ {year}年{month}月数据存入: {target_file}")
        except Exception as e:
            print(f"❌ {year}年{month}月下载失败，原因: {e}")

print("🎉 全部数据任务处理完毕！")