import os
import geopandas as gpd
import xarray as xr
import rioxarray
from shapely.geometry import mapping
os.environ['PROJ_LIB'] = r'D:\app\QGIS 3.44.3\share\proj'
# 1. 加载合肥市的矢量边界数据
print("正在加载合肥市矢量边界数据...")
# 确保相对路径正确，指向你的 geojson 文件
hefei_shp = gpd.read_file("../data/shp/合肥市.geojson")

if hefei_shp.crs != "EPSG:4326":
    hefei_shp = hefei_shp.to_crs("EPSG:4326")

# 2. 准备读取和裁剪 ERA5 数据
# 指向你刚刚解压出来的那个真实的 .nc 文件！
file_path = r"E:\ERA5_Data\hefei_era5_data_new\data_stream-oper_stepType-instant.nc"
output_path = r"E:\ERA5_Data\hefei_era5_clipped_final.nc"

if not os.path.exists(file_path):
    print(f"真找不到文件！请确认解压出来的文件名对不对：\n{file_path}")
else:
    try:
        print("正在打开真正解压后的 ERA5 数据文件...")
        # 真正的 NetCDF4 文件，用这个引擎绝对秒开
        ds = xr.open_dataset(file_path, engine="netcdf4")
        print("太棒了，成功打开文件！")

        # 3. 配置 rioxarray 空间维度和坐标系
        ds.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
        ds.rio.write_crs("epsg:4326", inplace=True)

        # 4. 执行裁剪 (Masking)
        print("正在使用合肥市矢量边界进行精确裁剪...")
        geometries = hefei_shp.geometry.apply(mapping)
        ds_clipped = ds.rio.clip(geometries, hefei_shp.crs, drop=True, all_touched=True)

        # 5. 保存裁剪后的精细化数据
        ds_clipped.to_netcdf(output_path)
        print(f"🎉 裁剪大功告成！已生成合肥市精确边界范围的气象数据：\n{output_path}")

    except Exception as e:
        print(f"操作失败，错误信息：{e}")