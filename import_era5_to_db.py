import os
import xarray as xr
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
import glob
import zipfile
import shutil
import tempfile
import time
import gc

DB_CONF = {
    "host": "localhost",
    "database": "weather",
    "user": "postgres",
    "password": "postgres"
}
DATA_DIR = r"E:\ERA5_Data"


def get_db_conn():
    return psycopg2.connect(**DB_CONF)


def is_zip_file(filepath):
    try:
        with open(filepath, 'rb') as f:
            return f.read(4) == b'PK\x03\x04'
    except:
        return False


def force_delete_dir(dir_path):

    for i in range(5):
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            return
        except PermissionError:
            print(f"⚠️ 文件被占用，等待释放 ({i + 1}/5)...")
            time.sleep(1)
            gc.collect()
    print(f"❌ 警告：无法删除临时文件夹 {dir_path}，请稍后手动清理。")


def process_and_import(file_path):
    print(f"--------------------------------------------------")
    print(f"📂 正在分析文件: {os.path.basename(file_path)}")

    temp_dir = None
    ds = None
    datasets_list = []

    try:
        if is_zip_file(file_path):
            print(f"📦 检测到 ZIP 压缩包，正在自动解压...")
            temp_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            extracted_files = glob.glob(os.path.join(temp_dir, "*.nc"))
            if not extracted_files:
                print("❌ 压缩包里没找到 .nc 文件！")
                return

            print(f"📄 发现 {len(extracted_files)} 个分块文件，正在内存中合并...")

            try:
                for p in extracted_files:

                    try:
                        d = xr.open_dataset(p, engine="netcdf4")
                    except:

                        d = xr.open_dataset(p, engine="scipy")
                    datasets_list.append(d)


                ds = xr.merge(datasets_list)

            except Exception as e:
                print(f"❌ 合并文件失败: {e}")
                return

        else:

            ds = xr.open_dataset(file_path)


        print("🔄 正在转换数据格式...")
        df = ds.to_dataframe().reset_index()


        if 'valid_time' in df.columns:
            df.rename(columns={'valid_time': 'record_time'}, inplace=True)
        elif 'time' in df.columns:
            df.rename(columns={'time': 'record_time'}, inplace=True)

        df = df.dropna()

        if 't2m' in df.columns:
            df['temperature'] = df['t2m'] - 273.15
        elif '2m_temperature' in df.columns:
            df['temperature'] = df['2m_temperature'] - 273.15

        if 'tp' in df.columns:
            df['precipitation'] = df['tp'] * 1000
        elif 'total_precipitation' in df.columns:
            df['precipitation'] = df['total_precipitation'] * 1000

        # 保留两位小数
        if 'temperature' in df.columns: df['temperature'] = df['temperature'].round(2)
        if 'precipitation' in df.columns: df['precipitation'] = df['precipitation'].round(2)

        # 检查必要列
        required_cols = ['record_time', 'latitude', 'longitude', 'temperature', 'precipitation']
        if not all(col in df.columns for col in required_cols):
            print(f"❌ 缺少必要列！当前包含: {df.columns.tolist()}")
            return

        # 4. 准备入库
        data_to_insert = list(zip(
            df['record_time'], df['latitude'], df['longitude'],
            df['temperature'], df['precipitation'],
            df['longitude'], df['latitude']
        ))

        count = len(data_to_insert)
        if count == 0:
            print("⚠️ 无有效数据")
            return

        # 5. 写入数据库
        print(f"🚀 正在写入 {count} 条数据...")
        conn = get_db_conn()
        cur = conn.cursor()

        insert_sql = """
            INSERT INTO era5_data (record_time, latitude, longitude, temperature, precipitation, geom)
            VALUES (%s, %s, %s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326))
        """
        execute_batch(cur, insert_sql, data_to_insert, page_size=2000)
        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ 入库成功！")

    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理资源：先关闭合并后的 ds
        if ds: ds.close()
        # 再关闭列表里打开的单个文件句柄
        for d in datasets_list:
            d.close()

        ds = None
        datasets_list = []
        gc.collect()

        if temp_dir:
            force_delete_dir(temp_dir)


# ... (main 函数保持不变) ...
if __name__ == "__main__":
    # 记得保留原本的 main 函数逻辑
    nc_files = glob.glob(os.path.join(DATA_DIR, "*.nc"))
    if not nc_files:
        print("未找到 .nc 文件")
    else:
        print(f"🎯 发现 {len(nc_files)} 个文件，开始处理...")
        for f in nc_files:
            process_and_import(f)
        print("\n🎉🎉🎉 所有任务结束！")