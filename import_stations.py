import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
import glob
import os

DB_CONF = {
    "host": "localhost",
    "database": "weather",
    "user": "postgres",
    "password": "postgres"
}
CSV_DIR = r"../data/csv"

def get_db_conn():
    return psycopg2.connect(**DB_CONF)


def process_and_import_csv(file_path):
    print(f"📂 正在处理文件: {os.path.basename(file_path)}")
    try:
        df = pd.read_csv(file_path)
        data_list = []

        for index, row in df.iterrows():
            temp_f = row['TEMP']
            if temp_f == 9999.9:
                temp_c = None
            else:
                temp_c = (temp_f - 32) * 5 / 9
                temp_c = round(temp_c, 2)


            prcp_in = row['PRCP']

            if prcp_in == 99.99:
                prcp_mm = None
            else:

                prcp_mm = prcp_in * 25.4
                prcp_mm = round(prcp_mm, 2)

            station_id = str(row['STATION'])

            data_list.append((
                station_id,
                row['DATE'],
                temp_c,
                prcp_mm
            ))

        # 3. 批量入库
        if not data_list:
            print("⚠️ 该文件没有有效数据")
            return

        conn = get_db_conn()
        cur = conn.cursor()

        print(f"🚀 正在写入 {len(data_list)} 条真实观测数据...")

        insert_sql = """
            INSERT INTO station_data (station_id, record_date, obs_temp, obs_prcp)
            VALUES (%s, %s, %s, %s)
        """

        execute_batch(cur, insert_sql, data_list, page_size=1000)

        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ 入库成功！")

    except Exception as e:
        print(f"❌ 处理失败: {e}")


if __name__ == "__main__":
    # 找到文件夹下所有 .csv 文件
    csv_files = glob.glob(os.path.join(CSV_DIR, "*.csv"))

    if not csv_files:
        print(f"在 {CSV_DIR} 下未找到 .csv 文件，请检查路径！")
    else:
        print(f"🎯 发现 {len(csv_files)} 个 CSV 文件，开始处理...")
        for f in csv_files:
            process_and_import_csv(f)
        print("\n🎉🎉🎉 所有站点数据入库完成！")