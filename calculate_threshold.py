import psycopg2
import json
DB_CONF = {
    "host": "localhost",
    "database": "weather",
    "user": "postgres",
    "password": "postgres"
}


def calculate_percentiles():
    print("⏳ 正在利用数据库计算 10 年历史数据的统计阈值...")
    conn = psycopg2.connect(**DB_CONF)
    cur = conn.cursor()

    try:
        sql_temp = """
            SELECT 
                PERCENTILE_CONT(0.90) WITHIN GROUP(ORDER BY temperature) as p90,
                PERCENTILE_CONT(0.95) WITHIN GROUP(ORDER BY temperature) as p95,
                PERCENTILE_CONT(0.99) WITHIN GROUP(ORDER BY temperature) as p99,
                MAX(temperature) as max_val
            FROM era5_data;
        """
        cur.execute(sql_temp)
        temp_stats = cur.fetchone()

        sql_prcp = """
            SELECT 
                PERCENTILE_CONT(0.90) WITHIN GROUP(ORDER BY precipitation) as p90,
                PERCENTILE_CONT(0.95) WITHIN GROUP(ORDER BY precipitation) as p95,
                PERCENTILE_CONT(0.99) WITHIN GROUP(ORDER BY precipitation) as p99,
                MAX(precipitation) as max_val
            FROM era5_data
            WHERE precipitation > 0.1; 
        """
        cur.execute(sql_prcp)
        prcp_stats = cur.fetchone()

        # 3. 整理结果
        results = {
            "temperature": {
                "threshold_90": float(temp_stats[0]),
                "threshold_95": float(temp_stats[1]),
                "threshold_99": float(temp_stats[2]),
                "max_history": float(temp_stats[3])
            },
            "precipitation": {
                "threshold_90": float(prcp_stats[0]),
                "threshold_95": float(prcp_stats[1]),
                "threshold_99": float(prcp_stats[2]),
                "max_history": float(prcp_stats[3])
            }
        }

        print("\n📊 ========== 极端天气统计报告 ==========")
        print(f"🔥 温度 95% 阈值: {results['temperature']['threshold_95']} ℃")
        print(f"🔥 温度 99% 阈值: {results['temperature']['threshold_99']} ℃")
        print(f"🔥 历史最高温:    {results['temperature']['max_history']} ℃")
        print("-" * 30)
        print(f"🌧️ 降水 95% 阈值: {results['precipitation']['threshold_95']} mm")
        print(f"🌧️ 降水 99% 阈值: {results['precipitation']['threshold_99']} mm")
        print(f"🌧️ 历史最大雨:    {results['precipitation']['max_history']} mm")
        print("========================================\n")

        with open('threshold_config.json', 'w') as f:
            json.dump(results, f, indent=4)
        print("✅ 阈值配置文件已保存为 'threshold_config.json'")

    except Exception as e:
        print(f"❌ 计算失败: {e}")
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    calculate_percentiles()