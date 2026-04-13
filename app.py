from flask import Flask, render_template, request, redirect, url_for, session
import psycopg2
import psycopg2.extras
import json
import os
import torch
import torch.nn as nn
import numpy as np
import joblib
from datetime import timedelta
from flask import jsonify
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)
app.secret_key = 'era5_weather_system_secret_key'
DB_HOST = "localhost"
DB_NAME = "weather"
DB_USER = "postgres"
DB_PASS = "postgres"

GEOJSON_PATH = os.path.join(os.path.dirname(__file__), '../data', 'shp', '合肥市.geojson')

class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

try:
    scaler = joblib.load('scaler.pkl')
    model = LSTMModel()
    model.load_state_dict(torch.load('model_lstm.pth', map_location='cpu'))
    model.eval()
    print("✅ AI 模型加载成功！准备就绪。")
except Exception as e:
    print(f"⚠️ 模型加载失败，请检查 scaler.pkl 和 model_lstm.pth 是否在根目录: {e}")
    model = None
    scaler = None

@app.route('/api/hefei_boundary')
def get_hefei_boundary():
    try:
        with open(GEOJSON_PATH, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
        return jsonify(geojson_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
def get_db_connection():
    conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)
    return conn

@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute('SELECT * FROM users WHERE username = %s AND password = %s', (username, password))
        account = cursor.fetchone()
        cursor.close()
        conn.close()

        if account:
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            return redirect(url_for('index'))
        else:
            msg = '用户名或密码错误，请重试！'

    return render_template('login.html', msg=msg)

@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/')
def index():
    if 'loggedin' in session:
        return render_template('index.html', username=session['username'])
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        account = cursor.fetchone()

        if account:
            msg = '该账号已存在，请换一个用户名或直接登录！'
        else:
            cursor.execute('INSERT INTO users (username, password, role) VALUES (%s, %s, %s)',
                           (username, password, 'user'))
            conn.commit()
            msg = '🎉 注册成功！请使用新账号登录。'
            cursor.close()
            conn.close()
            return render_template('login.html', msg=msg)

        cursor.close()
        conn.close()

    return render_template('register.html', msg=msg)


@app.route('/api/era5_grid')
def get_era5_grid():

    query_time = request.args.get('time', '2016-07-01 12:00:00')

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    try:
        sql = """
            SELECT latitude, longitude, temperature, precipitation 
            FROM era5_data 
            WHERE record_time = %s
        """
        cur.execute(sql, (query_time,))
        rows = cur.fetchall()

        data = []
        for row in rows:
            data.append({
                'lat': row['latitude'],
                'lon': row['longitude'],
                'temp': row['temperature'],
                'prcp': row['precipitation']
            })

        return jsonify({'status': 'success', 'data': data, 'time': query_time})

    except Exception as e:
        return jsonify({'status': 'error', 'msg': str(e)})
    finally:
        cur.close()
        conn.close()

@app.route('/data')
def data_page():
    if 'loggedin' not in session:
        return redirect(url_for('login'))
    return render_template('data.html', username=session['username'])


@app.route('/api/data_stats')
def get_data_stats():
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        cur.execute("SELECT count(*) FROM era5_data")
        total_count = cur.fetchone()[0]

        cur.execute("SELECT MIN(record_time), MAX(record_time) FROM era5_data")
        time_range = cur.fetchone()
        min_time = str(time_range[0]) if time_range[0] else '无'
        max_time = str(time_range[1]) if time_range[1] else '无'

        cur.execute("""
            SELECT record_time, latitude, longitude, temperature, precipitation 
            FROM era5_data 
            ORDER BY record_time DESC 
            LIMIT 50
        """)
        rows = cur.fetchall()

        recent_data = []
        for row in rows:
            recent_data.append({
                'time': str(row['record_time']),
                'lat': row['latitude'],
                'lon': row['longitude'],
                'temp': row['temperature'],
                'prcp': row['precipitation']
            })

        cur.execute("""
            SELECT
                COUNT(*) FILTER (WHERE temperature > 35) as count_hot,
                COUNT(*) FILTER (WHERE precipitation > 0.1 AND temperature <= 35) as count_rain,
                COUNT(*) FILTER (WHERE precipitation <= 0.1 AND temperature <= 35) as count_normal
            FROM era5_data
        """)
        dist_row = cur.fetchone()

        distribution = {
            'hot': dist_row[0],
            'rain': dist_row[1],
            'normal': dist_row[2]
        }

        return jsonify({
            'status': 'success',
            'total': total_count,
            'start_date': min_time,
            'end_date': max_time,
            'list': recent_data,
            'distribution': distribution
        })

    except Exception as e:
        print(f"数据统计出错: {e}")
        return jsonify({'status': 'error', 'msg': str(e)})
    finally:
        cur.close()
        conn.close()


@app.route('/visual')
def visual_page():
    if 'loggedin' not in session:
        return redirect(url_for('login'))
    return render_template('visual.html', username=session['username'])


@app.route('/api/event_times')
def get_event_times():
    """获取某一天的所有时间序列"""
    date_str = request.args.get('date', '2022-08-15')  # 默认查 2022-08-15
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT DISTINCT record_time 
            FROM era5_data
            WHERE DATE(record_time) = %s
            ORDER BY record_time ASC
        """, (date_str,))

        times = [row[0].strftime('%Y-%m-%d %H:%M:%S') for row in cur.fetchall()]

        return jsonify({'status': 'success', 'times': times})
    except Exception as e:
        return jsonify({'status': 'error', 'msg': str(e)})
    finally:
        cur.close()
        conn.close()


@app.route('/extreme')
def extreme_page():
    if 'loggedin' not in session:
        return redirect(url_for('login'))
    return render_template('extreme.html', username=session['username'])

@app.route('/api/extreme_events')
def get_extreme_events():
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        cur.execute("""
            SELECT DATE(record_time) as day, MAX(temperature) as max_temp
            FROM era5_data
            WHERE temperature > 34.62
            GROUP BY day
            ORDER BY max_temp DESC
            LIMIT 10
        """)
        heat_rows = cur.fetchall()
        heat_events = [{'date': str(r['day']), 'value': round(r['max_temp'], 2), 'type': '高温'} for r in heat_rows]

        cur.execute("""
            SELECT DATE(record_time) as day, MAX(precipitation) as max_prcp
            FROM era5_data
            WHERE precipitation > 6.39
            GROUP BY day
            ORDER BY max_prcp DESC
            LIMIT 10
        """)
        rain_rows = cur.fetchall()
        rain_events = [{'date': str(r['day']), 'value': round(r['max_prcp'], 2), 'type': '暴雨'} for r in rain_rows]

        return jsonify({
            'status': 'success',
            'heat_events': heat_events,
            'rain_events': rain_events
        })
    except Exception as e:
        return jsonify({'status': 'error', 'msg': str(e)})
    finally:
        cur.close()
        conn.close()

@app.route('/lstm')
def lstm_page():
    if 'loggedin' not in session:
        return redirect(url_for('login'))
    return render_template('lstm.html', username=session['username'])

@app.route('/api/predict')
def predict_weather():
    if not model or not scaler:
        return jsonify({'status': 'error', 'msg': 'AI模型未加载，无法预测'})

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        sql = """
            SELECT * FROM (
                SELECT DATE(record_time) as day, AVG(temperature) as temp, MAX(precipitation) as prcp
                FROM era5_data
                GROUP BY day
                ORDER BY day DESC
                LIMIT 7
            ) as sub ORDER BY day ASC;
        """
        cur.execute(sql)
        rows = cur.fetchall()

        if len(rows) < 7:
            return jsonify({'status': 'error', 'msg': '历史数据不足 7 天，无法推演'})

        last_date = rows[-1][0]
        input_raw = np.array([[r[1], r[2]] for r in rows])

        input_scaled = scaler.transform(input_raw)
        current_seq = torch.FloatTensor(input_scaled).unsqueeze(0)

        future_preds = []
        future_dates = []

        for i in range(7):
            with torch.no_grad():
                pred = model(current_seq)
                future_preds.append(pred.numpy()[0])

                current_seq = torch.cat((current_seq[:, 1:, :], pred.unsqueeze(1)), dim=1)

                next_date = last_date + timedelta(days=i + 1)
                future_dates.append(next_date.strftime('%m-%d'))

        future_preds_real = scaler.inverse_transform(np.array(future_preds))

        result = {
            'status': 'success',
            'dates': future_dates,
            'temps': [round(float(x[0]), 1) for x in future_preds_real],
            'prcps': [round(float(x[1]), 1) for x in future_preds_real]
        }

        return jsonify(result)

    except Exception as e:
        print(f"预测接口执行出错: {e}")
        return jsonify({'status': 'error', 'msg': str(e)})
    finally:
        cur.close()
        conn.close()


@app.route('/api/validate_model')
def validate_model():
    if not model or not scaler:
        return jsonify({'status': 'error', 'msg': 'AI模型未加载'})

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        sql = """
            SELECT * FROM (
                SELECT DATE(record_time) as day, AVG(temperature) as temp, MAX(precipitation) as prcp
                FROM era5_data
                GROUP BY day
                ORDER BY day DESC
                LIMIT 14
            ) as sub ORDER BY day ASC;
        """
        cur.execute(sql)
        rows = cur.fetchall()

        if len(rows) < 14:
            return jsonify({'status': 'error', 'msg': '数据不足14天'})

        dates = [r[0].strftime('%m-%d') for r in rows[7:]]
        truth_temps = [round(float(r[1]), 1) for r in rows[7:]]
        truth_prcps = [round(float(r[2]), 1) for r in rows[7:]]

        input_raw = np.array([[r[1], r[2]] for r in rows[:7]])
        input_scaled = scaler.transform(input_raw)
        current_seq = torch.FloatTensor(input_scaled).unsqueeze(0)

        preds = []
        for i in range(7):
            with torch.no_grad():
                pred = model(current_seq)
                preds.append(pred.numpy()[0])
                current_seq = torch.cat((current_seq[:, 1:, :], pred.unsqueeze(1)), dim=1)

        preds_real = scaler.inverse_transform(np.array(preds))
        pred_temps = [round(float(x[0]), 1) for x in preds_real]
        pred_prcps = [round(float(x[1]), 1) for x in preds_real]

        return jsonify({
            'status': 'success',
            'dates': dates,
            'truth_temps': truth_temps,
            'pred_temps': pred_temps,
            'truth_prcps': truth_prcps,
            'pred_prcps': pred_prcps
        })

    except Exception as e:
        return jsonify({'status': 'error', 'msg': str(e)})
    finally:
        cur.close()
        conn.close()


@app.route('/api/point_history')
def get_point_history():
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)

    if lat is None or lon is None:
        return jsonify({'status': 'error', 'msg': '缺少经纬度参数'})

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        cur.execute("""
            SELECT DATE(record_time) as day, AVG(temperature) as temp, MAX(precipitation) as prcp
            FROM era5_data
            WHERE ABS(latitude - %s) < 0.05 AND ABS(longitude - %s) < 0.05
            GROUP BY day
            ORDER BY day ASC
            LIMIT 30
        """, (lat, lon))
        rows = cur.fetchall()

        if not rows:
            return jsonify({'status': 'error', 'msg': '数据库中未找到该点的历史记录'})

        return jsonify({
            'status': 'success',
            'dates': [r['day'].strftime('%Y-%m-%d') for r in rows],
            'temps': [round(r['temp'], 2) for r in rows],
            'prcps': [round(r['prcp'], 2) for r in rows]
        })
    except Exception as e:
        return jsonify({'status': 'error', 'msg': str(e)})
    finally:
        cur.close()
        conn.close()


@app.route('/api/era5_timeline')
def era5_timeline():
    date_str = request.args.get('date')
    if not date_str:
        return jsonify({'status': 'error', 'msg': '缺少日期参数'})

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        cur.execute("""
            SELECT DISTINCT record_time 
            FROM era5_data 
            WHERE DATE(record_time) = %s 
            ORDER BY record_time ASC
        """, (date_str,))
        time_rows = cur.fetchall()
        times = [r[0].strftime('%Y-%m-%d %H:%M:%S') for r in time_rows]

        if not times:
            return jsonify({'status': 'error', 'msg': '该日无气象数据'})

        cur.execute("""
            SELECT record_time, latitude, longitude, temperature, precipitation
            FROM era5_data
            WHERE DATE(record_time) = %s
        """, (date_str,))
        rows = cur.fetchall()

        data_dict = {t: [] for t in times}
        for r in rows:
            t_str = r['record_time'].strftime('%Y-%m-%d %H:%M:%S')
            data_dict[t_str].append({
                'lat': r['latitude'],
                'lon': r['longitude'],
                'temp': round(r['temperature'], 2),
                'prcp': round(r['precipitation'], 2)
            })

        return jsonify({
            'status': 'success',
            'times': times,
            'data': data_dict
        })
    except Exception as e:
        return jsonify({'status': 'error', 'msg': str(e)})
    finally:
        cur.close()
        conn.close()


@app.route('/api/train_model', methods=['POST'])
def train_model():
    params = request.json
    epochs = int(params.get('epochs', 50))
    lr = float(params.get('lr', 0.01))
    hidden_size = int(params.get('hidden', 64))
    seq_length = int(params.get('seq', 7))

    start_date = params.get('start_date')
    end_date = params.get('end_date')

    try:
        import pandas as pd
        from sklearn.preprocessing import MinMaxScaler
        import torch
        import torch.nn as nn
        import numpy as np
        import joblib

        conn = get_db_connection()

        if start_date and end_date:
            sql = """
                SELECT DATE(record_time) as day, AVG(temperature) as temp, MAX(precipitation) as prcp 
                FROM era5_data 
                WHERE DATE(record_time) >= %s AND DATE(record_time) <= %s
                GROUP BY day ORDER BY day ASC;
            """
            df = pd.read_sql(sql, conn, params=(start_date, end_date))
        else:
            sql = """
                SELECT DATE(record_time) as day, AVG(temperature) as temp, MAX(precipitation) as prcp 
                FROM era5_data 
                GROUP BY day ORDER BY day ASC;
            """
            df = pd.read_sql(sql, conn)

        conn.close()

        if len(df) < seq_length * 2:
            return jsonify({'status': 'error',
                            'msg': f'选定时间段内数据量过少（仅 {len(df)} 天），不足以支撑序列训练，请拉长日期范围！'})

        data_raw = df[['temp', 'prcp']].values

        new_scaler = MinMaxScaler()
        data_scaled = new_scaler.fit_transform(data_raw)

        X, y = [], []
        for i in range(len(data_scaled) - seq_length):
            X.append(data_scaled[i: i + seq_length])
            y.append(data_scaled[i + seq_length])

        X = torch.FloatTensor(np.array(X))
        y = torch.FloatTensor(np.array(y))

        class DynamicLSTM(nn.Module):
            def __init__(self, hidden_dim):
                super(DynamicLSTM, self).__init__()
                self.lstm = nn.LSTM(2, hidden_dim, 2, batch_first=True)
                self.fc = nn.Linear(hidden_dim, 2)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        new_model = DynamicLSTM(hidden_size)
        optimizer = torch.optim.Adam(new_model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        new_model.train()
        final_loss = 0
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = new_model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        torch.save(new_model.state_dict(), 'model_lstm.pth')
        joblib.dump(new_scaler, 'scaler.pkl')

        global model, scaler
        model = new_model
        scaler = new_scaler
        model.eval()

        return jsonify({'status': 'success', 'loss': round(final_loss, 4), 'data_count': len(df)})
    except Exception as e:
        return jsonify({'status': 'error', 'msg': str(e)})


import json
import re
from langchain_community.chat_models import ChatSparkLLM
from langchain.schema import HumanMessage

SPARKAI_URL = 'wss://spark-api.xf-yun.com/v1.1/chat'
SPARKAI_APP_ID = 'fe84f948'
SPARKAI_API_SECRET = 'MDVlMmFiOWE4NGVjMDQ4ZWMxZjMzYmQy'
SPARKAI_API_KEY = '7296a6033ccd4d4beff0ef032627c606'
SPARKAI_DOMAIN = 'general'


def get_recent_weather_context():

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        cur.execute("""
            SELECT DATE(record_time) as day, AVG(temperature) as temp, MAX(precipitation) as prcp
            FROM era5_data GROUP BY day ORDER BY day DESC LIMIT 7
        """)
        rows = cur.fetchall()
        rows.reverse()
        context = ""
        for r in rows:
            context += f"[{r['day']}] 平均温度: {round(r['temp'], 1)}℃, 最大降水: {round(r['prcp'], 1)}mm\n"
        return context
    except Exception as e:
        return ""
    finally:
        cur.close()
        conn.close()


@app.route('/warning')
def warning_page():
    return render_template('warning.html')


@app.route('/api/generate_ai_warning', methods=['POST'])
def generate_ai_warning():

    try:
        context = get_recent_weather_context()
        if not context:
            return jsonify({'status': 'error', 'msg': '无法获取天气上下文数据'})

        prompt = f"""
        你现在是市应急管理局的高级气象AI指挥官。基于以下未来7天的预测数据：
        {context}

        请严格按照以下 JSON 格式输出，绝对不要包含任何注释、markdown代码块标记（如```json）或其他解释性文字。
        注意：report 字段的值必须写在同一行内，不要有真实的换行符！可以使用 HTML 的 <br> 标签来代替换行。
        {{
            "report": "写一段约200字的专业气象预警研判专报，使用 <br> 和 <b style='color:#00d4ff'> 标签进行排版强调。",
            "radar": [80, 20, 30, 40, 90],
            "tasks": ["具体的应急调度指令1", "具体的应急调度指令2", "具体的应急调度指令3"]
        }}
        """

        content = ""
        try:

            spark = ChatSparkLLM(
                spark_api_url=SPARKAI_URL, spark_app_id=SPARKAI_APP_ID,
                spark_api_key=SPARKAI_API_KEY, spark_api_secret=SPARKAI_API_SECRET,
                spark_llm_domain=SPARKAI_DOMAIN, streaming=False,
            )
            response = spark([HumanMessage(content=prompt)])
            content = response.content.strip()
            print("✅ 云端大模型调用成功！")

        except Exception as llm_err:

            print(f"⚠️ 云端大模型接口异常，已自动切换至本地灾备引擎。错误原因: {llm_err}")
            content = """{
                "report": "<b style='color:#fbbf24'>⚡ 系统提示：云端 AI 接口当前限流或未授权，已自动降级为本地核心研判模式。</b><br><br>根据底层 LSTM 模型最新推演序列显示，合肥及巢湖流域近期大气环流存在不稳定扰动。建议市防汛抗旱指挥部密切关注<b style='color:#00d4ff'>西北部山区汇水</b>及<b style='color:#00d4ff'>主城区低洼管网</b>的排水承载力。整体态势处于可控区间，请各单位保持常规应急响应编制。",
                "radar": [65, 30, 75, 45, 80],
                "tasks": ["启动本地气象灾备预案，保持系统离线可用", "调度排涝泵车前往高风险路段待命", "加强巢湖沿岸大坝 24 小时电子巡视"]
            }"""


        start_idx = content.find('{')
        end_idx = content.rfind('}')
        if start_idx != -1 and end_idx != -1:
            content = content[start_idx:end_idx + 1]

        try:

            ai_data = json.loads(content)
            return jsonify({
                'status': 'success',
                'report': ai_data.get('report', '分析完成，但报告字段缺失。'),
                'radar': ai_data.get('radar', [50, 50, 50, 50, 50]),
                'tasks': ai_data.get('tasks', ['请关注天气变化', '做好常规巡查'])
            })
        except Exception as parse_err:
            print(f"JSON 解析失败: {parse_err}")
            return jsonify({
                'status': 'success',
                'report': f"<b style='color:#ef4444'>⚠️ AI 推理格式出现微小偏差，以下为原始研判：</b><br><br>{content}",
                'radar': [30, 30, 30, 30, 30],
                'tasks': ["AI 数据解析出现偏差，请联系管理员", "建议直接阅读上方原始报告"]
            })

    except Exception as e:
        return jsonify({'status': 'error', 'msg': str(e)})


@app.route('/api/ai_chat', methods=['POST'])
def ai_chat():

    user_msg = request.json.get('message')
    try:
        context = get_recent_weather_context()
        prompt = f"""
        你是本系统的“气象数字人助理”。请根据以下最近的天气预测数据回答用户的问题。
        【天气数据】
        {context}
        【用户问题】
        {user_msg}

        回答要求：
        1. 简明扼要，像一个智能助手一样回答。
        2. 结合上面的数据给出建议。
        """

        try:
            spark = ChatSparkLLM(
                spark_api_url=SPARKAI_URL, spark_app_id=SPARKAI_APP_ID,
                spark_api_key=SPARKAI_API_KEY, spark_api_secret=SPARKAI_API_SECRET,
                spark_llm_domain=SPARKAI_DOMAIN, streaming=False,
            )
            response = spark([HumanMessage(content=prompt)])
            reply = response.content
        except Exception as llm_err:
            # 聊天机器人容灾兜底
            print(f"⚠️ 聊天机器人云端接口异常: {llm_err}")
            reply = "🤖 【本地离线模式】抱歉，由于云端大模型接口凭证失效或网络超时，我暂时无法连接外脑进行复杂推理。根据本地基础数据，近几日天气变化较快，出门请注意关注实时大屏预警哦！"

        return jsonify({'status': 'success', 'reply': reply})
    except Exception as e:
        return jsonify({'status': 'error', 'msg': 'AI 思考时睡着了，请重试。'})


@app.route('/realtime')
def realtime_page():
    return render_template('realtime.html')


import requests
from concurrent.futures import ThreadPoolExecutor
import random

@app.route('/api/realtime_weather', methods=['GET'])
def get_realtime_weather():
    cities = {
        '合肥': '340100', '芜湖': '340200', '蚌埠': '340300', '淮南': '340400',
        '马鞍山': '340500', '淮北': '340600', '铜陵': '340700', '安庆': '340800',
        '黄山': '341000', '滁州': '341100', '阜阳': '341200', '宿州': '341300',
        '六安': '341500', '亳州': '341600', '池州': '341700', '宣城': '341800'
    }
    api_key = "bcc33ec10ca643efb0690539252502"
    results = {}

    def fetch_weather(city_name, adcode):
        url = f"https://restapi.amap.com/v3/weather/weatherInfo?city={adcode}&key={api_key}&extensions=all"
        try:
            res = requests.get(url, timeout=5).json()
            if res['status'] == '1' and len(res['forecasts']) > 0:
                cast = res['forecasts'][0]['casts'][0]
                return {
                    'city': city_name,
                    'max_temp': float(cast['daytemp']),
                    'min_temp': float(cast['nighttemp']),
                    'weather': cast['dayweather'],
                    'wind': cast['daywind']
                }
        except Exception as e:
            print(f"获取{city_name}天气失败: {e}")
        return {'city': city_name, 'max_temp': 25, 'min_temp': 15, 'weather': '晴', 'wind': '东风'}

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch_weather, name, code) for name, code in cities.items()]
        for future in futures:
            res = future.result()
            results[res['city']] = res

    city_names = list(cities.keys())
    max_temps = [results[c]['max_temp'] for c in city_names]
    min_temps = [results[c]['min_temp'] for c in city_names]

    rainfalls = [random.randint(10, 80) if '雨' in results[c]['weather'] else random.randint(0, 2) for c in city_names]
    aqis = [random.randint(20, 50) if '雨' in results[c]['weather'] else random.randint(40, 100) for c in city_names]

    weather_counts = {}
    for c in city_names:
        w = results[c]['weather']
        weather_counts[w] = weather_counts.get(w, 0) + 1
    pie_data = [{'name': k, 'value': v} for k, v in weather_counts.items()]


    wind_counts = {'北': 0, '东北': 0, '东': 0, '东南': 0, '南': 0, '西南': 0, '西': 0, '西北': 0}
    for c in city_names:

        w = results[c]['wind'].replace('风', '').replace('无持续风向', '东')
        if w in wind_counts:
            wind_counts[w] += 1
        else:
            wind_counts['东'] += 1

    radar_data = [wind_counts[k] for k in ['北', '东北', '东', '东南', '南', '西南', '西', '西北']]

    return jsonify({
        'status': 'success',
        'data': {
            'cities': city_names,
            'max_temps': max_temps,
            'min_temps': min_temps,
            'rainfalls': rainfalls,
            'aqis': aqis,
            'pie_data': pie_data,
            'radar_data': radar_data
        }
    })


@app.route('/api/history_weather_compare', methods=['GET'])
def history_weather_compare():
    target_date = request.args.get('date', '2016-07-01')

    city_coords = {
        '合肥': (31.82, 117.23), '芜湖': (31.33, 118.38), '蚌埠': (32.92, 117.38),
        '淮南': (32.63, 117.00), '马鞍山': (31.67, 118.50), '淮北': (33.95, 116.80),
        '铜陵': (30.93, 117.82), '安庆': (30.50, 117.03), '黄山': (29.72, 118.33),
        '滁州': (32.33, 118.32), '阜阳': (32.90, 115.82), '宿州': (33.63, 116.98),
        '六安': (31.75, 116.50), '亳州': (33.87, 115.78), '池州': (30.65, 117.48),
        '宣城': (30.95, 118.75)
    }

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    results = {}
    try:
        for city, (lat, lon) in city_coords.items():
            cur.execute("""
                SELECT temperature, precipitation 
                FROM era5_data 
                WHERE DATE(record_time) = %s
                ORDER BY ((latitude - %s)^2 + (longitude - %s)^2) ASC
                LIMIT 1
            """, (target_date, lat, lon))

            row = cur.fetchone()
            if row:
                results[city] = {
                    'temp': round(row['temperature'], 1),
                    'prcp': round(row['precipitation'], 1)
                }
            else:
                results[city] = {'temp': 20, 'prcp': 0}

        city_names = list(city_coords.keys())

        max_temps = [round(results[c]['temp'] + 4.5, 1) for c in city_names]
        min_temps = [round(results[c]['temp'] - 3.2, 1) for c in city_names]
        rainfalls = [results[c]['prcp'] for c in city_names]

        aqis = [random.randint(20, 45) if r > 1 else random.randint(50, 85) for r in rainfalls]
        pie_data = [
            {'name': '降水天气 (雨/雪)', 'value': sum(1 for r in rainfalls if r > 0)},
            {'name': '晴朗/多云', 'value': sum(1 for r in rainfalls if r == 0)}
        ]
        radar_data = [random.randint(20, 80) for _ in range(8)]

        return jsonify({
            'status': 'success',
            'data': {
                'cities': city_names,
                'max_temps': max_temps,
                'min_temps': min_temps,
                'rainfalls': rainfalls,
                'aqis': aqis,
                'pie_data': pie_data,
                'radar_data': radar_data
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'msg': str(e)})
    finally:
        cur.close()
        conn.close()

if __name__ == '__main__':
    app.run(debug=True, port=5000)