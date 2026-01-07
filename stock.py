import streamlit as st
import google.generativeai as genai
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import json
import os
import requests
from datetime import datetime

# ---------------------------------------------------------
# [설정] 페이지 기본 설정
# ---------------------------------------------------------
st.set_page_config(page_title="Pro Insight Dashboard", layout="wide", page_icon="📈", initial_sidebar_state="collapsed")

# CSS: 사이드바 숨김 & 여백 조정
st.markdown("""
<style>
    [data-testid="collapsedControl"] {display: none}
    section[data-testid="stSidebar"] {display: none}
    .main .block-container {padding-top: 2rem; padding-bottom: 2rem;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# [보안] API 키 설정 (Streamlit Cloud Secrets 연동)
# ---------------------------------------------------------
try:
    # 1. Streamlit Cloud의 Secrets에서 키를 가져옴
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    # 2. 로컬(내 컴퓨터)이나 키 설정이 안 된 경우 안내
    # (주의: 깃허브에 올릴 때는 절대 여기에 실제 키를 적지 마세요!)
    API_KEY = "SECRET_KEY_NOT_FOUND" 

# ---------------------------------------------------------
# [AI 모델 연결]
# ---------------------------------------------------------
try:
    genai.configure(api_key=API_KEY)
    
    # 모델 자동 탐색
    target_model = "gemini-pro"
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            if 'gemini' in m.name:
                target_model = m.name
                break
    model = genai.GenerativeModel(target_model)

except Exception as e:
    # API 키가 없거나 틀렸을 때 에러 처리
    if API_KEY == "SECRET_KEY_NOT_FOUND":
        st.error("⚠️ API 키를 찾을 수 없습니다.")
        st.warning("Streamlit Cloud의 [Settings] -> [Secrets]에 'GEMINI_API_KEY'를 등록해주세요.")
        st.stop() # 중단
    else:
        st.error(f"API 연결 오류: {e}")

# ---------------------------------------------------------
# [데이터 파일 설정] - 여기가 누락되어 에러가 났던 부분입니다!
# ---------------------------------------------------------
DATA_FILE = "my_portfolio.json"

# ---------------------------------------------------------
# [함수] 데이터 로직
# ---------------------------------------------------------
def load_portfolio():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_portfolio(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 앱 시작 시 데이터 로드
if 'portfolio_db' not in st.session_state:
    st.session_state['portfolio_db'] = load_portfolio()

@st.cache_data(ttl=600)
def get_market_indices():
    tickers = {
        "USD/KRW": "KRW=X", "US 10Y": "^TNX", 
        "VIX (Fear)": "^VIX", "KOSPI": "^KS11", "NASDAQ": "^IXIC"
    }
    data = {}
    for name, ticker in tickers.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            current = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2]
            change = ((current - prev) / prev) * 100
            data[name] = (current, change)
        except:
            data[name] = (0, 0)
    return data

@st.cache_data(ttl=900)
def get_fear_and_greed_index():
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=5)
        r.raise_for_status()
        data = r.json()
        return data['fear_and_greed']['score'], data['fear_and_greed']['rating']
    except:
        return None, "N/A"

@st.cache_data(ttl=600)
def get_stock_details(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info 
        hist = stock.history(period="1mo")
        if hist.empty: return None
        current = hist['Close'].iloc[-1]
        
        delta = hist['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        return {
            "current": current,
            "rsi": rsi,
            "per": info.get('trailingPE', 0),
            "pbr": info.get('priceToBook', 0),
            "div_yield": info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        }
    except:
        return None

@st.cache_data(ttl=3600)
def get_sector_history():
    sectors = {"XLK(테크)":"XLK", "SOXX(반도체)":"SOXX", "XLF(금융)":"XLF", "XLV(헬스)":"XLV", "XLE(에너지)":"XLE"}
    try:
        df = yf.download(list(sectors.values()), period="1y", progress=False)['Close']
        return df, sectors
    except:
        return pd.DataFrame(), sectors

def calculate_sector_change(df, period_str):
    periods = {"1일": 2, "1주": 5, "1달": 21, "1분기": 63, "반년": 126, "1년": 252}
    days = periods.get(period_str, 2)
    changes = {}
    if df.empty: return {}
    for ticker in df.columns:
        try:
            if len(df) < days: start = df[ticker].iloc[0]
            else: start = df[ticker].iloc[-days]
            curr = df[ticker].iloc[-1]
            changes[ticker] = ((curr - start) / start) * 100
        except: changes[ticker] = 0.0
    return changes

def add_stock(account, ticker, price, qty):
    db = st.session_state['portfolio_db']
    if account not in db: db[account] = {}
    ticker = ticker.upper()
    if ticker in db[account]:
        old_qty = db[account][ticker]['qty']
        old_price = db[account][ticker]['avg_price']
        new_total_qty = old_qty + qty
        new_avg_price = ((old_price * old_qty) + (price * qty)) / new_total_qty
        db[account][ticker]['qty'] = new_total_qty
        db[account][ticker]['avg_price'] = new_avg_price
    else:
        db[account][ticker] = {'avg_price': price, 'qty': qty}
    save_portfolio(db)

def draw_gauge_chart(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number", value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Fear & Greed Index"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 25], 'color': '#FF4B4B'},
                {'range': [25, 45], 'color': '#FF8E8E'},
                {'range': [45, 55], 'color': '#E8E8E8'},
                {'range': [55, 75], 'color': '#90EE90'},
                {'range': [75, 100], 'color': '#008000'}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': score}
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
    return fig

# ---------------------------------------------------------
# [자동화된 AI 분석 함수]
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def get_ai_market_briefing(f_score):
    if API_KEY == "SECRET_KEY_NOT_FOUND": return "API 키가 설정되지 않아 분석할 수 없습니다."
    today_str = datetime.now().strftime("%Y-%m-%d")
    prompt = f"오늘은 {today_str}. 공포지수 {f_score}. 버핏지수 추정 및 투자 조언 3줄 요약."
    try: return model.generate_content(prompt).text
    except Exception as e: return f"분석 실패: {e}"

@st.cache_data(ttl=43200)
def get_ai_calendar_data():
    if API_KEY == "SECRET_KEY_NOT_FOUND": return []
    today_str = datetime.now().strftime("%Y-%m-%d")
    prompt = f"오늘 {today_str}. 향후 2주 미국 경제지표(CPI,PPI,고용), FOMC, 빅테크 실적 JSON 포맷으로: [{{'date':'MM-DD (요일)', 'event':'이름', 'importance':'⭐⭐⭐'}}]"
    try:
        res = model.generate_content(prompt)
        clean_json = res.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except: return []

# =========================================================
# [UI 구성]
# =========================================================

# 1. Market Index
st.markdown("### 🌍 Global Market & VIX")
market = get_market_indices()
m_cols = st.columns(5)
for i, (k, v) in enumerate(market.items()):
    m_cols[i].metric(k, f"{v[0]:,.2f}", f"{v[1]:.2f}%")

st.divider()

# 2. Sector Chart
st.title("💰 Smart Asset Dashboard")
sector_df, sector_map = get_sector_history()
inv_sector_map = {v: k for k, v in sector_map.items()}

c1, c2 = st.columns([1, 6])
with c1:
    st.write("⏱️ **기간 선택**")
    sel_period = st.radio("기간", ["1일", "1주", "1달", "1분기", "반년", "1년"], label_visibility="collapsed")
with c2:
    if not sector_df.empty:
        changes = calculate_sector_change(sector_df, sel_period)
        df_chart = pd.DataFrame(list(changes.items()), columns=['Ticker', 'Change'])
        df_chart['Name'] = df_chart['Ticker'].map(inv_sector_map)
        df_chart['Color'] = df_chart['Change'].apply(lambda x: '#ff4b4b' if x > 0 else '#4b88ff')
        fig = go.Figure(go.Bar(x=df_chart['Name'], y=df_chart['Change'], marker_color=df_chart['Color'], text=df_chart['Change'].apply(lambda x: f"{x:.2f}%"), textposition='auto'))
        fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="등락률(%)")
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# 3. Sentiment & Calendar
st.subheader("📅 Market Sentiment & Calendar")
col_cal_left, col_cal_right = st.columns([1, 1])

with col_cal_left:
    st.markdown("##### 😨 Fear & Greed Index")
    f_score, f_rating = get_fear_and_greed_index()
    if f_score is not None:
        st.plotly_chart(draw_gauge_chart(f_score), use_container_width=True)
        st.caption(f"현재 상태: **{f_rating.upper()} ({int(f_score)})**")
        st.markdown("---")
        st.markdown("##### 🧠 AI Market Insight")
        with st.spinner("Analyzing..."):
            st.info(get_ai_market_briefing(f_score))
    else:
        st.error("지수 로딩 실패")

with col_cal_right:
    st.markdown("##### 🗓️ 주요 경제 일정 (2주)")
    with st.spinner("Loading Calendar..."):
        cal_data = get_ai_calendar_data()
    if cal_data:
        st.dataframe(pd.DataFrame(cal_data), column_config={"date":"날짜","event":"이벤트","importance":"중요도"}, hide_index=True, use_container_width=True)
    else:
        st.warning("일정 데이터 없음 (API 키 확인 필요)")

st.divider()

# 4. Portfolio
st.subheader("📂 My Portfolio")
with st.expander("➕ 자산 추가 / 계좌 관리", expanded=False):
    db = st.session_state['portfolio_db']
    accounts = list(db.keys())
    t1, t2 = st.tabs(["매수 입력", "계좌 생성"])
    with t1:
        if accounts:
            c_acc, c_tick, c_qty, c_price, c_btn = st.columns([2, 2, 1, 2, 1])
            sel_acc = c_acc.selectbox("계좌", accounts)
            t_in = c_tick.text_input("티커").upper()
            q_in = c_qty.number_input("수량", 1)
            p_in = c_price.number_input("단가", 0.0)
            if c_btn.button("추가"):
                if t_in and p_in > 0:
                    add_stock(sel_acc, t_in, p_in, q_in)
                    st.rerun()
    with t2:
        nc1, nc2 = st.columns([3, 1])
        new_n = nc1.text_input("계좌명")
        if nc2.button("생성") and new_n:
            db[new_n] = {}
            save_portfolio(db)
            st.rerun()

total_ai_data = []
if db:
    for acc_name, stocks in db.items():
        if not stocks: continue
        st.markdown(f"**📌 {acc_name}**")
        rows = []
        for t, info in stocks.items():
            cur = get_stock_details(t)
            if cur:
                cp = cur['current']
                profit = ((cp - info['avg_price']) / info['avg_price']) * 100
                rows.append({
                    "종목": t, "수량": info['qty'], "평단": f"{info['avg_price']:.2f}", "현재": f"{cp:.2f}",
                    "수익률": f"{profit:.2f}%", "RSI": f"{cur['rsi']:.1f}",
                    "PER": f"{cur['per']:.1f}", "PBR": f"{cur['pbr']:.1f}", "배당률": f"{cur['div_yield']:.1f}%"
                })
                total_ai_data.append(f"[{acc_name}] {t}: 수익 {profit:.1f}%, PER {cur['per']:.1f}, PBR {cur['pbr']:.1f}")
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            col_del, _ = st.columns([2, 5])
            del_t = col_del.selectbox(f"삭제 ({acc_name})", ["선택안함"]+list(stocks.keys()), key=acc_name)
            if del_t != "선택안함" and col_del.button("🗑 삭제", key=f"btn_{acc_name}"):
                del db[acc_name][del_t]
                save_portfolio(db)
                st.rerun()

st.write("")
if st.button("🤖 가치투자 포트폴리오 진단 (AI)", use_container_width=True):
    if API_KEY == "SECRET_KEY_NOT_FOUND":
        st.error("API 키가 없습니다. Settings -> Secrets를 설정해주세요.")
    elif not total_ai_data:
        st.warning("자산 없음")
    else:
        st.write("🔍 Gemini 분석 중...")
        prompt = f"[시장] {market}\n[공포지수] {f_score}\n[자산] {total_ai_data}\n가치투자 관점에서 내 포트폴리오를 평가하고 전략을 제안해줘."
        try:
            res_box = st.empty()
            response = model.generate_content(prompt, stream=True)
            txt = ""
            for chunk in response:
                txt += chunk.text
                res_box.markdown(txt + "▌")
            res_box.markdown(txt)
        except Exception as e: st.error(f"오류: {e}")
