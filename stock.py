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
# [보안] API 키 설정 (자동 감지 로직)
# ---------------------------------------------------------
# 1. API 키 가져오기 시도
try:
    # 배포 환경(Streamlit Cloud)에서는 여기서 키를 가져옵니다.
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    # 로컬 환경이거나 설정이 안 된 경우 (임시)
    # 주의: 깃허브에 올릴 때는 아래 곳에 절대 실제 키를 적지 마세요!
    API_KEY = "여기에_본인의_API_KEY를_넣으세요" 

# 2. Gemini 모델 연결 및 설정
try:
    genai.configure(api_key=API_KEY)
    
    # 사용 가능한 모델 자동 탐색
    target_model = "gemini-pro" # 기본값
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            if 'gemini' in m.name:
                target_model = m.name
                break
    
    model = genai.GenerativeModel(target_model)

except Exception as e:
    st.error(f"⚠️ API 연결 실패: {e}")
    st.error("Streamlit Cloud의 'Secrets' 설정에 API 키가 등록되었는지 확인해주세요.")
    st.stop() # 키가 없으면 더 이상 진행하지 않고 멈춤

# ---------------------------------------------------------
# [함수] 데이터 로직
# ---------------------------------------------------------
# (이 아래부터는 기존 코드와 동일합니다. DATA_FILE = ... 부터 시작)

def load_portfolio():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_portfolio(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

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
    """CNN Fear and Greed Index 가져오기"""
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        r = requests.get(url, headers=headers, timeout=5)
        r.raise_for_status()
        data = r.json()
        score = data['fear_and_greed']['score']
        rating = data['fear_and_greed']['rating']
        return score, rating
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
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Fear & Greed Index"},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "black"},
            'bgcolor': "white",
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
# [자동화된 AI 분석 함수] (버튼 제거용)
# ---------------------------------------------------------
@st.cache_data(ttl=3600) # 1시간마다 자동 갱신
def get_ai_market_briefing(f_score):
    today_str = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""
    오늘은 {today_str}입니다.
    현재 Fear & Greed Index 점수는 {f_score if f_score else '알수없음'}입니다.
    
    1. 현재 '버핏 지수(Buffett Indicator)' 상태를 추정하여 시장이 고평가인지 저평가인지 알려주세요.
    2. 현재 공포/탐욕 단계에 따른 투자자의 행동 요령을 3줄로 조언해주세요.
    """
    try:
        return model.generate_content(prompt).text
    except Exception as e:
        return f"분석 실패: {e}"

@st.cache_data(ttl=43200) # 12시간마다 자동 갱신
def get_ai_calendar_data():
    today_str = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""
    오늘은 {today_str}입니다. 향후 2주간 미국 주요 경제 지표(CPI, PPI, 고용), FOMC, 빅테크 실적 발표를 찾아줘.
    반드시 아래 JSON 포맷으로만 답변해. 설명 없이 JSON만 줘.
    [
        {{"date": "MM-DD (요일)", "event": "이벤트명", "importance": "⭐⭐⭐"}}
    ]
    """
    try:
        res = model.generate_content(prompt)
        clean_json = res.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except Exception as e:
        return []

# =========================================================
# [UI 구성]
# =========================================================

# 1. 🌍 Market Index
st.markdown("### 🌍 Global Market & VIX")
market = get_market_indices()
m_cols = st.columns(5)
for i, (k, v) in enumerate(market.items()):
    m_cols[i].metric(k, f"{v[0]:,.2f}", f"{v[1]:.2f}%")

st.divider()

# 2. 💰 섹터 차트
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
        
        fig = go.Figure(go.Bar(
            x=df_chart['Name'], y=df_chart['Change'], marker_color=df_chart['Color'],
            text=df_chart['Change'].apply(lambda x: f"{x:.2f}%"), textposition='auto'
        ))
        fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="등락률(%)")
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# 3. 📅 시장 심리(Fear&Greed) & 경제 일정
st.subheader("📅 Market Sentiment & Calendar")
col_cal_left, col_cal_right = st.columns([1, 1])

# [왼쪽] Fear & Greed Index + 버핏지수 브리핑
with col_cal_left:
    st.markdown("##### 😨 Fear & Greed Index (실시간)")
    
    # Fear & Greed 데이터 가져오기
    f_score, f_rating = get_fear_and_greed_index()
    
    if f_score is not None:
        st.plotly_chart(draw_gauge_chart(f_score), use_container_width=True)
        st.caption(f"현재 상태: **{f_rating.upper()} ({int(f_score)})**")
        
        st.markdown("---")
        st.markdown("##### 🧠 AI Market Insight")
        # 자동 분석 (캐싱됨)
        with st.spinner("AI가 시장 심리를 분석 중입니다..."):
            briefing = get_ai_market_briefing(f_score)
            st.info(briefing)
    else:
        st.error("지수 데이터를 가져오는데 실패했습니다.")

# [오른쪽] 경제 일정
with col_cal_right:
    st.markdown("##### 🗓️ 주요 경제 일정 (2주)")
    
    # 자동 일정 로드 (캐싱됨)
    with st.spinner("경제 일정을 불러오는 중..."):
        cal_data = get_ai_calendar_data()
    
    if cal_data:
        df_cal = pd.DataFrame(cal_data)
        st.dataframe(df_cal, column_config={"date":"날짜","event":"이벤트","importance":"중요도"}, hide_index=True, use_container_width=True)
    else:
        st.warning("일정 데이터를 불러오지 못했습니다.")

st.divider()

# 4. 📂 My Portfolio
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
    if not total_ai_data: st.warning("자산 없음")
    else:
        st.write("🔍 Gemini가 밸류에이션을 분석 중입니다...")
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