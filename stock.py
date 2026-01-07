import streamlit as st
import google.generativeai as genai
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import json
import os
import requests
import re
from datetime import datetime

# ---------------------------------------------------------
# [설정] 페이지 설정 & CSS
# ---------------------------------------------------------
st.set_page_config(page_title="My Asset Dashboard", layout="wide", page_icon="💸", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .main .block-container {max-width: 1200px; padding-top: 2rem; padding-bottom: 5rem;}
    
    /* 카드 디자인 */
    .metric-card {
        background-color: white; border: 1px solid #e0e0e0; border-radius: 15px;
        padding: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 15px;
    }
    .card-title {font-size: 14px; color: #666; margin-bottom: 5px;}
    .card-value {font-size: 24px; font-weight: bold; color: #333;}
    .card-sub {font-size: 14px; color: #888;}
    
    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    .stTabs [data-baseweb="tab"] {background-color: #f8f9fa; border-radius: 8px; padding: 10px 20px;}
    .stTabs [aria-selected="true"] {background-color: #eef2ff; color: #4c6ef5; font-weight: bold;}
    
    [data-testid="collapsedControl"] {display: none}
    section[data-testid="stSidebar"] {display: none}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# [보안 & 설정]
# ---------------------------------------------------------
try: API_KEY = st.secrets["GEMINI_API_KEY"]
except: API_KEY = "SECRET_KEY_NOT_FOUND"

USER_FILE = "users.json"

# ---------------------------------------------------------
# [함수] 데이터 로직
# ---------------------------------------------------------
def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r", encoding="utf-8") as f: return json.load(f)
    return {"admin": "1234"}

def save_user(u, p):
    d = load_users(); d[u] = p
    with open(USER_FILE, "w", encoding="utf-8") as f: json.dump(d, f, indent=4)

def get_portfolio_file():
    u = st.session_state.get('username', 'guest')
    return f"portfolio_{u}.json"

def load_portfolio():
    f = get_portfolio_file()
    if os.path.exists(f):
        with open(f, "r", encoding="utf-8") as file: return json.load(file)
    return {}

def save_portfolio(data):
    f = get_portfolio_file()
    with open(f, "w", encoding="utf-8") as file: json.dump(data, file, indent=4)

def detect_country(ticker):
    """티커로 국내/해외 구분 (KS/KQ는 국내, 나머지는 해외)"""
    if ".KS" in ticker or ".KQ" in ticker: return "KR"
    return "US"

@st.cache_data(ttl=600)
def get_market_data():
    """주요 지수 및 환율 가져오기"""
    tickers = {
        "🇺🇸 다우": "^DJI", "🇺🇸 나스닥": "^IXIC", "🇺🇸 S&P500": "^GSPC",
        "🇰🇷 코스피": "^KS11", "🇰🇷 코스닥": "^KQ11",
        "₿ 비트코인": "BTC-USD", "🥇 금": "GC=F", "🛢 WTI": "CL=F", "💵 환율": "KRW=X"
    }
    res = {}
    for k, t in tickers.items():
        try:
            h = yf.Ticker(t).history(period="5d")
            c = float(h['Close'].iloc[-1]); p = float(h['Close'].iloc[-2])
            res[k] = (c, ((c-p)/p)*100)
        except: res[k] = (0.0, 0.0)
    return res

@st.cache_data(ttl=900)
def get_fear_greed():
    """CNN 공포지수 (헤더 보강)"""
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        h = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"}
        r = requests.get(url, headers=h, timeout=5)
        d = r.json()
        return d['fear_and_greed']['score'], d['fear_and_greed']['rating']
    except: return None, "Error"

@st.cache_data(ttl=300)
def get_prices(tickers):
    if not tickers: return {}
    try:
        # yfinance download
        data = yf.download(tickers, period="1d", progress=False)['Close']
        if data.empty: return {}
        
        # 1개 종목일 때
        if len(tickers) == 1:
            return {tickers[0]: float(data.iloc[-1])}
        
        # 여러 종목일 때 (Series -> Dict)
        last_row = data.iloc[-1]
        result = {}
        for t in tickers:
            # MultiIndex 컬럼일 경우 처리
            try: val = float(last_row[t])
            except: val = 0.0
            result[t] = val
        return result
    except: return {}

# ---------------------------------------------------------
# [AI 함수]
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def get_ai_briefing(indices, f_score):
    if API_KEY == "SECRET_KEY_NOT_FOUND": return "API 키가 없습니다."
    today = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""
    오늘은 {today}.
    [시장지표] {indices}
    [공포지수] {f_score}
    
    위 데이터를 바탕으로:
    1. '버핏 지수' 관점에서 현재 시장이 고평가인지 저평가인지 판단해줘.
    2. 현재 시황을 3줄로 요약하고 투자 전략을 제안해줘.
    """
    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel("gemini-pro")
        return model.generate_content(prompt).text
    except: return "분석 실패 (잠시 후 다시 시도하세요)"

@st.cache_data(ttl=43200)
def get_calendar():
    if API_KEY == "SECRET_KEY_NOT_FOUND": return []
    today = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""
    Today is {today}. Find major US/Korea economic events (CPI, FOMC, Earnings) for next 2 weeks.
    Return ONLY JSON: [{{"date":"MM-DD(Day)","event":"Event(KR)","importance":"⭐⭐⭐"}}]
    """
    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel("gemini-pro")
        res = model.generate_content(prompt).text
        match = re.search(r'\[.*\]', res, re.DOTALL)
        return json.loads(match.group(0)) if match else []
    except: return []

# ---------------------------------------------------------
# [페이지] 로그인 & 메인
# ---------------------------------------------------------
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.title("💸 Asset Manager Login")
    t1, t2 = st.tabs(["로그인", "회원가입"])
    with t1:
        with st.form("login"):
            id_ = st.text_input("아이디")
            pw = st.text_input("비밀번호", type="password")
            if st.form_submit_button("로그인"):
                u = load_users()
                if id_ in u and u[id_] == pw:
                    st.session_state['logged_in']=True; st.session_state['username']=id_; st.rerun()
                else: st.error("로그인 실패")
    with t2:
        with st.form("signup"):
            nid = st.text_input("새 아이디")
            npw = st.text_input("새 비밀번호", type="password")
            if st.form_submit_button("가입"):
                u = load_users()
                if nid not in u and nid and npw:
                    save_user(nid, npw); st.success("가입 완료")
                else: st.error("이미 있거나 입력 오류")
    st.stop()

# --- 로그인 후 메인 화면 ---
if 'portfolio_db' not in st.session_state: st.session_state['portfolio_db'] = load_portfolio()
db = st.session_state['portfolio_db']

# 1. 헤더
c1, c2 = st.columns([9, 1])
with c1: st.title("📈 Market & Portfolio")
with c2: 
    if st.button("로그아웃"): 
        st.session_state['logged_in']=False; st.rerun()

# 2. 시장 정보 (가로 스크롤 느낌)
st.subheader("1. 시장 정보")
market_data = get_market_data()
rate_krw = market_data.get("💵 환율", (1400.0, 0))[0] # 환율 추출

m_cols = st.columns(5)
idx = 0
for k, v in market_data.items():
    if k == "💵 환율": continue # 환율은 계산용으로 쓰고 표시는 따로 안함 (공간 절약)
    with m_cols[idx % 5]:
        st.metric(k, f"{v[0]:,.2f}", f"{v[1]:.2f}%")
    idx += 1

st.divider()

# 3. Fear & Greed + 경제 일정
c_left, c_right = st.columns([1, 1])

with c_left:
    st.subheader("2. Fear & Greed Index")
    fs, fr = get_fear_greed()
    if fs:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = fs, 
            title = {'text': f"<b>{fr}</b>", 'font': {'size': 20}},
            gauge = {
                'axis': {'range': [0, 100]}, 'bar': {'color': "black"},
                'steps': [{'range': [0, 25], 'color': '#FF4B4B'}, {'range': [75, 100], 'color': '#008000'}]
            }
        ))
        fig.update_layout(height=250, margin=dict(t=40,b=20,l=30,r=30))
        st.plotly_chart(fig, use_container_width=True)
    else: st.error("지수 로딩 실패")

with c_right:
    st.subheader("3. 주요 경제 일정")
    with st.spinner("일정 불러오는 중..."):
        cal_data = get_calendar()
    if cal_data:
        st.dataframe(pd.DataFrame(cal_data), column_config={"date":"날짜","event":"이벤트","importance":"중요도"}, hide_index=True, use_container_width=True, height=250)
    else: st.info("일정 데이터 없음")

# 4. 버핏지수 및 시황 분석
st.subheader("4. 버핏지수 및 시황 분석 (AI)")
with st.spinner("Gemini가 시장을 분석하고 있습니다..."):
    briefing = get_ai_briefing(market_data, fs)
    st.info(briefing)

st.divider()

# 5. 포트폴리오 (계산 로직)
st.subheader("5. 내 포트폴리오")

# 전체 종목 리스트업 및 현재가 조회
all_tickers = []
for acc in db.values(): all_tickers.extend(acc.keys())
all_tickers = list(set(all_tickers))
prices = get_prices(all_tickers)

# 자산 합산 변수
total_krw_eval = 0.0 # 총 평가금 (원화 환산)
total_krw_invest = 0.0 # 총 매수금 (원화 환산)
kr_eval = 0.0 # 국내 평가금
us_eval = 0.0 # 해외 평가금 (달러)

# 계산 루프
for acc in db.values():
    for t, info in acc.items():
        qty = float(info['qty'])
        avg = float(info['avg_price'])
        curr = float(prices.get(t, avg))
        
        country = detect_country(t)
        
        if country == "KR":
            # 국내: 원화 그대로 합산
            kr_eval += curr * qty
            total_krw_eval += curr * qty
            total_krw_invest += avg * qty
        else:
            # 해외: 달러 합산 & 원화 환산 합산
            us_eval += curr * qty
            total_krw_eval += (curr * qty) * rate_krw
            total_krw_invest += (avg * qty) * rate_krw

total_profit = total_krw_eval - total_krw_invest
total_yield = (total_profit / total_krw_invest * 100) if total_krw_invest > 0 else 0.0

# 5-1. 자산 현황 카드 (Toss Style)
st.markdown(f"""
<div class="metric-card">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <div class="card-title">총 평가금액 (원화 환산)</div>
            <div class="card-value">₩ {total_krw_eval:,.0f}</div>
            <div class="card-sub" style="color:{'red' if total_profit>=0 else 'blue'}">
                {total_profit:+,.0f}원 ({total_yield:+.2f}%)
            </div>
        </div>
        <div style="text-align:right; border-left:1px solid #eee; padding-left:20px;">
            <div class="card-title">🇰🇷 국내 주식</div>
            <div class="card-value" style="font-size:20px;">₩ {kr_eval:,.0f}</div>
            <br>
            <div class="card-title">🇺🇸 해외 주식</div>
            <div class="card-value" style="font-size:20px;">$ {us_eval:,.2f}</div>
            <div class="card-sub">(≈ ₩ {us_eval*rate_krw:,.0f})</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 5-2. 탭 기능 (목록, 거래, 관리)
pt1, pt2, pt3 = st.tabs(["📋 주식 목록", "🔄 거래하기", "⚙️ 계좌관리"])

with pt1:
    if not db: st.warning("보유 주식이 없습니다.")
    else:
        # 국내/해외 분리해서 보여주기
        rows_kr = []
        rows_us = []
        
        for acc_name, stocks in db.items():
            for t, info in stocks.items():
                qty = float(info['qty'])
                avg = float(info['avg_price'])
                curr = float(prices.get(t, avg))
                p_rate = ((curr - avg)/avg)*100
                
                # 데이터 행 생성
                row = {
                    "계좌": acc_name, "종목": t, "수량": qty, 
                    "평단": avg, "현재가": curr, "수익률": p_rate/100,
                    "평가금": curr * qty
                }
                
                if detect_country(t) == "KR": rows_kr.append(row)
                else: rows_us.append(row)
        
        c_kr, c_us = st.columns(2)
        with c_kr:
            st.markdown("##### 🇰🇷 국내 주식")
            if rows_kr: st.dataframe(pd.DataFrame(rows_kr), column_config={"수익률": st.column_config.NumberColumn(format="%.2f%%")}, hide_index=True)
            else: st.caption("없음")
        with c_us:
            st.markdown("##### 🇺🇸 해외 주식")
            if rows_us: st.dataframe(pd.DataFrame(rows_us), column_config={"수익률": st.column_config.NumberColumn(format="%.2f%%")}, hide_index=True)
            else: st.caption("없음")

with pt2:
    st.subheader("주문하기 (매수/매도)")
    if db:
        acc_list = list(db.keys())
        c1, c2, c3, c4, c5 = st.columns([2, 1, 2, 2, 2])
        sel_acc = c1.selectbox("계좌", acc_list)
        type_Order = c2.selectbox("유형", ["매수", "매도"])
        t_in = c3.text_input("종목코드").upper()
        q_in = c4.number_input("수량", 1)
        p_in = c5.number_input("단가", 0.0)
        
        if st.button("주문 실행", use_container_width=True):
            if t_in and p_in > 0:
                # 매수 로직
                if type_Order == "매수":
                    if t_in in db[sel_acc]:
                        old_q = db[sel_acc][t_in]['qty']
                        old_p = db[sel_acc][t_in]['avg_price']
                        new_q = old_q + q_in
                        new_p = ((old_p*old_q)+(p_in*q_in))/new_q
                        db[sel_acc][t_in] = {'avg_price':new_p, 'qty':new_q}
                    else: db[sel_acc][t_in] = {'avg_price':p_in, 'qty':q_in}
                    st.success("매수 완료")
                # 매도 로직
                else:
                    if t_in in db[sel_acc]:
                        curr_q = db[sel_acc][t_in]['qty']
                        if q_in >= curr_q: del db[sel_acc][t_in] # 전량매도
                        else: db[sel_acc][t_in]['qty'] -= q_in # 부분매도
                        st.success("매도 완료")
                    else: st.error("보유하지 않은 종목")
                save_portfolio(db); st.rerun()
    else: st.warning("계좌를 먼저 만드세요.")

with pt3:
    st.subheader("계좌 설정")
    with st.expander("➕ 계좌 추가", expanded=True):
        na = st.text_input("새 계좌명")
        if st.button("생성"):
            if na and na not in db: db[na]={}; save_portfolio(db); st.rerun()
    
    if db:
        with st.expander("🗑️ 계좌 삭제"):
            da = st.selectbox("삭제할 계좌", list(db.keys()))
            if st.button("삭제 실행"):
                del db[da]; save_portfolio(db); st.rerun()
