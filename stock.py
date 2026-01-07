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
st.set_page_config(page_title="Pro Asset Manager", layout="wide", page_icon="🔐")

st.markdown("""
<style>
    [data-testid="collapsedControl"] {display: none}
    section[data-testid="stSidebar"] {display: none}
    .main .block-container {padding-top: 2rem; padding-bottom: 2rem;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# [사용자 관리] 아이디/비밀번호 설정 (원하는대로 추가하세요)
# ---------------------------------------------------------
USERS = {
    "admin": "1234",      # 아이디: admin, 비번: 1234
    "guest": "0000",      # 아이디: guest, 비번: 0000
    "wife": "love1234"    # 예시: 와이프 계정
}

# ---------------------------------------------------------
# [보안] API 키 설정
# ---------------------------------------------------------
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    API_KEY = "SECRET_KEY_NOT_FOUND"

# ---------------------------------------------------------
# [로그인 화면 함수]
# ---------------------------------------------------------
def login_page():
    st.title("🔐 Asset Manager Login")
    st.write("나만의 포트폴리오를 관리하려면 로그인하세요.")

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        with st.form("login_form"):
            username = st.text_input("아이디 (ID)")
            password = st.text_input("비밀번호 (Password)", type="password")
            submit = st.form_submit_button("로그인")

            if submit:
                if username in USERS and USERS[username] == password:
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.success(f"환영합니다, {username}님!")
                    st.rerun() # 화면 새로고침해서 대시보드로 이동
                else:
                    st.error("아이디 또는 비밀번호가 틀렸습니다.")

def logout():
    st.session_state['logged_in'] = False
    st.session_state['username'] = None
    st.session_state.pop('portfolio_db', None) # 데이터 초기화
    st.rerun()

# ---------------------------------------------------------
# [데이터 관리] 유저별 파일 분리 로직 (핵심!)
# ---------------------------------------------------------
def get_user_file():
    # 로그인한 유저의 이름을 따서 파일명을 만듦 (예: portfolio_admin.json)
    user = st.session_state.get('username', 'guest')
    return f"portfolio_{user}.json"

def load_portfolio():
    file_name = get_user_file()
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_portfolio(data):
    file_name = get_user_file()
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# ---------------------------------------------------------
# [메인 로직 시작]
# ---------------------------------------------------------
# 1. 로그인 상태 확인
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# 2. 로그인이 안 되어 있으면 -> 로그인 페이지 보여주고 프로그램 종료(return)
if not st.session_state['logged_in']:
    login_page()
    st.stop() # 여기서 코드 실행 멈춤 (아래 대시보드 안 보여줌)

# =========================================================
# 이 아래부터는 "로그인 성공한 사람"만 볼 수 있는 코드입니다.
# =========================================================

# 상단바 (로그아웃 버튼)
col_head1, col_head2 = st.columns([8, 1])
with col_head1:
    st.write(f"👋 안녕하세요, **{st.session_state['username']}**님! 성투하세요!")
with col_head2:
    if st.button("로그아웃"):
        logout()

# ---------------------------------------------------------
# [AI 모델 연결]
# ---------------------------------------------------------
try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-pro")
except: pass

# ---------------------------------------------------------
# [기능 함수들] (기존과 동일)
# ---------------------------------------------------------
if 'portfolio_db' not in st.session_state:
    st.session_state['portfolio_db'] = load_portfolio()

@st.cache_data(ttl=600)
def get_market_indices():
    tickers = {"USD/KRW": "KRW=X", "US 10Y": "^TNX", "VIX": "^VIX", "KOSPI": "^KS11", "NASDAQ": "^IXIC"}
    data = {}
    for name, ticker in tickers.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            cur = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2]
            data[name] = (cur, ((cur - prev) / prev) * 100)
        except: data[name] = (0, 0)
    return data

@st.cache_data(ttl=900)
def get_fear_and_greed():
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=5)
        d = r.json()
        return d['fear_and_greed']['score'], d['fear_and_greed']['rating']
    except: return None, "N/A"

@st.cache_data(ttl=600)
def get_stock_details(t):
    try:
        s = yf.Ticker(t)
        h = s.history(period="1mo")
        if h.empty: return None
        cur = h['Close'].iloc[-1]
        delta = h['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + (gain/loss))).iloc[-1]
        i = s.info
        return {"current": cur, "rsi": rsi, "per": i.get('trailingPE',0), "pbr": i.get('priceToBook',0), "div": i.get('dividendYield',0)*100 if i.get('dividendYield') else 0}
    except: return None

@st.cache_data(ttl=3600)
def get_sector_data():
    try:
        s = {"XLK":"XLK", "SOXX":"SOXX", "XLF":"XLF", "XLV":"XLV", "XLE":"XLE"}
        df = yf.download(list(s.values()), period="6mo", progress=False)['Close']
        return df
    except: return pd.DataFrame()

def add_stock(acc, t, p, q):
    db = st.session_state['portfolio_db']
    if acc not in db: db[acc] = {}
    t = t.upper()
    if t in db[acc]:
        oq = db[acc][t]['qty']; op = db[acc][t]['avg_price']
        nq = oq + q; np = ((op * oq) + (p * q)) / nq
        db[acc][t] = {'avg_price': np, 'qty': nq}
    else: db[acc][t] = {'avg_price': p, 'qty': q}
    save_portfolio(db) # 유저별 파일에 저장

# ---------------------------------------------------------
# [UI - 대시보드]
# ---------------------------------------------------------
st.divider()
mk = get_market_indices()
cols = st.columns(5)
for i, (k, v) in enumerate(mk.items()): cols[i].metric(k, f"{v[0]:,.2f}", f"{v[1]:.2f}%")

st.divider()
c1, c2 = st.columns([1, 1])
with c1:
    st.subheader("😨 Fear & Greed Index")
    fs, fr = get_fear_and_greed()
    if fs:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=fs, title={'text':fr}, gauge={'axis':{'range':[0,100]}, 'bar':{'color':'black'}, 'steps':[{'range':[0,25],'color':'red'},{'range':[75,100],'color':'green'}]}))
        fig.update_layout(height=250, margin=dict(t=30,b=20,l=20,r=20))
        st.plotly_chart(fig, use_container_width=True)
    else: st.error("지수 로딩 실패")

with c2:
    st.subheader("📊 Sector Trend (1 Month)")
    sdf = get_sector_data()
    if not sdf.empty:
        # 1달 전 대비 등락률
        chg = ((sdf.iloc[-1] - sdf.iloc[-21]) / sdf.iloc[-21]) * 100
        fig = go.Figure(go.Bar(x=chg.index, y=chg.values, marker_color=['red' if x>0 else 'blue' for x in chg.values]))
        fig.update_layout(height=250, margin=dict(t=30,b=20,l=20,r=20))
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader(f"📂 My Portfolio ({st.session_state['username']})")

with st.expander("➕ 자산 관리 / 계좌 추가", expanded=False):
    db = st.session_state['portfolio_db']
    accs = list(db.keys())
    t1, t2 = st.tabs(["매수", "계좌생성"])
    with t1:
        if accs:
            c1, c2, c3, c4, c5 = st.columns([2,2,1,2,1])
            sa = c1.selectbox("계좌", accs)
            st_in = c2.text_input("티커").upper()
            sq = c3.number_input("수량",1)
            sp = c4.number_input("단가",0.0)
            if c5.button("추가"):
                if st_in and sp>0: add_stock(sa, st_in, sp, sq); st.rerun()
    with t2:
        na = st.text_input("새 계좌명")
        if st.button("생성"):
            if na: db[na] = {}; save_portfolio(db); st.rerun()

all_data = []
if db:
    for an, stocks in db.items():
        if not stocks: continue
        st.markdown(f"**📌 {an}**")
        rows = []
        for t, i in stocks.items():
            inf = get_stock_details(t)
            if inf:
                curr = inf['current']
                prof = ((curr - i['avg_price']) / i['avg_price']) * 100
                rows.append({"종목":t, "수량":i['qty'], "평단":f"{i['avg_price']:.2f}", "현재":f"{curr:.2f}", "수익률":f"{prof:.2f}%", "RSI":f"{inf['rsi']:.1f}", "PBR":f"{inf['pbr']:.1f}"})
                all_data.append(f"[{an}] {t}: 수익{prof:.1f}%, PBR{inf['pbr']:.1f}")
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            d_col, _ = st.columns([2,5])
            d_t = d_col.selectbox(f"삭제({an})", ["선택"]+list(stocks.keys()), key=an)
            if d_t!="선택" and d_col.button("삭제", key=f"d_{an}"):
                del db[an][d_t]; save_portfolio(db); st.rerun()

st.divider()
if st.button("🤖 AI 포트폴리오 진단"):
    if API_KEY == "SECRET_KEY_NOT_FOUND": st.error("API 키 없음")
    elif not all_data: st.warning("데이터 없음")
    else:
        with st.spinner("AI 분석 중..."):
            p = f"시장상황:{mk}. 공포지수:{fs}. 내자산:{all_data}. 전문가 관점에서 진단해줘."
            try:
                res = model.generate_content(p)
                st.info(res.text)
            except: st.error("분석 실패")
