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
# [보안] API 키 설정
# ---------------------------------------------------------
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    API_KEY = "SECRET_KEY_NOT_FOUND"

# ---------------------------------------------------------
# [회원 관리 시스템] JSON 파일로 유저 정보 관리
# ---------------------------------------------------------
USER_FILE = "users.json"

def load_users():
    """유저 목록 불러오기"""
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"admin": "1234"} # 기본 관리자 계정

def save_user(username, password):
    """신규 유저 저장하기"""
    users = load_users()
    users[username] = password
    with open(USER_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=4)

# ---------------------------------------------------------
# [로그인 & 회원가입 화면]
# ---------------------------------------------------------
def login_page():
    st.title("🔐 Smart Asset Home")
    st.write("개인 자산 관리 시스템에 오신 것을 환영합니다.")

    # 탭으로 로그인/회원가입 분리
    tab1, tab2 = st.tabs(["🔑 로그인", "📝 회원가입"])

    # 1. 로그인 탭
    with tab1:
        with st.form("login_form"):
            username = st.text_input("아이디")
            password = st.text_input("비밀번호", type="password")
            submit = st.form_submit_button("로그인")

            if submit:
                users_db = load_users()
                if username in users_db and users_db[username] == password:
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.success(f"{username}님 환영합니다!")
                    st.rerun()
                else:
                    st.error("아이디가 없거나 비밀번호가 틀렸습니다.")

    # 2. 회원가입 탭
    with tab2:
        with st.form("signup_form"):
            new_user = st.text_input("새 아이디 만들기")
            new_pw = st.text_input("새 비밀번호 설정", type="password")
            new_pw_chk = st.text_input("비밀번호 확인", type="password")
            signup_submit = st.form_submit_button("가입하기")

            if signup_submit:
                users_db = load_users()
                if new_user in users_db:
                    st.error("이미 존재하는 아이디입니다.")
                elif new_pw != new_pw_chk:
                    st.error("비밀번호가 서로 다릅니다.")
                elif not new_user or not new_pw:
                    st.error("아이디와 비밀번호를 입력해주세요.")
                else:
                    save_user(new_user, new_pw)
                    st.success("🎉 가입 성공! '로그인' 탭에서 접속해주세요.")

def logout():
    st.session_state['logged_in'] = False
    st.session_state['username'] = None
    if 'portfolio_db' in st.session_state:
        del st.session_state['portfolio_db']
    st.rerun()

# ---------------------------------------------------------
# [데이터 관리] 유저별 포트폴리오 파일 분리
# ---------------------------------------------------------
def get_user_file():
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
# [메인 로직 실행]
# ---------------------------------------------------------
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# 로그인이 안 되어 있으면 로그인 페이지 표시 후 중단
if not st.session_state['logged_in']:
    login_page()
    st.stop()

# =========================================================
# [대시보드 화면] (로그인 사용자만 접근 가능)
# =========================================================

# 상단 헤더
col_h1, col_h2 = st.columns([8, 1])
with col_h1:
    st.write(f"👋 **{st.session_state['username']}**님의 포트폴리오")
with col_h2:
    if st.button("로그아웃"):
        logout()

# AI 설정
try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-pro")
except: pass

# 데이터 로딩
if 'portfolio_db' not in st.session_state:
    st.session_state['portfolio_db'] = load_portfolio()

# --- [함수들] (기존 로직 유지) ---
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
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Referer": "https://www.cnn.com/"
        }
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
    save_portfolio(db)

# --- [UI 구성] ---
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
        chg = ((sdf.iloc[-1] - sdf.iloc[-21]) / sdf.iloc[-21]) * 100
        fig = go.Figure(go.Bar(x=chg.index, y=chg.values, marker_color=['red' if x>0 else 'blue' for x in chg.values]))
        fig.update_layout(height=250, margin=dict(t=30,b=20,l=20,r=20))
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("📂 My Portfolio")

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
