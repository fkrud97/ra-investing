import streamlit as st
import google.generativeai as genai
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import json
import os
import requests
import re # 정규표현식 추가 (JSON 추출용)
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
# [회원 관리 시스템]
# ---------------------------------------------------------
USER_FILE = "users.json"

def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"admin": "1234"}

def save_user(username, password):
    users = load_users()
    users[username] = password
    with open(USER_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=4)

def login_page():
    st.title("🔐 Smart Asset Home")
    st.write("개인 자산 관리 시스템에 오신 것을 환영합니다.")
    
    t1, t2 = st.tabs(["🔑 로그인", "📝 회원가입"])
    
    with t1:
        with st.form("login"):
            id_ = st.text_input("아이디")
            pw = st.text_input("비밀번호", type="password")
            if st.form_submit_button("로그인"):
                db = load_users()
                if id_ in db and db[id_] == pw:
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = id_
                    st.rerun()
                else: st.error("정보가 일치하지 않습니다.")
    
    with t2:
        with st.form("signup"):
            new_id = st.text_input("새 아이디")
            new_pw = st.text_input("새 비밀번호", type="password")
            if st.form_submit_button("가입"):
                db = load_users()
                if new_id in db: st.error("이미 있는 아이디입니다.")
                elif new_id and new_pw:
                    save_user(new_id, new_pw)
                    st.success("가입 완료! 로그인해주세요.")
                else: st.error("정보를 입력해주세요.")

def logout():
    st.session_state['logged_in'] = False
    st.session_state['username'] = None
    if 'portfolio_db' in st.session_state: del st.session_state['portfolio_db']
    st.rerun()

# ---------------------------------------------------------
# [데이터 관리] 유저별 포트폴리오
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
# [메인 실행 로직]
# ---------------------------------------------------------
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    login_page()
    st.stop()

# =========================================================
# [대시보드 화면] (로그인 사용자 전용)
# =========================================================

# 상단 헤더
c_h1, c_h2 = st.columns([8, 1])
with c_h1: st.write(f"👋 **{st.session_state['username']}**님의 대시보드")
with c_h2: 
    if st.button("로그아웃"): logout()

# AI 설정 (모델 자동 감지)
try:
    genai.configure(api_key=API_KEY)
    target_model = "gemini-pro"
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            if 'gemini' in m.name:
                target_model = m.name
                break
    model = genai.GenerativeModel(target_model)
except: pass

if 'portfolio_db' not in st.session_state:
    st.session_state['portfolio_db'] = load_portfolio()

# --- [데이터 함수 강화 수정] ---

@st.cache_data(ttl=600)
def get_market_indices():
    tickers = {"USD/KRW": "KRW=X", "US 10Y": "^TNX", "VIX": "^VIX", "KOSPI": "^KS11", "NASDAQ": "^IXIC"}
    data = {}
    for name, ticker in tickers.items():
        try:
            h = yf.Ticker(ticker).history(period="5d")
            c = h['Close'].iloc[-1]; p = h['Close'].iloc[-2]
            data[name] = (c, ((c - p) / p) * 100)
        except: data[name] = (0, 0)
    return data

@st.cache_data(ttl=900)
def get_fear_and_greed():
    """CNN 차단 우회를 위한 헤더 강화"""
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        # 헤더를 실제 브라우저처럼 위장
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.cnn.com/",
            "Origin": "https://www.cnn.com",
            "Accept-Language": "en-US,en;q=0.9"
        }
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        d = r.json()
        return d['fear_and_greed']['score'], d['fear_and_greed']['rating']
    except Exception as e:
        # 실패 시 로그 출력 (디버깅용)
        print(f"FearGreed Error: {e}")
        return None, "Error"

@st.cache_data(ttl=3600)
def get_sector_history():
    s = {"XLK":"XLK", "SOXX":"SOXX", "XLF":"XLF", "XLV":"XLV", "XLE":"XLE"}
    try:
        df = yf.download(list(s.values()), period="1y", progress=False)['Close']
        return df, s
    except: return pd.DataFrame(), s

def calculate_sector_change(df, period_str):
    periods = {"1일": 2, "1주": 5, "1달": 21, "1분기": 63, "반년": 126, "1년": 252}
    days = periods.get(period_str, 21)
    changes = {}
    if df.empty: return {}
    for t in df.columns:
        try:
            if len(df) < days: start = df[t].iloc[0]
            else: start = df[t].iloc[-days]
            curr = df[t].iloc[-1]
            changes[t] = ((curr - start) / start) * 100
        except: changes[t] = 0.0
    return changes

@st.cache_data(ttl=600)
def get_stock_details(t):
    try:
        s = yf.Ticker(t); h = s.history(period="1mo")
        if h.empty: return None
        cur = h['Close'].iloc[-1]
        delta = h['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + (gain/loss))).iloc[-1]
        i = s.info
        return {"current": cur, "rsi": rsi, "per": i.get('trailingPE',0), "pbr": i.get('priceToBook',0), "div": i.get('dividendYield',0)*100 if i.get('dividendYield') else 0}
    except: return None

# --- [AI 함수 강화 수정] ---

@st.cache_data(ttl=3600)
def get_ai_market_briefing(f_score):
    if API_KEY == "SECRET_KEY_NOT_FOUND": return "API 키가 설정되지 않았습니다."
    prompt = f"오늘 공포지수 {f_score}. 버핏지수 추정 및 투자 조언 3줄 요약."
    try: return model.generate_content(prompt).text
    except: return "분석 실패"

@st.cache_data(ttl=43200)
def get_ai_calendar_data():
    if API_KEY == "SECRET_KEY_NOT_FOUND": return []
    today = datetime.now().strftime("%Y-%m-%d")
    # Prompt 개선: JSON만 내놓으라고 강력하게 지시
    prompt = f"""
    Today is {today}. List 3-5 major US economic events (CPI, FOMC, Earnings) for next 2 weeks.
    Return ONLY JSON array. No markdown. No text.
    Format: [{{"date":"MM-DD(Day)","event":"Event Name(KR)","importance":"⭐⭐⭐"}}]
    """
    try:
        res = model.generate_content(prompt)
        text = res.text
        
        # JSON 추출 로직 강화 (앞뒤 잡담 제거)
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            clean_json = match.group(0)
            return json.loads(clean_json)
        else:
            return []
    except Exception as e:
        print(f"Calendar Error: {e}")
        return []

# --- [UI 구성] ---

st.divider()
mk = get_market_indices()
cols = st.columns(5)
for i, (k, v) in enumerate(mk.items()): cols[i].metric(k, f"{v[0]:,.2f}", f"{v[1]:.2f}%")

st.divider()
st.subheader("💰 Smart Asset Dashboard")
sdf, smap = get_sector_history()
inv_smap = {v: k for k, v in smap.items()}

# 섹터 기간 선택
c1, c2 = st.columns([1, 6])
with c1:
    st.write("⏱️ **기간**")
    sel_period = st.radio("기간", ["1일", "1주", "1달", "1분기", "반년", "1년"], label_visibility="collapsed")
with c2:
    if not sdf.empty:
        chg = calculate_sector_change(sdf, sel_period)
        df_c = pd.DataFrame(list(chg.items()), columns=['Ticker', 'Change'])
        df_c['Name'] = df_c['Ticker'].map(inv_smap)
        df_c['Color'] = df_c['Change'].apply(lambda x: '#ff4b4b' if x > 0 else '#4b88ff')
        fig = go.Figure(go.Bar(x=df_c['Name'], y=df_c['Change'], marker_color=df_c['Color'], text=df_c['Change'].apply(lambda x: f"{x:.2f}%"), textposition='auto'))
        fig.update_layout(height=250, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("📅 Market Sentiment & Calendar")
cc1, cc2 = st.columns([1, 1])

with cc1:
    st.markdown("##### 😨 Fear & Greed Index")
    fs, fr = get_fear_and_greed()
    if fs is not None:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=fs, title={'text':fr}, gauge={'axis':{'range':[0,100]}, 'bar':{'color':'black'}, 'steps':[{'range':[0,25],'color':'red'},{'range':[75,100],'color':'green'}]}))
        fig.update_layout(height=200, margin=dict(t=30,b=20,l=20,r=20))
        st.plotly_chart(fig, use_container_width=True)
        st.info(get_ai_market_briefing(fs))
    else: 
        st.error("지수 로딩 실패 (CNN 연결 오류)")
        st.caption("잠시 후 다시 시도하거나, 브라우저를 새로고침 해보세요.")

with cc2:
    st.markdown("##### 🗓️ 주요 경제 일정 (2주)")
    with st.spinner("Loading..."):
        cal = get_ai_calendar_data()
    if cal: 
        st.dataframe(pd.DataFrame(cal), column_config={"date":"날짜","event":"이벤트","importance":"중요도"}, hide_index=True, use_container_width=True)
    else: 
        if API_KEY == "SECRET_KEY_NOT_FOUND":
            st.warning("⚠️ API 키가 없습니다. Secrets 설정을 확인하세요.")
        else:
            st.warning("일정 데이터 없음 (AI 응답 오류)")
            st.caption("AI가 데이터를 생성하지 못했습니다. 새로고침 해보세요.")

st.divider()
st.subheader("📂 My Portfolio")

with st.expander("➕ 자산 추가 / 계좌 관리", expanded=False):
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
                if st_in and sp>0: 
                    if sa not in db: db[sa]={}
                    if st_in in db[sa]:
                        oq=db[sa][st_in]['qty']; op=db[sa][st_in]['avg_price']
                        nq=oq+sq; np=((op*oq)+(sp*sq))/nq
                        db[sa][st_in]={'avg_price':np,'qty':nq}
                    else: db[sa][st_in]={'avg_price':sp,'qty':sq}
                    save_portfolio(db); st.rerun()
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

st.write("")
if st.button("🤖 AI 가치투자 진단"):
    if API_KEY == "SECRET_KEY_NOT_FOUND": st.error("API 키 없음")
    elif not all_data: st.warning("데이터 없음")
    else:
        with st.spinner("분석 중..."):
            p = f"시장:{mk}. 공포:{fs}. 내자산:{all_data}. 가치투자 관점 진단 및 조언."
            try: st.info(model.generate_content(p).text)
            except: st.error("분석 실패")
