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
# [설정] 페이지 기본 설정 & 토스 스타일 CSS
# ---------------------------------------------------------
st.set_page_config(page_title="My Asset", layout="wide", page_icon="💸", initial_sidebar_state="collapsed")

# 토스증권 느낌의 CSS (카드 디자인, 폰트, 여백 등)
st.markdown("""
<style>
    /* 기본 배경 및 여백 */
    .main .block-container {max-width: 1000px; padding-top: 2rem; padding-bottom: 5rem;}
    
    /* 카드 스타일 컨테이너 */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    /* 텍스트 스타일 */
    .big-number {font-size: 28px; font-weight: 700; color: #333;}
    .sub-text {font-size: 14px; color: #666;}
    .profit-plus {color: #e72a2a; font-weight: 600;} /* 상승 빨강 */
    .profit-minus {color: #2a6ce7; font-weight: 600;} /* 하락 파랑 */
    
    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] {gap: 20px;}
    .stTabs [data-baseweb="tab"] {height: 50px; white-space: pre-wrap; background-color: #f9f9f9; border-radius: 10px; gap: 1px; padding-top: 10px; padding-bottom: 10px;}
    .stTabs [aria-selected="true"] {background-color: #eef2ff; color: #3b66ff; font-weight: bold;}
    
    /* 사이드바 숨김 */
    [data-testid="collapsedControl"] {display: none}
    section[data-testid="stSidebar"] {display: none}
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
# [회원 및 데이터 관리 시스템]
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
# [핵심 로직] 매수, 매도, 계좌관리
# ---------------------------------------------------------
def trade_stock(account, ticker, price, qty, type="buy"):
    db = st.session_state['portfolio_db']
    if account not in db: db[account] = {}
    ticker = ticker.upper()
    
    if type == "buy": # 매수 (물타기)
        if ticker in db[account]:
            old_qty = db[account][ticker]['qty']
            old_price = db[account][ticker]['avg_price']
            new_total_qty = old_qty + qty
            new_avg_price = ((old_price * old_qty) + (price * qty)) / new_total_qty
            db[account][ticker] = {'avg_price': new_avg_price, 'qty': new_total_qty}
        else:
            db[account][ticker] = {'avg_price': price, 'qty': qty}
        msg = f"✅ {ticker} {qty}주 매수 완료!"
        
    elif type == "sell": # 매도 (분할매도)
        if ticker not in db[account]: return "❌ 보유하지 않은 종목입니다."
        old_qty = db[account][ticker]['qty']
        
        if qty > old_qty: return "❌ 보유 수량보다 많이 팔 수 없습니다."
        
        if qty == old_qty: # 전량 매도
            del db[account][ticker]
            msg = f"🗑️ {ticker} 전량 매도 완료!"
        else: # 부분 매도 (평단가는 유지됨)
            db[account][ticker]['qty'] = old_qty - qty
            msg = f"📉 {ticker} {qty}주 매도 완료! (잔고: {old_qty - qty}주)"
            
    save_portfolio(db)
    return msg

def manage_account_action(action, old_name, new_name=None):
    db = st.session_state['portfolio_db']
    if action == "rename":
        if old_name in db and new_name:
            db[new_name] = db.pop(old_name)
            save_portfolio(db)
            st.rerun()
    elif action == "delete":
        if old_name in db:
            del db[old_name]
            save_portfolio(db)
            st.rerun()
    elif action == "create":
        if new_name and new_name not in db:
            db[new_name] = {}
            save_portfolio(db)
            st.rerun()

# ---------------------------------------------------------
# [데이터 페칭]
# ---------------------------------------------------------
@st.cache_data(ttl=600)
def get_market_indices():
    tickers = {"USD/KRW": "KRW=X", "S&P500": "^GSPC", "NASDAQ": "^IXIC", "KOSPI": "^KS11"}
    data = {}
    for name, ticker in tickers.items():
        try:
            h = yf.Ticker(ticker).history(period="5d")
            c = h['Close'].iloc[-1]; p = h['Close'].iloc[-2]
            data[name] = (c, ((c - p) / p) * 100)
        except: data[name] = (0, 0)
    return data

@st.cache_data(ttl=300)
def get_current_prices(ticker_list):
    """여러 종목의 현재가를 한번에 가져옴 (속도 최적화)"""
    if not ticker_list: return {}
    try:
        data = yf.download(ticker_list, period="1d", progress=False)['Close']
        if data.empty: return {}
        # 종목이 1개일 때와 여러개일 때 처리
        if len(ticker_list) == 1:
            return {ticker_list[0]: data.iloc[-1]}
        return data.iloc[-1].to_dict()
    except: return {}

# ---------------------------------------------------------
# [로그인 페이지]
# ---------------------------------------------------------
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False

def login_page():
    st.markdown("<h1 style='text-align: center;'>💸 Smart Asset</h1>", unsafe_allow_html=True)
    t1, t2 = st.tabs(["로그인", "회원가입"])
    with t1:
        with st.form("login"):
            id_ = st.text_input("아이디")
            pw = st.text_input("비밀번호", type="password")
            if st.form_submit_button("로그인", use_container_width=True):
                db = load_users()
                if id_ in db and db[id_] == pw:
                    st.session_state['logged_in'] = True; st.session_state['username'] = id_; st.rerun()
                else: st.error("정보가 일치하지 않습니다.")
    with t2:
        with st.form("signup"):
            n_id = st.text_input("새 아이디"); n_pw = st.text_input("새 비밀번호", type="password")
            if st.form_submit_button("가입하기", use_container_width=True):
                db = load_users()
                if n_id in db: st.error("이미 있는 아이디")
                elif n_id and n_pw: save_user(n_id, n_pw); st.success("가입 완료! 로그인하세요.")

if not st.session_state['logged_in']: login_page(); st.stop()

# ---------------------------------------------------------
# [메인 대시보드]
# ---------------------------------------------------------
# 데이터 로드
if 'portfolio_db' not in st.session_state: st.session_state['portfolio_db'] = load_portfolio()
db = st.session_state['portfolio_db']

# 1. 헤더 (유저 환영 및 로그아웃)
c_h1, c_h2 = st.columns([8, 1])
with c_h1: st.write(f"👋 반가워요, **{st.session_state['username']}**님")
with c_h2: 
    if st.button("로그아웃"): 
        st.session_state['logged_in'] = False; st.session_state['username'] = None; st.rerun()

# 2. 자산 전체 계산 (Hero Section)
total_invest = 0.0
total_eval = 0.0
all_tickers = []
for acc in db.values():
    all_tickers.extend(acc.keys())
    for info in acc.values():
        total_invest += info['avg_price'] * info['qty']

# 현재가 가져오기
all_tickers = list(set(all_tickers))
price_map = get_current_prices(all_tickers)

for acc in db.values():
    for t, info in acc.items():
        if t in price_map:
            total_eval += price_map[t] * info['qty']
        else:
            total_eval += info['avg_price'] * info['qty'] # 현재가 없으면 매수가로 대체

total_profit = total_eval - total_invest
total_yield = (total_profit / total_invest * 100) if total_invest > 0 else 0.0

# 3. 토스 스타일 메인 카드 (총 자산 현황)
st.markdown(f"""
<div class="metric-card">
    <div class="sub-text">총 보유자산</div>
    <div class="big-number">₩ {total_eval:,.0f}</div>
    <hr style="margin: 10px 0; border-color: #f0f0f0;">
    <div style="display: flex; justify-content: space-between;">
        <div>
            <span class="sub-text">투자원금</span><br>
            <strong>₩ {total_invest:,.0f}</strong>
        </div>
        <div style="text-align: right;">
            <span class="sub-text">총 수익</span><br>
            <span class="{ 'profit-plus' if total_profit >= 0 else 'profit-minus' }">
                {total_profit:+,.0f} ({total_yield:+.2f}%)
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 4. 탭 구성 (포트폴리오, 거래하기, 계좌관리, 시장정보)
tab_pf, tab_trade, tab_manage, tab_market = st.tabs(["📊 포트폴리오", "🔄 거래하기", "⚙️ 계좌관리", "🌍 시장정보"])

# [탭 1] 포트폴리오 (계좌별 상세)
with tab_pf:
    if not db:
        st.info("📌 계좌가 없습니다. '계좌관리' 탭에서 먼저 만들어주세요.")
    else:
        for acc_name, stocks in db.items():
            # 계좌별 요약 계산
            acc_invest = sum(i['avg_price'] * i['qty'] for i in stocks.values())
            acc_eval = sum((price_map.get(t, i['avg_price']) * i['qty']) for t, i in stocks.items())
            acc_profit = acc_eval - acc_invest
            acc_yield = (acc_profit / acc_invest * 100) if acc_invest > 0 else 0.0
            
            # 계좌 카드 헤더
            with st.expander(f"📂 {acc_name} (₩{acc_eval:,.0f})", expanded=True):
                # 계좌 요약
                c1, c2, c3 = st.columns(3)
                c1.metric("평가손익", f"{acc_profit:,.0f}", f"{acc_yield:.2f}%")
                c2.metric("매입금액", f"{acc_invest:,.0f}")
                
                # 종목 리스트 (DataFrame)
                if stocks:
                    rows = []
                    for t, info in stocks.items():
                        curr = price_map.get(t, info['avg_price'])
                        p_rate = ((curr - info['avg_price']) / info['avg_price']) * 100
                        val = curr * info['qty']
                        rows.append({
                            "종목": t,
                            "현재가": curr,
                            "수익률": p_rate / 100, # % 서식을 위해 소수로
                            "평가손익": (curr - info['avg_price']) * info['qty'],
                            "보유수량": info['qty'],
                            "매입가": info['avg_price']
                        })
                    
                    df = pd.DataFrame(rows)
                    st.dataframe(
                        df,
                        column_config={
                            "종목": "종목명",
                            "현재가": st.column_config.NumberColumn(format="%.2f"),
                            "수익률": st.column_config.NumberColumn(format="%.2f%%"),
                            "평가손익": st.column_config.NumberColumn(format="%.0f"),
                            "매입가": st.column_config.NumberColumn(format="%.2f")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.caption("보유 주식이 없습니다.")

# [탭 2] 거래하기 (매수/매도/분할매도)
with tab_trade:
    st.subheader("주문하기")
    if not db:
        st.warning("계좌를 먼저 생성해주세요.")
    else:
        tr_acc = st.selectbox("계좌 선택", list(db.keys()))
        col_type = st.radio("주문 유형", ["매수 (Buy)", "매도 (Sell)"], horizontal=True, label_visibility="collapsed")
        
        with st.form("trade_form"):
            c1, c2, c3 = st.columns([2, 1, 2])
            tr_ticker = c1.text_input("종목코드 (예: TSLA)").upper()
            tr_qty = c2.number_input("수량", min_value=1, value=1)
            tr_price = c3.number_input("거래단가", min_value=0.0, value=0.0)
            
            submitted = st.form_submit_button("주문 실행", use_container_width=True)
            
            if submitted:
                if not tr_ticker or tr_price <= 0:
                    st.error("종목과 가격을 정확히 입력해주세요.")
                else:
                    mode = "buy" if "매수" in col_type else "sell"
                    msg = trade_stock(tr_acc, tr_ticker, tr_price, tr_qty, mode)
                    if "❌" in msg: st.error(msg)
                    else: st.success(msg); st.rerun()

# [탭 3] 계좌 관리 (생성/수정/삭제)
with tab_manage:
    st.subheader("계좌 설정")
    
    # 1. 계좌 생성
    with st.expander("➕ 새 계좌 만들기", expanded=False):
        new_acc_name = st.text_input("계좌 이름 입력 (예: 비상금)")
        if st.button("계좌 생성"):
            manage_account_action("create", None, new_acc_name)

    # 2. 계좌 수정/삭제
    if db:
        with st.expander("🔧 계좌 이름 변경 / 삭제", expanded=False):
            target_acc = st.selectbox("관리할 계좌", list(db.keys()))
            
            c_ren, c_del = st.columns([3, 1])
            with c_ren:
                rename_to = st.text_input("새로운 이름")
                if st.button("이름 변경"):
                    manage_account_action("rename", target_acc, rename_to)
            with c_del:
                st.write("") # 줄맞춤용
                st.write("") 
                if st.button("🗑️ 계좌 삭제", type="primary"):
                    manage_account_action("delete", target_acc)
    else:
        st.info("생성된 계좌가 없습니다.")

# [탭 4] 시장 정보 (Market)
with tab_market:
    st.markdown("##### 🌍 주요 지수")
    indices = get_market_indices()
    m_cols = st.columns(4)
    for i, (k, v) in enumerate(indices.items()):
        color = "off" if v[1] == 0 else ("inverse" if v[1] > 0 else "normal") # 상승=초록(st.metric 기본)
        m_cols[i].metric(k, f"{v[0]:,.2f}", f"{v[1]:.2f}%")
    
    # AI 브리핑 (기존 기능 연동)
    if st.button("🤖 AI 시장 브리핑 (Gemini)"):
        if API_KEY == "SECRET_KEY_NOT_FOUND":
            st.error("API 키가 설정되지 않았습니다.")
        else:
            try:
                genai.configure(api_key=API_KEY)
                model = genai.GenerativeModel("gemini-pro")
                with st.spinner("시장 분석 중..."):
                    res = model.generate_content(f"현재 시장 지표: {indices}. 투자자에게 3줄 요약 조언.")
                    st.info(res.text)
            except: st.error("AI 분석 실패")
