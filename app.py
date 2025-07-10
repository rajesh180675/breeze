import streamlit as st
import pandas as pd
from breeze_connect import BreezeConnect
import os
from dotenv import load_dotenv
from datetime import datetime
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Pro Options Chain Analyzer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HELPER & SETUP FUNCTIONS ---
def load_credentials():
    if 'BREEZE_API_KEY' in st.secrets:
        api_key, api_secret = st.secrets["BREEZE_API_KEY"], st.secrets["BREEZE_API_SECRET"]
    else:
        load_dotenv()
        api_key, api_secret = os.getenv("BREEZE_API_KEY"), os.getenv("BREEZE_API_SECRET")
    return api_key, api_secret

@st.cache_resource(show_spinner="Connecting to Breeze API...")
def initialize_breeze(api_key, api_secret, session_token):
    try:
        breeze = BreezeConnect(api_key=api_key)
        breeze.generate_session(api_secret=api_secret, session_token=session_token)
        st.success("Successfully connected to Breeze API!")
        return breeze
    except Exception as e:
        st.error(f"Connection Failed: {e}")
        return None

def robust_date_parse(date_string):
    formats = ["%Y-%m-%dT%H:%M:%S.%fZ", "%d-%b-%Y", "%Y-%m-%d"]
    for fmt in formats:
        try: return datetime.strptime(date_string, fmt)
        except (ValueError, TypeError): continue
    return None

# --- DATA FETCHING WITH RETRY LOGIC ---
@st.cache_data(ttl=3600, show_spinner="Fetching available expiry dates...")
def get_expiry_map(_breeze, symbol):
    try:
        spot_data = _breeze.get_quotes(stock_code=symbol, exchange_code="NSE", product_type="cash")
        if not spot_data.get('Success'): raise Exception(f"Could not get spot price: {spot_data.get('Error')}")
        spot_price = float(spot_data['Success'][0]['ltp'])
        step = 100 if symbol == "BANKNIFTY" else 50
        nearby_strike = round(spot_price / step) * step
        data = _breeze.get_option_chain_quotes(stock_code=symbol, exchange_code="NFO", product_type="options", right="Call", expiry_date=None, strike_price=nearby_strike)
        if not data.get('Success'): raise Exception(f"API Error fetching expiries: {data.get('Error')}")
        raw_dates = sorted(list(set(item['expiry_date'] for item in data['Success'])))
        expiry_map = {parsed_date.strftime("%d-%b-%Y"): d for d in raw_dates if (parsed_date := robust_date_parse(d))}
        return expiry_map
    except Exception as e:
        st.error(f"Could not fetch expiry dates: {e}")
        return {}

def get_options_chain_data_with_retry(_breeze, symbol, api_expiry_date, max_retries=3):
    """Fetch data with automatic retry on failure using exponential backoff."""
    for attempt in range(max_retries):
        try:
            spot_data = _breeze.get_quotes(stock_code=symbol, exchange_code="NSE", product_type="cash")
            if not spot_data.get('Success'): raise Exception(f"API Error getting spot price: {spot_data.get('Error')}")
            spot_price = float(spot_data['Success'][0]['ltp'])
            call_data = _breeze.get_option_chain_quotes(stock_code=symbol, exchange_code="NFO", product_type="options", right="Call", expiry_date=api_expiry_date)
            if not call_data.get('Success'): raise Exception(f"API Error getting Call options: {call_data.get('Error')}")
            put_data = _breeze.get_option_chain_quotes(stock_code=symbol, exchange_code="NFO", product_type="options", right="Put", expiry_date=api_expiry_date)
            if not put_data.get('Success'): raise Exception(f"API Error getting Put options: {put_data.get('Error')}")
            st.session_state.last_fetch_time = datetime.now()
            return call_data.get('Success', []) + put_data.get('Success', []), spot_price
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"Failed to fetch options chain after {max_retries} attempts: {e}")
                return None, None
            time.sleep(1 * (2 ** attempt)) # Exponential backoff: 1s, 2s, 4s

# --- DATA ANALYSIS & VISUALIZATION ---
def process_and_analyze(raw_data):
    if not raw_data:
        st.warning("No options data received from API")
        return pd.DataFrame()
    df = pd.DataFrame(raw_data)
    expected_cols = ['oi', 'oi_change', 'ltp', 'volume', 'strike_price', 'right']
    for col in expected_cols:
        if col not in df.columns: df[col] = 0
    df = df.apply(pd.to_numeric, errors='ignore')
    calls_df, puts_df = df[df['right'] == 'Call'], df[df['right'] == 'Put']
    chain = pd.merge(calls_df, puts_df, on="strike_price", suffixes=('_call', '_put'), how="outer").sort_values("strike_price").fillna(0)
    chain.rename(columns={'oi_call': 'Call OI', 'oi_change_call': 'Call Chng OI', 'ltp_call': 'Call LTP', 'strike_price': 'Strike', 'ltp_put': 'Put LTP', 'oi_change_put': 'Put Chng OI', 'oi_put': 'Put OI', 'volume_call':'Call Volume', 'volume_put':'Put Volume'}, inplace=True)
    return chain

def calculate_metrics(chain_df, spot_price):
    """Calculate all key analytics including enhanced sentiment."""
    # Vectorized Max Pain Calculation for performance
    strikes = chain_df['Strike'].values
    call_oi = chain_df['Call OI'].values
    put_oi = chain_df['Put OI'].values
    strike_matrix = strikes[:, np.newaxis]
    call_pain = np.sum(np.maximum(strike_matrix - strikes, 0) * call_oi, axis=1)
    put_pain = np.sum(np.maximum(strikes - strike_matrix, 0) * put_oi, axis=1)
    total_pain = call_pain + put_pain
    max_pain = strikes[np.argmin(total_pain)] if len(total_pain) > 0 else 0

    total_call_oi = chain_df['Call OI'].sum()
    total_put_oi = chain_df['Put OI'].sum()
    pcr = round(total_put_oi / total_call_oi if total_call_oi > 0 else 0, 2)
    net_oi_change = chain_df['Put Chng OI'].sum() - chain_df['Call Chng OI'].sum()
    
    # Enhanced Sentiment Score
    sentiment_score = 0
    if pcr > 1.2: sentiment_score += 30
    elif pcr < 0.8: sentiment_score -= 30
    else: sentiment_score += (pcr - 1) * 75
    if net_oi_change > 0: sentiment_score += 30
    elif net_oi_change < 0: sentiment_score -= 30
    if spot_price < max_pain: sentiment_score += 15
    elif spot_price > max_pain: sentiment_score -= 15
    
    return {
        'max_pain': max_pain,
        'resistance': chain_df.nlargest(3, 'Call OI')['Strike'].tolist(),
        'support': chain_df.nlargest(3, 'Put OI')['Strike'].tolist(),
        'full_pcr': pcr,
        'net_oi_change': net_oi_change,
        'sentiment': max(-100, min(100, sentiment_score))
    }

def create_oi_chart(chain_df, atm_strike, spot_price, max_pain=None):
    # Same as before...
    pass

# ... (other functions like create_heatmap, style_dataframe remain the same)

def track_historical_data(symbol, expiry, metrics):
    """Store historical metrics for trend analysis in session state."""
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = []
    
    st.session_state.historical_data.append({
        'timestamp': datetime.now(), 'symbol': symbol, 'expiry': expiry,
        'pcr': metrics['full_pcr'], 'max_pain': metrics['max_pain'], 'sentiment': metrics['sentiment']
    })
    
    # Keep only the last 200 data points for performance
    if len(st.session_state.historical_data) > 200:
        st.session_state.historical_data = st.session_state.historical_data[-200:]

# --- MAIN APPLICATION UI ---
st.title("🚀 Pro Options Chain Analyzer")

api_key, api_secret = load_credentials()
if not api_key or not api_secret:
    st.error("API_KEY or API_SECRET is not configured.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Controls")
    session_token = st.text_input("Enter Daily Session Token", type="password")
    symbol = st.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY", "FINNIFTY"])
    if 'strike_range_pct' not in st.session_state: st.session_state.strike_range_pct = 15
    st.session_state.strike_range_pct = st.slider("Strike Display Range (%)", 5, 50, st.session_state.strike_range_pct)
    auto_refresh_interval = st.select_slider("Auto-refresh Interval (s)", [0, 30, 60, 120], 0)

if auto_refresh_interval > 0: st_autorefresh(interval=auto_refresh_interval * 1000, key="datarefresh")

if session_token:
    breeze = initialize_breeze(api_key, api_secret, session_token)
    if breeze:
        expiry_map = get_expiry_map(breeze, symbol)
        if expiry_map:
            with st.form("options_form"):
                selected_display_date = st.selectbox("Select Expiry Date", list(expiry_map.keys()))
                load_button = st.form_submit_button("📊 Analyze Options Chain")
            
            if load_button:
                st.session_state.run_analysis = True
                st.session_state.selected_display_date = selected_display_date

            if 'run_analysis' in st.session_state and st.session_state.run_analysis:
                selected_api_date = expiry_map[st.session_state.selected_display_date]
                raw_data, spot_price = get_options_chain_data_with_retry(breeze, symbol, selected_api_date)
                
                if raw_data and spot_price:
                    full_chain_df = process_and_analyze(raw_data)
                    if not full_chain_df.empty:
                        metrics = calculate_metrics(full_chain_df, spot_price)
                        track_historical_data(symbol, st.session_state.selected_display_date, metrics)
                        
                        # --- Main Dashboard ---
                        atm_strike = min(full_chain_df['Strike'], key=lambda x: abs(x - spot_price))
                        lower_bound = spot_price * (1 - st.session_state.strike_range_pct / 100)
                        upper_bound = spot_price * (1 + st.session_state.strike_range_pct / 100)
                        display_df = full_chain_df[(full_chain_df['Strike'] >= lower_bound) & (full_chain_df['Strike'] <= upper_bound)]
                        
                        # ... [Rest of the UI display logic, which is largely unchanged] ...

else:
    st.info("👋 Please enter your daily session token in the sidebar to begin.")
