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
from scipy.stats import norm
from scipy.optimize import minimize_scalar

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Pro Options & Greeks Analyzer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HELPER & SETUP FUNCTIONS ---
def load_credentials():
    if 'BREEZE_API_KEY' in st.secrets:
        return st.secrets["BREEZE_API_KEY"], st.secrets["BREEZE_API_SECRET"]
    else:
        load_dotenv()
        return os.getenv("BREEZE_API_KEY"), os.getenv("BREEZE_API_SECRET")

@st.cache_resource(show_spinner="Connecting to Breeze API...")
def initialize_breeze(api_key, api_secret, session_token):
    try:
        breeze = BreezeConnect(api_key=api_key)
        breeze.generate_session(api_secret=api_secret, session_token=session_token)
        st.success("API Connection Successful!")
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

# --- GREEKS & IV CALCULATION ---
def black_scholes_price(volatility, option_type, spot, strike, t, r):
    if t <= 0: return 0
    d1 = (np.log(spot / strike) + (r + 0.5 * volatility**2) * t) / (volatility * np.sqrt(t))
    d2 = d1 - volatility * np.sqrt(t)
    if option_type == 'Call':
        return spot * norm.cdf(d1) - strike * np.exp(-r * t) * norm.cdf(d2)
    else:
        return strike * np.exp(-r * t) * norm.cdf(-d2) - spot * norm.cdf(-d1)

@st.cache_data(max_entries=5000)
def calculate_iv(option_type, spot, strike, market_price, t, r=0.07):
    if t <= 0 or market_price <= 0.01: return 0
    objective = lambda vol: abs(black_scholes_price(vol, option_type, spot, strike, t, r) - market_price)
    result = minimize_scalar(objective, bounds=(0.001, 4.0), method='bounded')
    return result.x

@st.cache_data(max_entries=5000)
def calculate_greeks(iv, option_type, spot, strike, t, r=0.07):
    if iv is None or iv <= 0 or t <= 0: return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0}
    d1 = (np.log(spot / strike) + (r + 0.5 * iv**2) * t) / (iv * np.sqrt(t))
    d2 = d1 - iv * np.sqrt(t)
    gamma = norm.pdf(d1) / (spot * iv * np.sqrt(t))
    vega = spot * norm.pdf(d1) * np.sqrt(t) / 100
    if option_type == 'Call':
        delta = norm.cdf(d1)
        theta = (-spot * norm.pdf(d1) * iv / (2 * np.sqrt(t)) - r * strike * np.exp(-r * t) * norm.cdf(d2)) / 365
    else:
        delta = norm.cdf(d1) - 1
        theta = (-spot * norm.pdf(d1) * iv / (2 * np.sqrt(t)) + r * strike * np.exp(-r * t) * norm.cdf(-d2)) / 365
    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta}

# --- DATA FETCHING ---
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
            time.sleep(1 * (2 ** attempt))

# --- DATA PROCESSING & ANALYSIS ---
def process_and_analyze(raw_data, spot_price, expiry_date):
    if not raw_data: return pd.DataFrame()
    df = pd.DataFrame(raw_data)
    for col in ['oi', 'oi_change', 'ltp', 'volume', 'strike_price', 'right']:
        if col not in df.columns: df[col] = 0
    df = df.apply(pd.to_numeric, errors='ignore')
    
    calls = df[df['right'] == 'Call']
    puts = df[df['right'] == 'Put']
    chain = pd.merge(calls, puts, on="strike_price", suffixes=('_call', '_put'), how="outer").sort_values("strike_price").fillna(0)
    
    expiry_dt = datetime.strptime(expiry_date, "%d-%b-%Y")
    t = (expiry_dt - datetime.now()).total_seconds() / (365 * 24 * 3600)

    if t > 0:
        with st.spinner("Calculating IV and Greeks... This may take a moment."):
            chain['Call IV'] = np.vectorize(calculate_iv)('Call', spot_price, chain['strike_price'], chain['ltp_call'], t)
            chain['Put IV'] = np.vectorize(calculate_iv)('Put', spot_price, chain['strike_price'], chain['ltp_put'], t)
            call_greeks = chain.apply(lambda r: calculate_greeks(r['Call IV'], 'Call', spot_price, r['strike_price'], t), axis=1, result_type='expand')
            put_greeks = chain.apply(lambda r: calculate_greeks(r['Put IV'], 'Put', spot_price, r['strike_price'], t), axis=1, result_type='expand')
            chain = pd.concat([chain, call_greeks.add_suffix('_call'), put_greeks.add_suffix('_put')], axis=1)
    
    chain.rename(columns={'oi_call': 'Call OI', 'oi_change_call': 'Call Chng OI', 'ltp_call': 'Call LTP', 'strike_price': 'Strike', 'ltp_put': 'Put LTP', 'oi_change_put': 'Put Chng OI', 'oi_put': 'Put OI', 'volume_call':'Call Volume', 'volume_put':'Put Volume'}, inplace=True)
    return chain

def calculate_dashboard_metrics(chain_df, spot_price):
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
    
    sentiment_score = 0
    if pcr > 1.2: sentiment_score += 30
    elif pcr < 0.8: sentiment_score -= 30
    else: sentiment_score += (pcr - 1) * 75
    if net_oi_change > 50000: sentiment_score += 30
    elif net_oi_change < -50000: sentiment_score -= 30
    if spot_price < max_pain: sentiment_score += 15
    elif spot_price > max_pain: sentiment_score -= 15
    
    return {
        'max_pain': max_pain,
        'resistance': chain_df.nlargest(3, 'Call OI')['Strike'].tolist(),
        'support': chain_df.nlargest(3, 'Put OI')['Strike'].tolist(),
        'full_pcr': pcr,
        'net_oi_change': net_oi_change,
        'sentiment': int(max(-100, min(100, sentiment_score)))
    }

def track_historical_data(symbol, expiry, metrics):
    if 'historical_data' not in st.session_state: st.session_state.historical_data = []
    st.session_state.historical_data.append({'timestamp': datetime.now(), 'symbol': symbol, 'expiry': expiry, **metrics})
    if len(st.session_state.historical_data) > 200: st.session_state.historical_data = st.session_state.historical_data[-200:]

# --- VISUALIZATION FUNCTIONS ---
def create_oi_chart(df, atm, spot, max_pain):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['Strike'], y=df['Call OI'], name='Call OI', marker_color='rgba(239, 83, 80, 0.7)'))
    fig.add_trace(go.Bar(x=df['Strike'], y=df['Put OI'], name='Put OI', marker_color='rgba(46, 125, 50, 0.7)'))
    fig.add_vline(x=spot, line_width=2, line_dash="solid", line_color="blue", annotation_text="Spot")
    fig.add_vline(x=atm, line_width=2, line_dash="dash", line_color="black", annotation_text="ATM")
    if max_pain: fig.add_vline(x=max_pain, line_width=2, line_dash="dot", line_color="purple", annotation_text="Max Pain")
    fig.update_layout(title_text='Open Interest Distribution', barmode='group', height=400, hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def create_heatmap(df):
    heat_df = df.set_index('Strike')[['Call LTP', 'Put LTP']].sort_index(ascending=False)
    fig = go.Figure(data=go.Heatmap(z=heat_df.values, x=heat_df.columns, y=heat_df.index, colorscale="Viridis", hovertemplate='Strike: %{y}<br>Type: %{x}<br>Premium: %{z:,.2f}<extra></extra>'))
    fig.update_layout(title_text='Premium Heatmap', yaxis_title='Strike Price', height=500)
    return fig

def create_iv_smile_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Strike'], y=df['Call IV']*100, mode='lines+markers', name='Call IV'))
    fig.add_trace(go.Scatter(x=df['Strike'], y=df['Put IV']*100, mode='lines+markers', name='Put IV'))
    fig.update_layout(title_text='Implied Volatility Smile', yaxis_title='Implied Volatility (%)', hovermode='x unified')
    return fig

def style_dataframe(df, atm_strike, spot_price):
    def style_row(row):
        styles = [''] * len(row)
        if row.Strike < spot_price: styles[3:8] = ['background-color: #ffebee'] * 5 # Call side
        if row.Strike > spot_price: styles[9:] = ['background-color: #e8f5e9'] * 5 # Put side
        if row.Strike == atm_strike: styles = ['background-color: #fff9c4'] * len(row)
        return styles
    return df.style.apply(style_row, axis=1).format('{:,.2f}', na_rep='-').format({'Strike': '{:,.0f}', 'Call OI': '{:,.0f}', 'Call Chng OI': '{:,.0f}', 'Put OI': '{:,.0f}', 'Put Chng OI': '{:,.0f}', 'Call Volume': '{:,.0f}', 'Put Volume': '{:,.0f}'})

# --- MAIN APPLICATION UI ---
def main():
    st.title("🚀 Pro Options & Greeks Analyzer")
    
    api_key, api_secret = load_credentials()
    if not api_key or not api_secret:
        st.error("API credentials not found. Please configure them in Streamlit secrets or a .env file.")
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
                        full_chain_df = process_and_analyze(raw_data, spot_price, st.session_state.selected_display_date)
                        if not full_chain_df.empty:
                            metrics = calculate_dashboard_metrics(full_chain_df, spot_price)
                            track_historical_data(symbol, st.session_state.selected_display_date, metrics)
                            atm_strike = min(full_chain_df['Strike'], key=lambda x: abs(x - spot_price))
                            
                            lower_bound = spot_price * (1 - st.session_state.strike_range_pct / 100)
                            upper_bound = spot_price * (1 + st.session_state.strike_range_pct / 100)
                            display_df = full_chain_df[(full_chain_df['Strike'] >= lower_bound) & (full_chain_df['Strike'] <= upper_bound)]

                            data_age = (datetime.now() - st.session_state.get('last_fetch_time', datetime.now())).seconds
                            staleness_indicator = "🟢" if data_age < 30 else "🟡" if data_age < 60 else "🔴"
                            
                            st.header(f"{symbol} at {spot_price:,.2f}")
                            st.caption(f"Expiry: {st.session_state.selected_display_date} | Updated: {datetime.now().strftime('%I:%M:%S %p')} {staleness_indicator}")
                            
                            sentiment = metrics['sentiment']
                            sentiment_text = "BULLISH" if sentiment > 30 else "BEARISH" if sentiment < -30 else "NEUTRAL"
                            st.progress((sentiment + 100) / 200, text=f"Market Sentiment: {sentiment_text} ({sentiment:+.0f}/100)")
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("PCR (Full Chain)", f"{metrics['full_pcr']:.2f}")
                            col2.metric("Max Pain", f"{metrics['max_pain']:,.0f}")
                            col3.metric("Net OI Change (Puts-Calls)", f"{metrics['net_oi_change']:+,.0f}")
                            
                            st.markdown(f"**Key Resistance (High Call OI):** {', '.join(f'{s:,.0f}' for s in metrics['resistance'])}")
                            st.markdown(f"**Key Support (High Put OI):** {', '.join(f'{s:,.0f}' for s in metrics['support'])}")
                            st.divider()

                            # --- Main Data Table ---
                            cols_to_display = ['Call OI', 'Call Chng OI', 'Call IV', 'Call LTP', 'Call Delta', 'Strike', 'Put Delta', 'Put LTP', 'Put IV', 'Put Chng OI', 'Put OI']
                            greeks_cols_to_add = ['Call Gamma', 'Call Vega', 'Call Theta', 'Put Gamma', 'Put Vega', 'Put Theta']
                            for col in greeks_cols_to_add:
                                if col in full_chain_df.columns: cols_to_display.insert(cols_to_display.index('Strike'), col)
                            
                            st.dataframe(style_dataframe(display_df, atm_strike, spot_price), use_container_width=True)
                            st.download_button(label="📥 Download Displayed Data as CSV", data=display_df.to_csv(index=False).encode('utf-8'), file_name=f"{symbol}_{st.session_state.selected_display_date}.csv", mime="text/csv")

                            # --- Tabs for Advanced Analysis ---
                            tab_list = ["📊 OI Analysis", "🔥 Premium Heatmap", "😊 IV Smile", "🧮 Greeks", "⏳ History"]
                            tabs = st.tabs(tab_list)
                            with tabs[0]: st.plotly_chart(create_oi_chart(display_df, atm_strike, spot_price, metrics['max_pain']), use_container_width=True)
                            with tabs[1]: st.plotly_chart(create_heatmap(display_df), use_container_width=True)
                            with tabs[2]: st.plotly_chart(create_iv_smile_chart(display_df), use_container_width=True)
                            with tabs[3]:
                                st.subheader("Greeks Analysis")
                                greeks_df = full_chain_df[['Strike', 'Call Delta', 'Call Gamma', 'Call Vega', 'Call Theta', 'Put Delta', 'Put Gamma', 'Put Vega', 'Put Theta']]
                                st.dataframe(greeks_df.style.format("{:.3f}").background_gradient(cmap='coolwarm_r', axis=1, subset=['Call Delta', 'Put Delta']))
                            with tabs[4]:
                                st.subheader("Intra-session Trend")
                                if 'historical_data' in st.session_state and len(st.session_state.historical_data) > 1:
                                    hist_df = pd.DataFrame(st.session_state.historical_data)
                                    st.line_chart(hist_df.set_index('timestamp')[['sentiment', 'full_pcr']])
                                else:
                                    st.info("Historical data for this session will appear here after a few fetches.")

    else:
        st.info("👋 Please enter your daily session token in the sidebar to begin.")

if __name__ == "__main__":
    main()
