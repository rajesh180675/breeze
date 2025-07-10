import streamlit as st
import pandas as pd
from breeze_connect import BreezeConnect
import os
from dotenv import load_dotenv
from datetime import datetime
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Pro Options Chain Analyzer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HELPER & SETUP FUNCTIONS ---
def load_credentials():
    """Loads API credentials from Streamlit secrets or a local .env file."""
    if 'BREEZE_API_KEY' in st.secrets:
        api_key, api_secret = st.secrets["BREEZE_API_KEY"], st.secrets["BREEZE_API_SECRET"]
    else:
        load_dotenv()
        api_key, api_secret = os.getenv("BREEZE_API_KEY"), os.getenv("BREEZE_API_SECRET")
    return api_key, api_secret

@st.cache_resource(show_spinner="Connecting to Breeze API...")
def initialize_breeze(api_key, api_secret, session_token):
    """Initializes and caches the BreezeConnect instance."""
    try:
        breeze = BreezeConnect(api_key=api_key)
        breeze.generate_session(api_secret=api_secret, session_token=session_token)
        st.success("Successfully connected to Breeze API!")
        return breeze
    except Exception as e:
        st.error(f"Connection Failed: {e}")
        return None

def robust_date_parse(date_string):
    """Tries to parse a date string from a list of known formats."""
    formats = ["%Y-%m-%dT%H:%M:%S.%fZ", "%d-%b-%Y", "%Y-%m-%d"]
    for fmt in formats:
        try: return datetime.strptime(date_string, fmt)
        except (ValueError, TypeError): continue
    return None

# --- DATA FETCHING ---
@st.cache_data(ttl=3600, show_spinner="Fetching available expiry dates...")
def get_expiry_map(_breeze, symbol):
    """Fetches expiry dates and returns a ready-to-use map."""
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

@st.cache_data(ttl=15, show_spinner="Fetching latest options chain data...")
def get_options_chain_data(_breeze, symbol, api_expiry_date):
    """Fetches options chain using the robust two-call method."""
    try:
        spot_data = _breeze.get_quotes(stock_code=symbol, exchange_code="NSE", product_type="cash")
        if not spot_data.get('Success'): raise Exception(f"API Error getting spot price: {spot_data.get('Error')}")
        spot_price = float(spot_data['Success'][0]['ltp'])
        call_data = _breeze.get_option_chain_quotes(stock_code=symbol, exchange_code="NFO", product_type="options", right="Call", expiry_date=api_expiry_date)
        if not call_data.get('Success'): raise Exception(f"API Error getting Call options: {call_data.get('Error')}")
        put_data = _breeze.get_option_chain_quotes(stock_code=symbol, exchange_code="NFO", product_type="options", right="Put", expiry_date=api_expiry_date)
        if not put_data.get('Success'): raise Exception(f"API Error getting Put options: {put_data.get('Error')}")
        st.session_state.last_fetch_time = datetime.now() # For staleness indicator
        return call_data.get('Success', []) + put_data.get('Success', []), spot_price
    except Exception as e:
        st.error(f"Failed to fetch options chain: {e}")
        return None, None

# --- DATA ANALYSIS & VISUALIZATION ---
def process_and_analyze(raw_data):
    """Processes raw data defensively to prevent KeyErrors."""
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
    chain.rename(columns={'oi_call': 'Call OI', 'oi_change_call': 'Call Chng OI', 'ltp_call': 'Call LTP', 'strike_price': 'Strike', 'ltp_put': 'Put LTP', 'oi_change_put': 'Put Chng OI', 'oi_put': 'Put OI'}, inplace=True)
    return chain

def calculate_metrics(chain_df, spot_price):
    """Calculate all key analytics for the dashboard."""
    strikes = chain_df['Strike'].values
    max_pain_values = [(
        ((strikes - s) * chain_df['Call OI']).where(strikes > s, 0).sum() + 
        ((s - strikes) * chain_df['Put OI']).where(strikes < s, 0).sum()
    ) for s in strikes]
    
    total_call_oi = chain_df['Call OI'].sum()
    total_put_oi = chain_df['Put OI'].sum()
    
    max_pain = strikes[np.argmin(max_pain_values)] if max_pain_values else 0
    sentiment_score = 0
    pcr = round(total_put_oi / total_call_oi if total_call_oi > 0 else 0, 2)
    net_oi_change = chain_df['Put Chng OI'].sum() - chain_df['Call Chng OI'].sum()

    # Calculate sentiment score
    if pcr > 1.2: sentiment_score += 35
    elif pcr < 0.8: sentiment_score -= 35
    else: sentiment_score += (pcr - 1) * 60
    if net_oi_change > 0: sentiment_score += 35
    elif net_oi_change < 0: sentiment_score -= 35
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
    """Creates an enhanced Plotly chart for Open Interest distribution."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=chain_df['Strike'], y=chain_df['Call OI'], name='Call OI', marker_color='rgba(239, 83, 80, 0.7)'))
    fig.add_trace(go.Bar(x=chain_df['Strike'], y=chain_df['Put OI'], name='Put OI', marker_color='rgba(46, 125, 50, 0.7)'))
    fig.add_vline(x=spot_price, line_width=2, line_dash="solid", line_color="blue", annotation_text="Spot", annotation_position="top left")
    fig.add_vline(x=atm_strike, line_width=2, line_dash="dash", line_color="black", annotation_text="ATM", annotation_position="top right")
    if max_pain:
        fig.add_vline(x=max_pain, line_width=2, line_dash="dot", line_color="purple", annotation_text="Max Pain")
    fig.update_layout(title_text='Open Interest Distribution', xaxis_title='Strike Price', yaxis_title='Open Interest', barmode='group', height=400, hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def create_heatmap(df):
    """Creates a heatmap of option premiums."""
    heat_df = df.set_index('Strike')[['Call LTP', 'Put LTP']].sort_index(ascending=False)
    fig = go.Figure(data=go.Heatmap(
        z=heat_df.values,
        x=heat_df.columns,
        y=heat_df.index,
        colorscale="Viridis",
        hovertemplate='Strike: %{y}<br>Type: %{x}<br>Premium: %{z:,.2f}<extra></extra>'
    ))
    fig.update_layout(title_text='Premium Heatmap', yaxis_title='Strike Price', height=500)
    return fig

def style_dataframe(df, atm_strike, spot_price):
    """Applies conditional formatting to the DataFrame."""
    def style_row(row):
        styles = [''] * len(row)
        if row.Strike < spot_price: styles[0:3] = ['background-color: #ffebee'] * 3
        if row.Strike > spot_price: styles[4:7] = ['background-color: #e8f5e9'] * 3
        if row.Strike == atm_strike: styles = ['background-color: #fff9c4'] * len(row)
        return styles
    return df.style.apply(style_row, axis=1).format('{:,.2f}', na_rep='-').format({'Strike': '{:,.0f}', 'Call OI': '{:,.0f}', 'Call Chng OI': '{:,.0f}', 'Put OI': '{:,.0f}', 'Put Chng OI': '{:,.0f}'})

# --- MAIN APPLICATION UI ---
st.title("🚀 Pro Options Chain Analyzer")

api_key, api_secret = load_credentials()
if not api_key or not api_secret:
    st.error("API_KEY or API_SECRET is not configured. Please set them in your Streamlit secrets or .env file.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Controls")
    session_token = st.text_input("Enter Your Daily Session Token", type="password")
    symbol = st.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY", "FINNIFTY"])
    if 'strike_range_pct' not in st.session_state: st.session_state.strike_range_pct = 15
    st.session_state.strike_range_pct = st.slider("Strike Display Range (%)", 5, 50, st.session_state.strike_range_pct)
    auto_refresh_interval = st.select_slider("Auto-refresh Interval (seconds)", [0, 30, 60, 120], 0, help="Set to 0 to disable.")

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
                raw_data, spot_price = get_options_chain_data(breeze, symbol, selected_api_date)
                
                if raw_data and spot_price:
                    full_chain_df = process_and_analyze(raw_data)
                    if not full_chain_df.empty:
                        metrics = calculate_metrics(full_chain_df, spot_price)
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
                        filtered_pcr = round(display_df['Put OI'].sum() / display_df['Call OI'].sum() if display_df['Call OI'].sum() > 0 else 0, 2)
                        col1.metric("PCR (Displayed)", f"{filtered_pcr:.2f}", delta=f"Full Chain: {metrics['full_pcr']:.2f}", delta_color="off")
                        col2.metric("Max Pain", f"{metrics['max_pain']:,.0f}")
                        col3.metric("Net OI Change (Puts-Calls)", f"{metrics['net_oi_change']:+,.0f}")
                        
                        st.markdown(f"**Key Resistance (High Call OI):** {', '.join(f'{s:,.0f}' for s in metrics['resistance'])}")
                        st.markdown(f"**Key Support (High Put OI):** {', '.join(f'{s:,.0f}' for s in metrics['support'])}")
                        st.divider()

                        st.dataframe(style_dataframe(display_df, atm_strike, spot_price), use_container_width=True)
                        st.download_button(label="📥 Download Displayed Data as CSV", data=display_df.to_csv(index=False).encode('utf-8'), file_name=f"{symbol}_{st.session_state.selected_display_date}.csv", mime="text/csv")
                        
                        tab1, tab2, tab3 = st.tabs(["📊 OI Distribution", "🔥 Premium Heatmap", "🎯 Strike Drill-Down"])
                        with tab1:
                            st.plotly_chart(create_oi_chart(display_df, atm_strike, spot_price, metrics['max_pain']), use_container_width=True)
                        with tab2:
                            st.plotly_chart(create_heatmap(display_df), use_container_width=True)
                        with tab3:
                            strike_list = display_df['Strike'].tolist()
                            default_index = strike_list.index(atm_strike) if atm_strike in strike_list else 0
                            selected_strike = st.selectbox("Select Strike for Detailed Analysis", strike_list, index=default_index)
                            strike_data = display_df[display_df['Strike'] == selected_strike].iloc[0]
                            c1, c2 = st.columns(2)
                            c1.metric("Call Premium", f"₹{strike_data['Call LTP']:,.2f}")
                            c1.metric("Call OI", f"{strike_data['Call OI']:,.0f}", delta=f"{strike_data['Call Chng OI']:+,.0f}")
                            c2.metric("Put Premium", f"₹{strike_data['Put LTP']:,.2f}")
                            c2.metric("Put OI", f"{strike_data['Put OI']:,.0f}", delta=f"{strike_data['Put Chng OI']:+,.0f}")
else:
    st.info("👋 Please enter your daily session token in the sidebar to begin.")
