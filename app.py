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
    page_title="Nifty Options Chain Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD CREDENTIALS ---
def load_credentials():
    """Loads API credentials from Streamlit secrets or a local .env file."""
    if 'BREEZE_API_KEY' in st.secrets:
        api_key = st.secrets["BREEZE_API_KEY"]
        api_secret = st.secrets["BREEZE_API_SECRET"]
    else:
        load_dotenv()
        api_key = os.getenv("BREEZE_API_KEY")
        api_secret = os.getenv("BREEZE_API_SECRET")
    return api_key, api_secret

# --- BREEZE API INITIALIZATION & CACHING ---
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

# --- DATA FETCHING & PROCESSING ---
def robust_date_parse(date_string):
    """Tries to parse a date string from a list of known formats."""
    formats = ["%Y-%m-%dT%H:%M:%S.%fZ", "%d-%b-%Y", "%Y-%m-%d"]
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except (ValueError, TypeError):
            continue
    return None

@st.cache_data(ttl=3600, show_spinner="Fetching available expiry dates...")
def get_expiry_map(_breeze, symbol):
    """Fetches expiry dates and returns a ready-to-use map."""
    try:
        spot_data = _breeze.get_quotes(stock_code=symbol, exchange_code="NSE", product_type="cash")
        if not spot_data.get('Success'):
            raise Exception(f"Could not get spot price for expiry lookup: {spot_data.get('Error')}")
        spot_price = float(spot_data['Success'][0]['ltp'])
        step = 100 if symbol == "BANKNIFTY" else 50
        nearby_strike = round(spot_price / step) * step
        data = _breeze.get_option_chain_quotes(stock_code=symbol, exchange_code="NFO", product_type="options", right="Call", expiry_date=None, strike_price=nearby_strike)
        if not data.get('Success'):
            raise Exception(f"API Error fetching expiries: {data.get('Error', 'Unknown error')}")
        raw_dates = sorted(list(set(item['expiry_date'] for item in data['Success'])))
        expiry_map = {}
        for d in raw_dates:
            parsed_date = robust_date_parse(d)
            if parsed_date:
                expiry_map[parsed_date.strftime("%d-%b-%Y")] = d 
            else:
                expiry_map[d] = d
        return expiry_map
    except Exception as e:
        st.error(f"Could not fetch expiry dates: {e}")
        return {}

@st.cache_data(ttl=15, show_spinner="Fetching latest options chain data...")
def get_options_chain_data(_breeze, symbol, api_expiry_date):
    """Fetches options chain using the robust two-call method."""
    try:
        spot_data = _breeze.get_quotes(stock_code=symbol, exchange_code="NSE", product_type="cash")
        if not spot_data.get('Success'): raise Exception(f"API Error getting spot price: {spot_data.get('Error', 'Unknown error')}")
        spot_price = float(spot_data['Success'][0]['ltp'])
        call_options_data = _breeze.get_option_chain_quotes(stock_code=symbol, exchange_code="NFO", product_type="options", right="Call", expiry_date=api_expiry_date, strike_price=None)
        if not call_options_data.get('Success'): raise Exception(f"API Error getting Call options: {call_options_data.get('Error', 'Unknown error')}")
        put_options_data = _breeze.get_option_chain_quotes(stock_code=symbol, exchange_code="NFO", product_type="options", right="Put", expiry_date=api_expiry_date, strike_price=None)
        if not put_options_data.get('Success'): raise Exception(f"API Error getting Put options: {put_options_data.get('Error', 'Unknown error')}")
        combined_options_data = call_options_data.get('Success', []) + put_options_data.get('Success', [])
        return combined_options_data, spot_price
    except Exception as e:
        st.error(f"Failed to fetch options chain: {e}")
        return None, None

# --- DATA ANALYSIS & FEATURE ENHANCEMENTS ---
def process_and_analyze(raw_data, spot_price):
    """Processes raw data defensively to prevent KeyErrors."""
    if not raw_data:
        st.warning("No options data received from API")
        return pd.DataFrame(), {}
    df = pd.DataFrame(raw_data)
    expected_cols = ['oi', 'oi_change', 'ltp', 'volume', 'strike_price', 'right']
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        st.info(f"Note: API did not return columns {missing_cols}. Using default value 0.")
        for col in missing_cols:
            df[col] = 0
    df = df.apply(pd.to_numeric, errors='ignore')
    calls_df, puts_df = df[df['right'] == 'Call'], df[df['right'] == 'Put']
    chain = pd.merge(calls_df, puts_df, on="strike_price", suffixes=('_call', '_put'), how="outer").sort_values("strike_price").fillna(0)
    chain.rename(columns={'oi_call': 'Call OI', 'oi_change_call': 'Call Chng OI', 'ltp_call': 'Call LTP', 'strike_price': 'Strike', 'ltp_put': 'Put LTP', 'oi_change_put': 'Put Chng OI', 'oi_put': 'Put OI'}, inplace=True)
    return chain

def calculate_additional_metrics(chain_df):
    """Calculate additional options analytics like Max Pain."""
    def calculate_max_pain(df):
        strikes = df['Strike'].values
        max_pain_values = []
        for strike in strikes:
            call_pain = ((strikes - strike) * df['Call OI']).where(strikes > strike, 0).sum()
            put_pain = ((strike - strikes) * df['Put OI']).where(strikes < strike, 0).sum()
            max_pain_values.append(call_pain + put_pain)
        if max_pain_values:
            return strikes[np.argmin(max_pain_values)]
        return 0
    return {'max_pain': calculate_max_pain(chain_df)}

def create_oi_chart(chain_df, atm_strike):
    """Creates a Plotly chart for Open Interest distribution."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=chain_df['Strike'], y=chain_df['Call OI'], name='Call OI', marker_color='green'))
    fig.add_trace(go.Bar(x=chain_df['Strike'], y=chain_df['Put OI'], name='Put OI', marker_color='red'))
    fig.add_vline(x=atm_strike, line_width=2, line_dash="dash", line_color="black", name="ATM")
    fig.update_layout(title_text='Open Interest Distribution by Strike', xaxis_title='Strike Price', yaxis_title='Open Interest', barmode='group', height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def style_dataframe(df, atm_strike, spot_price):
    def style_row(row):
        styles = [''] * len(row)
        if row.Strike < spot_price: styles[0:3] = ['background-color: #e8f5e9'] * 3
        if row.Strike > spot_price: styles[4:7] = ['background-color: #ffebee'] * 3
        if row.Strike == atm_strike: styles = ['background-color: #ffffc0'] * len(row)
        return styles
    return df.style.apply(style_row, axis=1).format({
        'Call OI': '{:,.0f}', 'Call Chng OI': '{:,.0f}', 'Call LTP': '{:,.2f}',
        'Strike': '{:,.0f}',
        'Put LTP': '{:,.2f}', 'Put Chng OI': '{:,.0f}', 'Put OI': '{:,.0f}'
    })

# --- MAIN APPLICATION UI ---
st.title("📈 Nifty Options Chain Analyzer")
st.markdown("A comprehensive tool for analyzing options data, powered by the ICICI Breeze API.")

# CORRECTED CODE: Load credentials at the start of the main script flow
api_key, api_secret = load_credentials()
if not api_key or not api_secret:
    st.error("API_KEY or API_SECRET is not configured. Please set them in your Streamlit secrets or .env file.")
    st.stop()

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Controls")
    session_token = st.text_input("Enter Your Daily Session Token", type="password")
    symbol = st.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY", "FINNIFTY"])
    auto_refresh_interval = st.select_slider("Auto-refresh Interval (seconds)", options=[0, 30, 60, 120], value=0, help="Set to 0 to disable.")
    strike_range_pct = st.slider("Strike Range (%)", min_value=5, max_value=50, value=15, help="Show strikes within this percentage of the spot price.")

if auto_refresh_interval > 0:
    st_autorefresh(interval=auto_refresh_interval * 1000, key="datarefresh")

if session_token:
    breeze = initialize_breeze(api_key, api_secret, session_token)
    if breeze:
        expiry_map = get_expiry_map(breeze, symbol)
        if expiry_map:
            display_dates = list(expiry_map.keys())
            selected_display_date = st.selectbox("Select Expiry Date", display_dates)
            
            # Use a session state to persist the button click across reruns from auto-refresh
            if st.button("🚀 Load Options Chain"):
                st.session_state.run_load = True

            if 'run_load' in st.session_state and st.session_state.run_load:
                selected_api_date = expiry_map[selected_display_date]
                raw_data, spot_price = get_options_chain_data(breeze, symbol, selected_api_date)
                
                if raw_data and spot_price:
                    full_chain_df = process_and_analyze(raw_data, spot_price)
                    if not full_chain_df.empty:
                        additional_metrics = calculate_additional_metrics(full_chain_df)
                        atm_strike = min(full_chain_df['Strike'], key=lambda x: abs(x - spot_price))
                        
                        lower_bound = spot_price * (1 - strike_range_pct / 100)
                        upper_bound = spot_price * (1 + strike_range_pct / 100)
                        display_chain_df = full_chain_df[(full_chain_df['Strike'] >= lower_bound) & (full_chain_df['Strike'] <= upper_bound)]

                        total_call_oi = display_chain_df['Call OI'].sum()
                        total_put_oi = display_chain_df['Put OI'].sum()
                        pcr_oi = round(total_put_oi / total_call_oi if total_call_oi > 0 else 0, 2)
                        max_call_oi_strike = display_chain_df.loc[display_chain_df['Call OI'].idxmax()]['Strike'] if not display_chain_df['Call OI'].empty else 0
                        max_put_oi_strike = display_chain_df.loc[display_chain_df['Put OI'].idxmax()]['Strike'] if not display_chain_df['Put OI'].empty else 0

                        st.header(f"{symbol} at {spot_price:,.2f}")
                        st.caption(f"Data for expiry: {selected_display_date} | Last updated: {datetime.now().strftime('%I:%M:%S %p')}")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("PCR (Visible Range)", f"{pcr_oi:.2f}")
                        col2.metric("Max Pain (Full Chain)", f"{additional_metrics['max_pain']:,.0f}")
                        col3.metric("Max Call OI Strike", f"{max_call_oi_strike:,.0f}")
                        col4.metric("Max Put OI Strike", f"{max_put_oi_strike:,.0f}")

                        st.dataframe(style_dataframe(display_chain_df, atm_strike, spot_price), use_container_width=True)

                        csv = display_chain_df.to_csv(index=False).encode('utf-8')
                        st.download_button(label="Download Displayed Data as CSV", data=csv, file_name=f"{symbol}_{selected_display_date}_options.csv", mime="text/csv")
                        
                        with st.expander("📊 Show Open Interest Chart", expanded=True):
                            st.plotly_chart(create_oi_chart(display_chain_df, atm_strike), use_container_width=True)
else:
    st.info("👋 Please enter your daily session token in the sidebar to begin.")
