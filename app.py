import streamlit as st
import pandas as pd
from breeze_connect import BreezeConnect
import os
from dotenv import load_dotenv
from datetime import datetime

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

# --- DATA FETCHING & PROCESSING (ROBUST VERSION) ---

@st.cache_data(ttl=3600, show_spinner="Fetching available expiry dates...")
def get_expiry_map(_breeze, symbol):
    """
    Fetches expiry dates and returns a ready-to-use map.
    This is the ROBUST FIX: All date parsing is encapsulated here.
    The main script will never parse dates, preventing the error.
    """
    try:
        spot_data = _breeze.get_quotes(stock_code=symbol, exchange_code="NSE", product_type="cash")
        if not spot_data.get('Success'):
            raise Exception(f"Could not get spot price for expiry lookup: {spot_data.get('Error')}")
        spot_price = float(spot_data['Success'][0]['ltp'])
        
        step = 100 if symbol == "BANKNIFTY" else 50
        nearby_strike = round(spot_price / step) * step

        data = _breeze.get_option_chain_quotes(
            stock_code=symbol, exchange_code="NFO", product_type="options",
            right="Call", expiry_date=None, strike_price=nearby_strike
        )
        if not data.get('Success'):
            raise Exception(f"API Error fetching expiries: {data.get('Error', 'Unknown error')}")
        
        raw_dates = sorted(list(set(item['expiry_date'] for item in data['Success'])))
        
        # Create the display-to-api map with robust date parsing
        expiry_map = {}
        for d in raw_dates:
            try:
                # Try parsing as ISO format first
                if 'T' in d and 'Z' in d:
                    parsed_date = datetime.strptime(d, "%Y-%m-%dT%H:%M:%S.%fZ")
                    display_date = parsed_date.strftime("%d-%b-%Y")
                # Try parsing as simple date format
                elif '-' in d and len(d.split('-')) == 3:
                    try:
                        # Try dd-MMM-yyyy format
                        parsed_date = datetime.strptime(d, "%d-%b-%Y")
                        display_date = parsed_date.strftime("%d-%b-%Y")
                    except ValueError:
                        try:
                            # Try yyyy-mm-dd format
                            parsed_date = datetime.strptime(d, "%Y-%m-%d")
                            display_date = parsed_date.strftime("%d-%b-%Y")
                        except ValueError:
                            # If all parsing fails, use the raw date as both key and value
                            display_date = d
                else:
                    # If format is unrecognized, use raw date
                    display_date = d
                
                expiry_map[display_date] = d
                
            except Exception as parse_error:
                # If any parsing fails, use the raw date as both display and API date
                st.warning(f"Could not parse date format for '{d}': {parse_error}. Using raw format.")
                expiry_map[d] = d
        
        return expiry_map
        
    except Exception as e:
        st.error(f"Could not fetch expiry dates: {e}")
        return {}

@st.cache_data(ttl=15, show_spinner="Fetching latest options chain data...")
def get_options_chain_data(_breeze, symbol, api_expiry_date):
    """Fetches options chain using the raw API expiry date."""
    try:
        spot_data = _breeze.get_quotes(stock_code=symbol, exchange_code="NSE", product_type="cash")
        if not spot_data.get('Success'):
             raise Exception(f"API Error getting spot price: {spot_data.get('Error', 'Unknown error')}")
        spot_price = float(spot_data['Success'][0]['ltp'])

        # Fetch Call options
        call_options_data = _breeze.get_option_chain_quotes(
            stock_code=symbol, exchange_code="NFO", product_type="options",
            right="Call", expiry_date=api_expiry_date
        )
        if not call_options_data.get('Success'):
             raise Exception(f"API Error getting Call options: {call_options_data.get('Error', 'Unknown error')}")
        
        # Fetch Put options
        put_options_data = _breeze.get_option_chain_quotes(
            stock_code=symbol, exchange_code="NFO", product_type="options",
            right="Put", expiry_date=api_expiry_date
        )
        if not put_options_data.get('Success'):
             raise Exception(f"API Error getting Put options: {put_options_data.get('Error', 'Unknown error')}")
        
        # Combine both Call and Put options data
        combined_options_data = call_options_data['Success'] + put_options_data['Success']
        
        return combined_options_data, spot_price
    except Exception as e:
        st.error(f"Failed to fetch options chain: {e}")
        return None, None

# --- DATA ANALYSIS & STYLING ---
def process_and_analyze(raw_data, spot_price):
    df = pd.DataFrame(raw_data).apply(pd.to_numeric, errors='ignore')
    calls_df, puts_df = df[df['right'] == 'Call'], df[df['right'] == 'Put']
    chain = pd.merge(calls_df, puts_df, on="strike_price", suffixes=('_call', '_put'), how="outer").sort_values("strike_price").fillna(0)
    chain = chain[['oi_call', 'oi_change_call', 'ltp_call', 'strike_price', 'ltp_put', 'oi_change_put', 'oi_put']]
    chain.columns = ['Call OI', 'Call Chng OI', 'Call LTP', 'Strike', 'Put LTP', 'Put Chng OI', 'Put OI']
    atm_strike = min(chain['Strike'], key=lambda x: abs(x - spot_price))
    total_call_oi, total_put_oi = chain['Call OI'].sum(), chain['Put OI'].sum()
    pcr_oi = round(total_put_oi / total_call_oi if total_call_oi > 0 else 0, 2)
    max_call_oi_strike = chain.loc[chain['Call OI'].idxmax()]['Strike'] if not chain['Call OI'].empty else 0
    max_put_oi_strike = chain.loc[chain['Put OI'].idxmax()]['Strike'] if not chain['Put OI'].empty else 0
    return chain, {'pcr_oi': pcr_oi, 'max_call_oi_strike': max_call_oi_strike, 'max_put_oi_strike': max_put_oi_strike, 'atm_strike': atm_strike}

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
st.markdown("An enterprise-grade tool for analyzing Nifty options data, powered by the ICICI Breeze API.")

api_key, api_secret = load_credentials()
if not api_key or not api_secret:
    st.error("API_KEY or API_SECRET is not configured. Please set them in your Streamlit secrets or .env file.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Controls")
    session_token = st.text_input("Enter Your Daily Session Token", type="password")
    symbol = st.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY", "FINNIFTY"])

if session_token:
    breeze = initialize_breeze(api_key, api_secret, session_token)
    
    if breeze:
        # 1. Get the pre-processed map directly from the cached function.
        expiry_map = get_expiry_map(breeze, symbol)
        
        if expiry_map:
            # 2. The main script now deals with simple lists and lookups. No parsing!
            display_dates = list(expiry_map.keys())

            with st.form("options_form"):
                selected_display_date = st.selectbox("Select Expiry Date", display_dates)
                load_button = st.form_submit_button("🚀 Load Options Chain")

            if load_button:
                # 3. Look up the raw API date from the map.
                selected_api_date = expiry_map[selected_display_date]
                
                raw_data, spot_price = get_options_chain_data(breeze, symbol, selected_api_date)
                
                if raw_data and spot_price:
                    chain_df, analytics = process_and_analyze(raw_data, spot_price)
                    st.header(f"{symbol} at {spot_price:,.2f}")
                    st.caption(f"Data for expiry: {selected_display_date} | Last updated: {datetime.now().strftime('%I:%M:%S %p')}")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("PCR (OI)", f"{analytics['pcr_oi']:.2f}")
                    col2.metric("Max Call OI Strike", f"{analytics['max_call_oi_strike']:,.0f}")
                    col3.metric("Max Put OI Strike", f"{analytics['max_put_oi_strike']:,.0f}")
                    col4.metric("ATM Strike", f"{analytics['atm_strike']:,.0f}")
                    st.dataframe(style_dataframe(chain_df, analytics['atm_strike'], spot_price), use_container_width=True)
else:
    st.info("👋 Please enter your daily session token in the sidebar to begin.")
