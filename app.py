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

# --- DATA FETCHING FUNCTIONS & CACHING ---
@st.cache_data(ttl=3600, show_spinner="Fetching available expiry dates...")
def get_expiry_dates(_breeze, symbol):
    """
    Fetches and caches all available expiry dates for 1 hour.
    WORKAROUND: Provides a nearby strike price to satisfy API requirements.
    """
    try:
        # 1. Get current spot price to find a valid nearby strike
        spot_data = _breeze.get_quotes(stock_code=symbol, exchange_code="NSE", product_type="cash")
        if not spot_data.get('Success'):
            raise Exception(f"Could not get spot price for expiry lookup: {spot_data.get('Error')}")
        spot_price = float(spot_data['Success'][0]['ltp'])
        
        # 2. Calculate a valid strike price step (50 for Nifty/Finnifty, 100 for Banknifty)
        step = 100 if symbol == "BANKNIFTY" else 50
        nearby_strike = round(spot_price / step) * step

        # 3. Call API with the nearby_strike to get all expiries for that strike
        data = _breeze.get_option_chain_quotes(
            stock_code=symbol,
            exchange_code="NFO",
            product_type="options",
            right="Call",
            expiry_date=None, # Leave expiry blank
            strike_price=nearby_strike # Provide a valid strike price
        )
        if not data.get('Success'):
            raise Exception(f"API Error: {data.get('Error', 'Unknown error')}")
        
        dates = sorted(list(set(item['expiry_date'] for item in data['Success'])))
        return [datetime.strptime(d, "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%d-%b-%Y") for d in dates]
    except Exception as e:
        st.error(f"Could not fetch expiry dates: {e}")
        return []

@st.cache_data(ttl=15, show_spinner="Fetching latest options chain data...")
def get_options_chain_data(_breeze, symbol, expiry_date):
    """Fetches and caches the full options chain for 15 seconds."""
    try:
        api_expiry_date = datetime.strptime(expiry_date, "%d-%b-%Y").strftime("%Y-%m-%d") + "T06:00:00.000Z"
        
        spot_data = _breeze.get_quotes(stock_code=symbol, exchange_code="NSE", product_type="cash")
        if not spot_data.get('Success'):
             raise Exception(f"API Error getting spot price: {spot_data.get('Error', 'Unknown error')}")
        spot_price = float(spot_data['Success'][0]['ltp'])

        options_data = _breeze.get_option_chain_quotes(
            stock_code=symbol, exchange_code="NFO", product_type="options",
            right=None, expiry_date=api_expiry_date
        )
        if not options_data.get('Success'):
             raise Exception(f"API Error getting options chain: {options_data.get('Error', 'Unknown error')}")
        
        return options_data['Success'], spot_price
    except Exception as e:
        st.error(f"Failed to fetch options chain: {e}")
        return None, None

# --- DATA PROCESSING & ANALYSIS ---
def process_and_analyze(raw_data, spot_price):
    """Processes raw API data into a clean DataFrame and calculates analytics."""
    df = pd.DataFrame(raw_data).apply(pd.to_numeric, errors='ignore')
    
    calls_df = df[df['right'] == 'Call']
    puts_df = df[df['right'] == 'Put']

    chain = pd.merge(calls_df, puts_df, on="strike_price", suffixes=('_call', '_put'), how="outer").sort_values("strike_price").fillna(0)
    
    chain = chain[['oi_call', 'oi_change_call', 'ltp_call', 'strike_price', 'ltp_put', 'oi_change_put', 'oi_put']]
    chain.columns = ['Call OI', 'Call Chng OI', 'Call LTP', 'Strike', 'Put LTP', 'Put Chng OI', 'Put OI']

    atm_strike = min(chain['Strike'], key=lambda x: abs(x - spot_price))
    
    total_call_oi = chain['Call OI'].sum()
    total_put_oi = chain['Put OI'].sum()
    pcr_oi = round(total_put_oi / total_call_oi if total_call_oi > 0 else 0, 2)
    max_call_oi_strike = chain.loc[chain['Call OI'].idxmax()]['Strike'] if not chain['Call OI'].empty else 0
    max_put_oi_strike = chain.loc[chain['Put OI'].idxmax()]['Strike'] if not chain['Put OI'].empty else 0

    analytics = {
        'pcr_oi': pcr_oi, 'max_call_oi_strike': max_call_oi_strike,
        'max_put_oi_strike': max_put_oi_strike, 'atm_strike': atm_strike
    }
    
    return chain, analytics

def style_dataframe(df, atm_strike, spot_price):
    """Applies conditional formatting to the DataFrame."""
    def style_row(row):
        styles = [''] * len(row)
        # Highlight ITM Calls
        if row.Strike < spot_price:
            styles[0:3] = ['background-color: #e8f5e9'] * 3 # Light green for Calls
        # Highlight ITM Puts
        if row.Strike > spot_price:
            styles[4:7] = ['background-color: #ffebee'] * 3 # Light red for Puts
        # Highlight ATM strike (overwrites other styles)
        if row.Strike == atm_strike:
            styles = ['background-color: #ffffc0'] * len(row) # Light yellow for ATM
        return styles

    return df.style.apply(style_row, axis=1).format({
        'Call OI': '{:,.0f}', 'Call Chng OI': '{:,.0f}', 'Call LTP': '{:,.2f}',
        'Strike': '{:,.0f}',
        'Put LTP': '{:,.2f}', 'Put Chng OI': '{:,.0f}', 'Put OI': '{:,.0f}'
    })

# --- MAIN APPLICATION UI ---
st.title("📈 Nifty Options Chain Analyzer")
st.markdown("An enterprise-grade tool for analyzing Nifty options data, powered by the ICICI Breeze API.")

# Load API Key and Secret
api_key, api_secret = load_credentials()
if not api_key or not api_secret:
    st.error("API_KEY or API_SECRET is not configured. Please set them in your Streamlit secrets or .env file.")
    st.stop()

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("⚙️ Controls")
    
    session_token = st.text_input("Enter Your Daily Session Token", type="password")
    symbol = st.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY", "FINNIFTY"])

# --- Main App Logic ---
if session_token:
    breeze = initialize_breeze(api_key, api_secret, session_token)
    
    if breeze:
        expiry_dates = get_expiry_dates(breeze, symbol)
        
        if expiry_dates:
            with st.form("options_form"):
                selected_expiry = st.selectbox("Select Expiry Date", expiry_dates)
                load_button = st.form_submit_button("🚀 Load Options Chain")

            if load_button:
                raw_data, spot_price = get_options_chain_data(breeze, symbol, selected_expiry)
                
                if raw_data and spot_price:
                    chain_df, analytics = process_and_analyze(raw_data, spot_price)
                    
                    st.header(f"{symbol} at {spot_price:,.2f}")
                    st.caption(f"Data for expiry: {selected_expiry} | Last updated: {datetime.now().strftime('%I:%M:%S %p')}")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("PCR (OI)", f"{analytics['pcr_oi']:.2f}")
                    col2.metric("Max Call OI Strike", f"{analytics['max_call_oi_strike']:,.0f}")
                    col3.metric("Max Put OI Strike", f"{analytics['max_put_oi_strike']:,.0f}")
                    col4.metric("ATM Strike", f"{analytics['atm_strike']:,.0f}")
                    
                    st.dataframe(style_dataframe(chain_df, analytics['atm_strike'], spot_price), use_container_width=True)
else:
    st.info("👋 Please enter your daily session token in the sidebar to begin.")
