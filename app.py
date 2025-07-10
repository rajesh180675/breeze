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
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Pro Options Analyzer", page_icon="🚀", layout="wide")

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

@st.cache_data(max_entries=1000)
def calculate_iv(option_type, spot, strike, market_price, t, r=0.07):
    if t <= 0 or market_price <= 0: return 0
    objective = lambda vol: abs(black_scholes_price(vol, option_type, spot, strike, t, r) - market_price)
    result = minimize_scalar(objective, bounds=(0.001, 5.0), method='bounded')
    return result.x

def calculate_greeks(iv, option_type, spot, strike, t, r=0.07):
    if iv is None or iv <= 0 or t <= 0: 
        return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0}
    
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
    
    return {'delta': round(delta, 4), 'gamma': round(gamma, 4), 
            'vega': round(vega, 4), 'theta': round(theta, 4)}

# --- DATA FETCHING ---
@st.cache_data(ttl=3600, show_spinner="Fetching expiry dates...")
def get_expiry_map(_breeze, symbol):
    try:
        spot_data = _breeze.get_quotes(stock_code=symbol, exchange_code="NSE", product_type="cash")
        if not spot_data.get('Success'): 
            raise Exception(f"Could not get spot price: {spot_data.get('Error')}")
        
        spot_price = float(spot_data['Success'][0]['ltp'])
        step = 100 if symbol == "BANKNIFTY" else 50
        nearby_strike = round(spot_price / step) * step
        
        data = _breeze.get_option_chain_quotes(
            stock_code=symbol, exchange_code="NFO", product_type="options", 
            right="Call", expiry_date=None, strike_price=nearby_strike
        )
        
        if not data.get('Success'): 
            raise Exception(f"API Error fetching expiries: {data.get('Error')}")
        
        raw_dates = sorted(list(set(item['expiry_date'] for item in data['Success'])))
        expiry_map = {parsed_date.strftime("%d-%b-%Y"): d for d in raw_dates 
                     if (parsed_date := robust_date_parse(d))}
        return expiry_map
    except Exception as e:
        st.error(f"Could not fetch expiry dates: {e}")
        return {}

def get_options_chain_data_with_retry(_breeze, symbol, api_expiry_date, max_retries=3):
    for attempt in range(max_retries):
        try:
            spot_data = _breeze.get_quotes(stock_code=symbol, exchange_code="NSE", product_type="cash")
            if not spot_data.get('Success'): 
                raise Exception(f"API Error getting spot price: {spot_data.get('Error')}")
            spot_price = float(spot_data['Success'][0]['ltp'])
            
            call_data = _breeze.get_option_chain_quotes(
                stock_code=symbol, exchange_code="NFO", product_type="options", 
                right="Call", expiry_date=api_expiry_date
            )
            if not call_data.get('Success'): 
                raise Exception(f"API Error getting Call options: {call_data.get('Error')}")
            
            put_data = _breeze.get_option_chain_quotes(
                stock_code=symbol, exchange_code="NFO", product_type="options", 
                right="Put", expiry_date=api_expiry_date
            )
            if not put_data.get('Success'): 
                raise Exception(f"API Error getting Put options: {put_data.get('Error')}")
            
            st.session_state.last_fetch_time = datetime.now()
            return call_data.get('Success', []) + put_data.get('Success', []), spot_price
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"Failed to fetch options chain after {max_retries} attempts: {e}")
                return None, None
            time.sleep(1 * (2 ** attempt))

# --- DATA ANALYSIS & VISUALIZATION ---
def process_and_analyze(raw_data, spot_price, expiry_date):
    if not raw_data:
        st.warning("No options data received.")
        return pd.DataFrame()
    
    df = pd.DataFrame(raw_data)
    expected = ['oi', 'oi_change', 'ltp', 'volume', 'strike_price', 'right']
    for col in expected:
        if col not in df.columns: df[col] = 0
    df = df.apply(pd.to_numeric, errors='ignore')
    
    calls = df[df['right'] == 'Call']
    puts = df[df['right'] == 'Put']
    chain = pd.merge(calls, puts, on="strike_price", suffixes=('_call', '_put'), how="outer")
    chain = chain.sort_values("strike_price").fillna(0)
    
    # Calculate Time to Expiry in years
    t = max((datetime.strptime(expiry_date, "%d-%b-%Y") - datetime.now()).total_seconds() / (365 * 24 * 3600), 0)
    
    if t > 0:
        # Vectorized IV calculation
        chain['Call IV'] = chain.apply(lambda row: calculate_iv('Call', spot_price, row['strike_price'], row['ltp_call'], t) * 100 if row['ltp_call'] > 0 else 0, axis=1)
        chain['Put IV'] = chain.apply(lambda row: calculate_iv('Put', spot_price, row['strike_price'], row['ltp_put'], t) * 100 if row['ltp_put'] > 0 else 0, axis=1)
        
        # Calculate Greeks
        call_greeks = chain.apply(lambda row: pd.Series(calculate_greeks(row['Call IV']/100, 'Call', spot_price, row['strike_price'], t)), axis=1)
        put_greeks = chain.apply(lambda row: pd.Series(calculate_greeks(row['Put IV']/100, 'Put', spot_price, row['strike_price'], t)), axis=1)
        
        chain = pd.concat([chain, 
                          call_greeks.add_prefix('call_'), 
                          put_greeks.add_prefix('put_')], axis=1)
    
    # Rename columns for display
    chain.rename(columns={
        'oi_call': 'Call OI', 'oi_change_call': 'Call Chng OI', 'ltp_call': 'Call LTP',
        'strike_price': 'Strike', 'ltp_put': 'Put LTP', 'oi_change_put': 'Put Chng OI',
        'oi_put': 'Put OI', 'volume_call': 'Call Volume', 'volume_put': 'Put Volume'
    }, inplace=True)
    
    return chain

def calculate_dashboard_metrics(chain_df, spot_price):
    # Vectorized Max Pain
    strikes = chain_df['Strike'].values
    call_oi = chain_df['Call OI'].values
    put_oi = chain_df['Put OI'].values
    
    strike_matrix = strikes[:, np.newaxis]
    call_pain = np.sum(np.maximum(strike_matrix - strikes, 0) * call_oi, axis=1)
    put_pain = np.sum(np.maximum(strikes - strike_matrix, 0) * put_oi, axis=1)
    total_pain = call_pain + put_pain
    max_pain = strikes[np.argmin(total_pain)] if len(total_pain) > 0 else 0
    
    # PCR and other metrics
    total_call_oi = chain_df['Call OI'].sum()
    total_put_oi = chain_df['Put OI'].sum()
    pcr = round(total_put_oi / total_call_oi if total_call_oi > 0 else 0, 2)
    net_oi_change = chain_df['Put Chng OI'].sum() - chain_df['Call Chng OI'].sum()
    
    # Enhanced Sentiment Score
    sentiment_score = 0
    
    # PCR Analysis
    if pcr > 1.2: sentiment_score += 30
    elif pcr < 0.8: sentiment_score -= 30
    else: sentiment_score += (pcr - 1) * 75
    
    # OI Change Analysis
    if net_oi_change > 0: sentiment_score += 25
    elif net_oi_change < 0: sentiment_score -= 25
    
    # Max Pain Analysis
    if spot_price < max_pain: sentiment_score += 20
    elif spot_price > max_pain: sentiment_score -= 20
    
    # Volume Analysis
    if 'Call Volume' in chain_df.columns and 'Put Volume' in chain_df.columns:
        call_volume = chain_df['Call Volume'].sum()
        put_volume = chain_df['Put Volume'].sum()
        volume_ratio = put_volume / call_volume if call_volume > 0 else 0
        if volume_ratio > 1.1: sentiment_score += 15
        elif volume_ratio < 0.9: sentiment_score -= 15
    
    return {
        'max_pain': max_pain,
        'resistance': chain_df.nlargest(3, 'Call OI')['Strike'].tolist(),
        'support': chain_df.nlargest(3, 'Put OI')['Strike'].tolist(),
        'pcr': pcr,
        'net_oi_change': net_oi_change,
        'sentiment': max(-100, min(100, sentiment_score))
    }

def create_oi_chart(chain_df, atm_strike, spot_price, max_pain=None):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=chain_df['Strike'], y=chain_df['Call OI'], 
                        name='Call OI', marker_color='rgba(239, 83, 80, 0.7)'))
    fig.add_trace(go.Bar(x=chain_df['Strike'], y=chain_df['Put OI'], 
                        name='Put OI', marker_color='rgba(46, 125, 50, 0.7)'))
    
    fig.add_vline(x=spot_price, line_width=2, line_dash="solid", line_color="blue", 
                  annotation_text="Spot", annotation_position="top left")
    fig.add_vline(x=atm_strike, line_width=2, line_dash="dash", line_color="black", 
                  annotation_text="ATM", annotation_position="top right")
    if max_pain:
        fig.add_vline(x=max_pain, line_width=2, line_dash="dot", line_color="purple", 
                      annotation_text="Max Pain")
    
    fig.update_layout(title_text='Open Interest Distribution', xaxis_title='Strike Price', 
                     yaxis_title='Open Interest', barmode='group', height=400, 
                     hovermode='x unified')
    return fig

def create_heatmap(df):
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

def create_iv_smile_chart(chain_df):
    iv_data = []
    for _, row in chain_df.iterrows():
        if row['Call IV'] > 0:
            iv_data.append({'Strike': row['Strike'], 'IV': row['Call IV'], 'Type': 'Call'})
        if row['Put IV'] > 0:
            iv_data.append({'Strike': row['Strike'], 'IV': row['Put IV'], 'Type': 'Put'})
    
    if not iv_data:
        return None
    
    iv_df = pd.DataFrame(iv_data)
    
    fig = go.Figure()
    for option_type in ['Call', 'Put']:
        data = iv_df[iv_df['Type'] == option_type]
        if not data.empty:
            fig.add_trace(go.Scatter(
                x=data['Strike'], 
                y=data['IV'],
                mode='lines+markers',
                name=f'{option_type} IV',
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title='Implied Volatility Smile',
        xaxis_title='Strike Price',
        yaxis_title='Implied Volatility (%)',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_volume_profile(chain_df):
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=chain_df['Strike'],
        y=chain_df['Call Volume'],
        name='Call Volume',
        marker_color='rgba(239, 83, 80, 0.7)'
    ))
    
    fig.add_trace(go.Bar(
        x=chain_df['Strike'],
        y=chain_df['Put Volume'],
        name='Put Volume',
        marker_color='rgba(46, 125, 50, 0.7)'
    ))
    
    fig.update_layout(
        title='Volume Profile',
        xaxis_title='Strike Price',
        yaxis_title='Volume',
        barmode='group',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def display_sentiment_gauge(sentiment_score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=sentiment_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Market Sentiment"},
        delta={'reference': 0},
        gauge={
            'axis': {'range': [-100, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-100, -50], 'color': "darkred"},
                {'range': [-50, -20], 'color': "lightcoral"},
                {'range': [-20, 20], 'color': "lightgray"},
                {'range': [20, 50], 'color': "lightgreen"},
                {'range': [50, 100], 'color': "darkgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def track_historical_data(symbol, expiry, metrics):
    if 'historical_data' not in st.session_state: 
        st.session_state.historical_data = []
    
    st.session_state.historical_data.append({
        'timestamp': datetime.now(), 
        'symbol': symbol, 
        'expiry': expiry, 
        **metrics
    })
    
    if len(st.session_state.historical_data) > 200: 
        st.session_state.historical_data = st.session_state.historical_data[-200:]

# --- MAIN APPLICATION UI ---
def main():
    st.title("🚀 Pro Options & Greeks Analyzer")
    
    # Initialize session state
    if 'last_fetch_time' not in st.session_state:
        st.session_state.last_fetch_time = None
    if 'run_analysis' not in st.session_state:
        st.session_state.run_analysis = False
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Credentials
        with st.expander("🔐 API Credentials", expanded=True):
            api_key, api_secret = load_credentials()
            session_token = st.text_input("Session Token", type="password", 
                                        help="Get from https://api.icicidirect.com/apiuser/login")
        
        # Symbol Selection
        symbol = st.selectbox("📊 Select Symbol", ["NIFTY", "BANKNIFTY", "FINNIFTY"])
        
        # Auto-refresh Settings
        st.subheader("🔄 Auto-Refresh")
        auto_refresh = st.checkbox("Enable Auto-Refresh")
        if auto_refresh:
            refresh_interval = st.slider("Refresh Interval (seconds)", 10, 300, 60)
            st_autorefresh(interval=refresh_interval * 1000, key="datarefresh")
        
        # Display Settings
        st.subheader("📈 Display Options")
        show_greeks = st.checkbox("Show Greeks", value=True)
        show_iv_smile = st.checkbox("Show IV Smile", value=True)
        show_volume = st.checkbox("Show Volume Profile", value=True)
        
        # Export Options
        st.subheader("💾 Export Data")
        export_format = st.selectbox("Export Format", ["JSON", "CSV", "Excel"])
    
    # Main Content Area
    if not session_token:
        st.warning("⚠️ Please enter your session token to proceed")
        st.info("Get your session token from: https://api.icicidirect.com/apiuser/login")
        return
    
    # Initialize Breeze Connection
    breeze = initialize_breeze(api_key, api_secret, session_token)
    if not breeze:
        return
    
    # Fetch Expiry Dates
    expiry_map = get_expiry_map(breeze, symbol)
    if not expiry_map:
        st.error("Failed to fetch expiry dates. Please check your connection.")
        return
    
    # Expiry Selection
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        selected_expiry = st.selectbox("📅 Select Expiry", list(expiry_map.keys()))
        st.session_state.selected_display_date = selected_expiry
    
    with col2:
        if st.session_state.last_fetch_time:
            st.info(f"Last updated: {st.session_state.last_fetch_time.strftime('%H:%M:%S')}")
    
    with col3:
        if st.button("🔄 Refresh Data", type="primary"):
            st.session_state.run_analysis = True
    
    # Fetch and analyze data
    if st.session_state.run_analysis:
        with st.spinner("Fetching options chain data..."):
            api_expiry_date = expiry_map[selected_expiry]
            raw_data, spot_price = get_options_chain_data_with_retry(breeze, symbol, api_expiry_date)
            
            if raw_data and spot_price:
                full_chain_df = process_and_analyze(raw_data, spot_price, selected_expiry)
                
                if not full_chain_df.empty:
                    # Calculate metrics
                    metrics = calculate_dashboard_metrics(full_chain_df, spot_price)
                    atm_strike = full_chain_df.iloc[(full_chain_df['Strike'] - spot_price).abs().argsort()[:1]]['Strike'].values[0]
                    
                    # Track historical data
                    track_historical_data(symbol, selected_expiry, metrics)
                    
                    # Display Key Metrics
                    st.subheader("📊 Key Metrics")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("Spot Price", f"₹{spot_price:,.2f}")
                    with col2:
                        st.metric("ATM Strike", f"₹{atm_strike:,.0f}")
                    with col3:
                        st.metric("Max Pain", f"₹{metrics['max_pain']:,.0f}")
                    with col4:
                        st.metric("PCR", f"{metrics['pcr']:.2f}")
                    with col5:
                        sentiment_text = "Bullish" if metrics['sentiment'] > 20 else "Bearish" if metrics['sentiment'] < -20 else "Neutral"
                        st.metric("Sentiment", sentiment_text, f"{metrics['sentiment']:.0f}")
                    
                    # Sentiment Gauge
                    st.plotly_chart(display_sentiment_gauge(metrics['sentiment']), use_container_width=True)
                    
                    # Support & Resistance Levels
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**Key Resistance Levels:** {', '.join(map(str, metrics['resistance']))}")
                    with col2:
                        st.success(f"**Key Support Levels:** {', '.join(map(str, metrics['support']))}")
                    
                    # Tabs for different views
                    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 OI Analysis", "🔥 Heatmap", "😊 IV Smile", "📈 Volume", "🧮 Greeks", "⏳ History"])
                    
                    with tab1:
                        st.plotly_chart(create_oi_chart(full_chain_df, atm_strike, spot_price, metrics['max_pain']), 
                                      use_container_width=True)
                    
                    with tab2:
                        st.plotly_chart(create_heatmap(full_chain_df), use_container_width=True)
                    
                    with tab3:
                        if show_iv_smile and 'Call IV' in full_chain_df.columns:
                            iv_chart = create_iv_smile_chart(full_chain_df)
                            if iv_chart:
                                st.plotly_chart(iv_chart, use_container_width=True)
                            else:
                                st.info("IV Smile chart not available")
                    
                    with tab4:
                        if show_volume:
                            st.plotly_chart(create_volume_profile(full_chain_df), use_container_width=True)
                    
                    with tab5:
                        if show_greeks and 'call_delta' in full_chain_df.columns:
                            # Display Greeks in a clean table
                            greeks_cols = ['Strike', 'call_delta', 'call_gamma', 'call_vega', 'call_theta', 
                                         'put_delta', 'put_gamma', 'put_vega', 'put_theta']
                            greeks_df = full_chain_df[greeks_cols].copy()
                            greeks_df.columns = ['Strike', 'Call Δ', 'Call Γ', 'Call V', 'Call Θ',
                                               'Put Δ', 'Put Γ', 'Put V', 'Put Θ']
                            
                            # Filter for near ATM strikes
                            atm_idx = (greeks_df['Strike'] - spot_price).abs().idxmin()
                            start_idx = max(0, atm_idx - 5)
                            end_idx = min(len(greeks_df), atm_idx + 6)
                            
                            st.dataframe(
                                greeks_df.iloc[start_idx:end_idx].style.format({
                                    'Strike': '{:.0f}',
                                    'Call Δ': '{:.3f}', 'Call Γ': '{:.4f}', 'Call V': '{:.3f}', 'Call Θ': '{:.3f}',
                                    'Put Δ': '{:.3f}', 'Put Γ': '{:.4f}', 'Put V': '{:.3f}', 'Put Θ': '{:.3f}'
                                }).background_gradient(subset=['Call Δ', 'Put Δ'], cmap='RdYlGn'),
                                use_container_width=True
                            )
                    
                    with tab6:
                        if 'historical_data' in st.session_state and st.session_state.historical_data:
                            hist_df = pd.DataFrame(st.session_state.historical_data)
                            
                            # Sentiment & PCR Trend
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=hist_df['timestamp'], y=hist_df['sentiment'], 
                                                   mode='lines+markers', name='Sentiment', yaxis='y'))
                            fig.add_trace(go.Scatter(x=hist_df['timestamp'], y=hist_df['pcr'], 
                                                   mode='lines+markers', name='PCR', yaxis='y2'))
                            
                            fig.update_layout(
                                title='Historical Sentiment & PCR',
                                xaxis_title='Time',
                                yaxis=dict(title='Sentiment', side='left'),
                                yaxis2=dict(title='PCR', side='right', overlaying='y'),
                                hovermode='x unified',
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Historical data will be tracked during this session.")
                    
                    # Options Chain Table
                    st.subheader("📋 Options Chain Data")
                    
                    # Filters
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        strike_range = st.slider("Strike Range", 
                                               int(full_chain_df['Strike'].min()), 
                                               int(full_chain_df['Strike'].max()),
                                               (int(spot_price - 500), int(spot_price + 500)))
                    with col2:
                        oi_filter = st.number_input("Min OI Filter", value=0, step=1000)
                    with col3:
                        show_itm_only = st.checkbox("Show ITM Only")
                    
                    # Apply filters
                    filtered_df = full_chain_df[
                        (full_chain_df['Strike'] >= strike_range[0]) & 
                        (full_chain_df['Strike'] <= strike_range[1]) &
                        ((full_chain_df['Call OI'] >= oi_filter) | (full_chain_df['Put OI'] >= oi_filter))
                    ]
                    
                    if show_itm_only:
                        filtered_df = filtered_df[
                            ((filtered_df['Strike'] < spot_price) & (filtered_df['Put LTP'] > 0)) |
                            ((filtered_df['Strike'] > spot_price) & (filtered_df['Call LTP'] > 0))
                        ]
                    
                                       # Display columns
                    display_cols = ['Call OI', 'Call Chng OI', 'Call LTP', 'Call Volume', 'Strike', 
                                  'Put LTP', 'Put Volume', 'Put Chng OI', 'Put OI']
                    
                    if 'Call IV' in filtered_df.columns:
                        display_cols.extend(['Call IV', 'Put IV'])
                    
                    # Style the dataframe
                    styled_df = filtered_df[display_cols].style.format({
                        'Call OI': '{:,.0f}',
                        'Call Chng OI': '{:,.0f}',
                        'Call LTP': '{:,.2f}',
                        'Call Volume': '{:,.0f}',
                        'Strike': '{:,.0f}',
                        'Put LTP': '{:,.2f}',
                        'Put Chng OI': '{:,.0f}',
                        'Put OI': '{:,.0f}',
                        'Put Volume': '{:,.0f}',
                        'Call IV': '{:.1f}%',
                        'Put IV': '{:.1f}%'
                    }).background_gradient(subset=['Call OI', 'Put OI'], cmap='YlOrRd')
                    
                    st.dataframe(styled_df, use_container_width=True, height=600)
                    
                    # Export functionality
                    if st.sidebar.button("📥 Export Data"):
                        export_data_dict = {
                            'metadata': {
                                'symbol': symbol,
                                'expiry': selected_expiry,
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'spot_price': spot_price,
                                'metrics': metrics
                            },
                            'chain_data': full_chain_df.to_dict('records')
                        }
                        
                        if export_format == "JSON":
                            json_str = json.dumps(export_data_dict, indent=2, default=str)
                            st.download_button(
                                label="Download JSON",
                                data=json_str,
                                file_name=f"{symbol}_options_chain_{selected_expiry.replace(' ', '_')}.json",
                                mime="application/json"
                            )
                        elif export_format == "CSV":
                            csv = full_chain_df.to_csv(index=False)
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name=f"{symbol}_options_chain_{selected_expiry.replace(' ', '_')}.csv",
                                mime="text/csv"
                            )
                        elif export_format == "Excel":
                            # Create Excel file in memory
                            from io import BytesIO
                            output = BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                full_chain_df.to_excel(writer, sheet_name='Options Chain', index=False)
                                pd.DataFrame([metrics]).to_excel(writer, sheet_name='Metrics', index=False)
                                if 'historical_data' in st.session_state and st.session_state.historical_data:
                                    pd.DataFrame(st.session_state.historical_data).to_excel(
                                        writer, sheet_name='Historical', index=False
                                    )
                            excel_data = output.getvalue()
                            st.download_button(
                                label="Download Excel",
                                data=excel_data,
                                file_name=f"{symbol}_options_chain_{selected_expiry.replace(' ', '_')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                
                else:
                    st.error("No data to display. The options chain might be empty.")
            else:
                st.error("Failed to fetch options data. Please try again.")
    else:
        st.info("👆 Click 'Refresh Data' to load the options chain")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built with ❤️ using Streamlit | Data from ICICI Direct Breeze API</p>
            <p style='font-size: 0.8em; color: gray;'>
                Disclaimer: This tool is for educational purposes only. 
                Please do your own research before making any trading decisions.
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )

# Run the application
if __name__ == "__main__":
    main()
