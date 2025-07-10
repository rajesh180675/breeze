import streamlit as st
import pandas as pd
from breeze_connect import BreezeConnect
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import time
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import json
import logging
from io import BytesIO
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
@dataclass
class AppConfig:
    """Centralized configuration management"""
    SYMBOLS: List[str] = None
    STRIKE_STEPS: Dict[str, int] = None
    DEFAULT_RISK_FREE_RATE: float = 0.07
    MAX_RETRIES: int = 3
    CACHE_TTL: int = 3600
    MAX_HISTORICAL_RECORDS: int = 200
    
    def __post_init__(self):
        if self.SYMBOLS is None:
            self.SYMBOLS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"]
        if self.STRIKE_STEPS is None:
            self.STRIKE_STEPS = {
                "BANKNIFTY": 100, 
                "NIFTY": 50, 
                "FINNIFTY": 50,
                "MIDCPNIFTY": 25,
                "SENSEX": 100
            }
    
    @classmethod
    def get_strike_step(cls, symbol: str) -> int:
        config = cls()
        return config.STRIKE_STEPS.get(symbol, 50)

# Initialize configuration
config = AppConfig()

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Pro Options Analyzer", 
    page_icon="🚀", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM EXCEPTIONS ---
class BreezeAPIError(Exception):
    """Custom exception for Breeze API errors"""
    pass

# --- HELPER & SETUP FUNCTIONS ---
def load_credentials() -> Tuple[str, str]:
    """Load API credentials from secrets or environment"""
    if 'BREEZE_API_KEY' in st.secrets:
        return st.secrets["BREEZE_API_KEY"], st.secrets["BREEZE_API_SECRET"]
    else:
        load_dotenv()
        return os.getenv("BREEZE_API_KEY"), os.getenv("BREEZE_API_SECRET")

def handle_api_error(response: Dict[str, Any]) -> List[Dict]:
    """Centralized API error handling"""
    if not response.get('Success'):
        error_msg = response.get('Error', 'Unknown API error')
        if 'session' in error_msg.lower():
            raise BreezeAPIError("Session expired. Please refresh your session token.")
        elif 'rate limit' in error_msg.lower():
            raise BreezeAPIError("Rate limit exceeded. Please wait before retrying.")
        else:
            raise BreezeAPIError(f"API Error: {error_msg}")
    return response['Success']

@st.cache_resource(show_spinner="Connecting to Breeze API...")
def initialize_breeze(api_key: str, api_secret: str, session_token: str) -> Optional[BreezeConnect]:
    """Initialize Breeze API connection"""
    try:
        logger.info("Initializing Breeze connection")
        breeze = BreezeConnect(api_key=api_key)
        breeze.generate_session(api_secret=api_secret, session_token=session_token)
        st.success("API Connection Successful!")
        return breeze
    except Exception as e:
        logger.error(f"Failed to initialize Breeze: {e}")
        st.error(f"Connection Failed: {e}")
        return None

def robust_date_parse(date_string: str) -> Optional[datetime]:
    """Parse dates in multiple formats"""
    formats = [
        "%Y-%m-%dT%H:%M:%S.%fZ", 
        "%d-%b-%Y", 
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%Y%m%d"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except (ValueError, TypeError):
            continue
    return None

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names from Breeze API response"""
    column_mapping = {
        'open_interest': 'oi',
        'openInterest': 'oi',
        'open_int': 'oi',
        'oi_change': 'oi_change',
        'change_oi': 'oi_change',
        'changeInOI': 'oi_change',
        'last_traded_price': 'ltp',
        'lastPrice': 'ltp',
        'last_price': 'ltp',
        'total_qty_traded': 'volume',
        'totalTradedVolume': 'volume',
        'traded_volume': 'volume',
        'volume_traded': 'volume',
        'strike': 'strike_price',
        'strikePrice': 'strike_price',
        'option_type': 'right',
        'optionType': 'right',
        'call_put': 'right'
    }
    
    # Rename columns based on mapping
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    df.rename(columns=column_mapping, inplace=True)
    
    # Ensure required columns exist with default values
    required_columns = ['oi', 'oi_change', 'ltp', 'volume', 'strike_price', 'right']
    for col in required_columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found, creating with default value 0")
            df[col] = 0
    
    return df

def validate_option_data(df: pd.DataFrame) -> bool:
    """Validate option chain data integrity"""
    required_cols = ['strike_price', 'ltp', 'oi', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.warning(f"Missing columns in data: {missing_cols}")
        return False
    
    # Check for data quality
    if df['ltp'].isna().all() or (df['ltp'] == 0).all():
        st.warning("No valid LTP data found")
        return False
    
    return True

# --- GREEKS & IV CALCULATION ---
def black_scholes_price(volatility: float, option_type: str, spot: float, 
                       strike: float, t: float, r: float) -> float:
    """Calculate Black-Scholes option price"""
    if t <= 0 or volatility <= 0:
        return 0
    
    try:
        d1 = (np.log(spot / strike) + (r + 0.5 * volatility**2) * t) / (volatility * np.sqrt(t))
        d2 = d1 - volatility * np.sqrt(t)
        
        if option_type == 'Call':
            return spot * norm.cdf(d1) - strike * np.exp(-r * t) * norm.cdf(d2)
        else:
            return strike * np.exp(-r * t) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    except:
        return 0

@st.cache_data(max_entries=1000)
def calculate_iv(option_type: str, spot: float, strike: float, 
                market_price: float, t: float, r: float = 0.07) -> float:
    """Calculate implied volatility using optimization"""
    if t <= 0 or market_price <= 0 or spot <= 0 or strike <= 0:
        return 0
    
    try:
        objective = lambda vol: abs(black_scholes_price(vol, option_type, spot, strike, t, r) - market_price)
        result = minimize_scalar(objective, bounds=(0.001, 5.0), method='bounded')
        return result.x
    except:
        return 0

def calculate_greeks_vectorized(iv_array: np.ndarray, option_type: str, spot: float, 
                               strikes: np.ndarray, t: float, r: float = 0.07) -> pd.DataFrame:
    """Vectorized Greeks calculation for better performance"""
    iv_array = np.array(iv_array)
    strikes = np.array(strikes)
    
    # Initialize results
    results = pd.DataFrame(index=range(len(strikes)), 
                          columns=['delta', 'gamma', 'vega', 'theta', 'rho'])
    results.fillna(0, inplace=True)
    
    # Handle edge cases
    mask = (iv_array > 0) & (t > 0) & (strikes > 0)
    if not mask.any():
        return results
    
    # Vectorized calculations
    valid_iv = iv_array[mask]
    valid_strikes = strikes[mask]
    
    try:
        d1 = (np.log(spot / valid_strikes) + (r + 0.5 * valid_iv**2) * t) / (valid_iv * np.sqrt(t))
        d2 = d1 - valid_iv * np.sqrt(t)
        
        gamma = norm.pdf(d1) / (spot * valid_iv * np.sqrt(t))
        vega = spot * norm.pdf(d1) * np.sqrt(t) / 100
        
        if option_type == 'Call':
            delta = norm.cdf(d1)
            theta = (-spot * norm.pdf(d1) * valid_iv / (2 * np.sqrt(t)) - 
                     r * valid_strikes * np.exp(-r * t) * norm.cdf(d2)) / 365
            rho = valid_strikes * t * np.exp(-r * t) * norm.cdf(d2) / 100
        else:
            delta = norm.cdf(d1) - 1
            theta = (-spot * norm.pdf(d1) * valid_iv / (2 * np.sqrt(t)) + 
                     r * valid_strikes * np.exp(-r * t) * norm.cdf(-d2)) / 365
            rho = -valid_strikes * t * np.exp(-r * t) * norm.cdf(-d2) / 100
        
        results.loc[mask, 'delta'] = delta
        results.loc[mask, 'gamma'] = gamma
        results.loc[mask, 'vega'] = vega
        results.loc[mask, 'theta'] = theta
        results.loc[mask, 'rho'] = rho
    except Exception as e:
        logger.error(f"Error calculating Greeks: {e}")
    
    return results.round(4)

# --- DATA FETCHING ---
@st.cache_data(ttl=config.CACHE_TTL, show_spinner="Fetching expiry dates...")
def get_expiry_map(_breeze: BreezeConnect, symbol: str) -> Dict[str, str]:
    """Fetch available expiry dates for the symbol"""
    try:
        logger.info(f"Fetching expiry dates for {symbol}")
        
        # Get spot price
        spot_data = _breeze.get_quotes(stock_code=symbol, exchange_code="NSE", product_type="cash")
        spot_data = handle_api_error(spot_data)
        spot_price = float(spot_data[0]['ltp'])
        
        # Calculate nearby strike
        step = config.get_strike_step(symbol)
        nearby_strike = round(spot_price / step) * step
        
        # Get option chain for nearby strike
        data = _breeze.get_option_chain_quotes(
            stock_code=symbol, exchange_code="NFO", product_type="options", 
            right="Call", expiry_date=None, strike_price=nearby_strike
        )
        data = handle_api_error(data)
        
        # Parse expiry dates
        raw_dates = sorted(list(set(item['expiry_date'] for item in data)))
        expiry_map = {}
        
        for d in raw_dates:
            parsed_date = robust_date_parse(d)
            if parsed_date and parsed_date > datetime.now():
                expiry_map[parsed_date.strftime("%d-%b-%Y")] = d
        
        logger.info(f"Found {len(expiry_map)} expiry dates")
        return expiry_map
        
    except BreezeAPIError as e:
        st.error(str(e))
        return {}
    except Exception as e:
        logger.error(f"Error fetching expiry dates: {e}")
        st.error(f"Could not fetch expiry dates: {e}")
        return {}

def fetch_data_with_progress(_breeze: BreezeConnect, symbol: str, 
                           api_expiry_date: str) -> Tuple[Optional[List], Optional[float]]:
    """Fetch options chain data with progress indicator"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Fetch spot price
        status_text.text("Fetching spot price...")
        progress_bar.progress(25)
        
        spot_data = _breeze.get_quotes(stock_code=symbol, exchange_code="NSE", product_type="cash")
        spot_data = handle_api_error(spot_data)
        spot_price = float(spot_data[0]['ltp'])
        
        # Fetch call options
        status_text.text("Fetching call options...")
        progress_bar.progress(50)
        
        call_data = _breeze.get_option_chain_quotes(
            stock_code=symbol, exchange_code="NFO", product_type="options", 
            right="Call", expiry_date=api_expiry_date
        )
        call_data = handle_api_error(call_data)
        
        # Fetch put options
        status_text.text("Fetching put options...")
        progress_bar.progress(75)
        
        put_data = _breeze.get_option_chain_quotes(
            stock_code=symbol, exchange_code="NFO", product_type="options", 
            right="Put", expiry_date=api_expiry_date
        )
        put_data = handle_api_error(put_data)
        
        status_text.text("Complete!")
        progress_bar.progress(100)
        
        # Clear progress indicators
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        st.session_state.last_fetch_time = datetime.now()
        return call_data + put_data, spot_price
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        raise e

def get_options_chain_data_with_retry(_breeze: BreezeConnect, symbol: str, 
                                    api_expiry_date: str, max_retries: int = 3) -> Tuple[Optional[List], Optional[float]]:
    """Fetch options chain data with retry logic"""
    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching options chain for {symbol}, attempt {attempt + 1}")
            return fetch_data_with_progress(_breeze, symbol, api_expiry_date)
        except BreezeAPIError as e:
            st.error(str(e))
            return None, None
        except Exception as e:
            logger.error(f"Error fetching data (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                st.error(f"Failed to fetch options chain after {max_retries} attempts: {e}")
                return None, None
            time.sleep(1 * (2 ** attempt))

# --- DATA ANALYSIS & VISUALIZATION ---
def process_and_analyze(raw_data: List[Dict], spot_price: float, expiry_date: str) -> pd.DataFrame:
    """Process raw options data and calculate Greeks"""
    if not raw_data:
        st.warning("No options data received.")
        return pd.DataFrame()
    
    df = pd.DataFrame(raw_data)
    
    # Normalize column names first
    df = normalize_column_names(df)
    
    # Validate data after normalization
    if not validate_option_data(df):
        return pd.DataFrame()
    
    # Convert to numeric
    numeric_columns = ['oi', 'oi_change', 'ltp', 'volume', 'strike_price']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Separate calls and puts
    calls = df[df['right'].str.upper() == 'CALL'].copy()
    puts = df[df['right'].str.upper() == 'PUT'].copy()
    
    # Merge into chain
    chain = pd.merge(calls, puts, on="strike_price", suffixes=('_call', '_put'), how="outer")
    chain = chain.sort_values("strike_price").fillna(0)
    
    # Calculate Time to Expiry in years
    t = max((datetime.strptime(expiry_date, "%d-%b-%Y") - datetime.now()).total_seconds() / (365 * 24 * 3600), 0)
    
    if t > 0:
        # Vectorized IV calculation
        chain['Call IV'] = chain.apply(
            lambda row: calculate_iv('Call', spot_price, row['strike_price'], 
                                   row['ltp_call'], t) * 100 if row['ltp_call'] > 0 else 0, 
            axis=1
        )
        chain['Put IV'] = chain.apply(
            lambda row: calculate_iv('Put', spot_price, row['strike_price'], 
                                   row['ltp_put'], t) * 100 if row['ltp_put'] > 0 else 0, 
            axis=1
        )
        
        # Calculate Greeks using vectorized function
        strikes = chain['strike_price'].values
        call_ivs = chain['Call IV'].values / 100
        put_ivs = chain['Put IV'].values / 100
        
        call_greeks = calculate_greeks_vectorized(call_ivs, 'Call', spot_price, strikes, t)
        put_greeks = calculate_greeks_vectorized(put_ivs, 'Put', spot_price, strikes, t)
        
        # Add Greeks to chain
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

def calculate_dashboard_metrics(chain_df: pd.DataFrame, spot_price: float) -> Dict[str, Any]:
    """Calculate key metrics from options chain"""
    # Vectorized Max Pain calculation
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
    if pcr > 1.2:
        sentiment_score += 30
    elif pcr < 0.8:
        sentiment_score -= 30
    else:
        sentiment_score += (pcr - 1) * 75
    
    # OI Change Analysis
    if net_oi_change > 0:
        sentiment_score += 25
    elif net_oi_change < 0:
        sentiment_score -= 25
    
    # Max Pain Analysis
    if spot_price < max_pain:
        sentiment_score += 20
    elif spot_price > max_pain:
        sentiment_score -= 20
    
    # Volume Analysis
    if 'Call Volume' in chain_df.columns and 'Put Volume' in chain_df.columns:
        call_volume = chain_df['Call Volume'].sum()
        put_volume = chain_df['Put Volume'].sum()
        volume_ratio = put_volume / call_volume if call_volume > 0 else 0
        if volume_ratio > 1.1:
            sentiment_score += 15
        elif volume_ratio < 0.9:
            sentiment_score -= 15
    
    # IV Skew Analysis
    if 'Call IV' in chain_df.columns and 'Put IV' in chain_df.columns:
        atm_idx = (chain_df['Strike'] - spot_price).abs().idxmin()
        if atm_idx > 0 and atm_idx < len(chain_df) - 1:
            call_iv_skew = chain_df.loc[atm_idx, 'Call IV'] - chain_df['Call IV'].mean()
            put_iv_skew = chain_df.loc[atm_idx, 'Put IV'] - chain_df['Put IV'].mean()
            if put_iv_skew > call_iv_skew:
                sentiment_score += 10
            else:
                sentiment_score -= 10
    
    return {
        'max_pain': max_pain,
        'resistance': chain_df.nlargest(3, 'Call OI')['Strike'].tolist(),
        'support': chain_df.nlargest(3, 'Put OI')['Strike'].tolist(),
        'pcr': pcr,
        'net_oi_change': net_oi_change,
        'sentiment': max(-100, min(100, sentiment_score)),
        'total_call_oi': total_call_oi,
        'total_put_oi': total_put_oi
    }

def create_oi_chart(chain_df: pd.DataFrame, atm_strike: float, spot_price: float, 
                   max_pain: Optional[float] = None) -> go.Figure:
    """Create Open Interest distribution chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=chain_df['Strike'], 
        y=chain_df['Call OI'], 
        name='Call OI', 
        marker_color='rgba(239, 83, 80, 0.7)',
        hovertemplate='Strike: %{x}<br>Call OI: %{y:,.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        x=chain_df['Strike'], 
        y=chain_df['Put OI'], 
        name='Put OI', 
        marker_color='rgba(46, 125, 50, 0.7)',
        hovertemplate='Strike: %{x}<br>Put OI: %{y:,.0f}<extra></extra>'
    ))
    
    # Add reference lines
    fig.add_vline(x=spot_price, line_width=2, line_dash="solid", line_color="blue", 
                  annotation_text="Spot", annotation_position="top left")
    fig.add_vline(x=atm_strike, line_width=2, line_dash="dash", line_color="black", 
                  annotation_text="ATM", annotation_position="top right")
    if max_pain:
        fig.add_vline(x=max_pain, line_width=2, line_dash="dot", line_color="purple", 
                      annotation_text="Max Pain")
    
    fig.update_layout(
        title_text='Open Interest Distribution', 
        xaxis_title='Strike Price', 
        yaxis_title='Open Interest', 
        barmode='group', 
        height=400, 
        hovermode='x unified',
        showlegend=True,
        legend=dict(x=0.7, y=0.95)
    )
    
    return fig

def create_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create premium heatmap"""
    heat_df = df.set_index('Strike')[['Call LTP', 'Put LTP']].sort_index(ascending=False)
    
    fig = go.Figure(data=go.Heatmap(
        z=heat_df.values,
        x=heat_df.columns,
        y=heat_df.index,
        colorscale="Viridis",
        hovertemplate='Strike: %{y}<br>Type: %{x}<br>Premium: %{z:,.2f}<extra></extra>',
        colorbar=dict(title="Premium")
    ))
    
    fig.update_layout(
        title_text='Premium Heatmap', 
        yaxis_title='Strike Price', 
        height=500,
        xaxis=dict(side='top')
    )
    
    return fig

def create_iv_smile_chart(chain_df: pd.DataFrame, spot_price: float) -> Optional[go.Figure]:
    """Create IV smile chart"""
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
                line=dict(width=2),
                hovertemplate='Strike: %{x}<br>IV: %{y:.1f}%<extra></extra>'
            ))
    
    # Add spot price line
    fig.add_vline(x=spot_price, line_width=1, line_dash="dash", line_color="gray", 
                  annotation_text="Spot")
    
    fig.update_layout(
        title='Implied Volatility Smile',
        xaxis_title='Strike Price',
        yaxis_title='Implied Volatility (%)',
        height=400,
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def create_volume_profile(chain_df: pd.DataFrame) -> go.Figure:
    """Create volume profile chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=chain_df['Strike'],
        y=chain_df['Call Volume'],
        name='Call Volume',
        marker_color='rgba(239, 83, 80, 0.7)',
        hovertemplate='Strike: %{x}<br>Call Volume: %{y:,.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        x=chain_df['Strike'],
        y=chain_df['Put Volume'],
        name='Put Volume',
        marker_color='rgba(46, 125, 50, 0.7)',
        hovertemplate='Strike: %{x}<br>Put Volume: %{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Volume Profile',
        xaxis_title='Strike Price',
        yaxis_title='Volume',
        barmode='group',
        height=400,
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def display_sentiment_gauge(sentiment_score: float) -> go.Figure:
    """Create sentiment gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=sentiment_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Market Sentiment", 'font': {'size': 24}},
        delta={'reference': 0, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge={
            'axis': {'range': [-100, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
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

def create_greeks_surface(chain_df: pd.DataFrame, greek: str, option_type: str) -> go.Figure:
    """Create 3D surface plot for Greeks"""
    greek_col = f"{option_type.lower()}_{greek}"
    if greek_col not in chain_df.columns:
        return None
    
    # Create meshgrid for surface plot
    strikes = chain_df['Strike'].values
    greek_values = chain_df[greek_col].values
    
    fig = go.Figure(data=[go.Scatter3d(
        x=strikes,
        y=[1] * len(strikes),  # Single expiry
        z=greek_values,
        mode='markers+lines',
        marker=dict(size=5, color=greek_values, colorscale='Viridis'),
        line=dict(color='darkblue', width=2),
        name=f'{option_type} {greek.capitalize()}'
    )])
    
    fig.update_layout(
        title=f'{option_type} {greek.capitalize()} Profile',
        scene=dict(
            xaxis_title='Strike Price',
            yaxis_title='Time',
            zaxis_title=greek.capitalize()
        ),
        height=500
    )
    
    return fig

def track_historical_data_efficient(symbol: str, expiry: str, metrics: Dict[str, Any]) -> None:
    """Efficient historical data tracking with compression"""
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = pd.DataFrame()
    
    new_row = pd.DataFrame([{
        'timestamp': datetime.now(),
        'symbol': symbol,
        'expiry': expiry,
        **metrics
    }])
    
    st.session_state.historical_data = pd.concat([
        st.session_state.historical_data, 
        new_row
    ], ignore_index=True).tail(config.MAX_HISTORICAL_RECORDS)

def prepare_export_data(df: pd.DataFrame, format_type: str) -> Optional[pd.DataFrame]:
    """Prepare and validate data for export"""
    if df.empty:
        st.error("No data to export")
        return None
    
    # Remove any infinite or NaN values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    # Format based on export type
    if format_type == "Excel":
        # Ensure numeric columns are properly formatted
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].round(2)
    
    return df

def create_strategy_payoff(chain_df: pd.DataFrame, spot_price: float) -> go.Figure:
    """Create strategy payoff diagram"""
    strikes = chain_df['Strike'].values
    
    # Example: Long Straddle at ATM
    atm_idx = (chain_df['Strike'] - spot_price).abs().idxmin()
    atm_strike = chain_df.loc[atm_idx, 'Strike']
    call_premium = chain_df.loc[atm_idx, 'Call LTP']
    put_premium = chain_df.loc[atm_idx, 'Put LTP']
    
    # Calculate payoff
    spot_range = np.linspace(strikes.min(), strikes.max(), 100)
    straddle_payoff = np.maximum(spot_range - atm_strike, 0) + np.maximum(atm_strike - spot_range, 0) - (call_premium + put_premium)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=spot_range,
        y=straddle_payoff,
        mode='lines',
        name='Long Straddle',
        line=dict(width=3)
    ))
    
    # Add breakeven lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=spot_price, line_dash="dash", line_color="blue", annotation_text="Current Spot")
    
    fig.update_layout(
        title=f'Long Straddle Payoff (Strike: {atm_strike})',
        xaxis_title='Spot Price at Expiry',
        yaxis_title='Profit/Loss',
        height=400,
        hovermode='x unified'
    )
    
    return fig

# --- MAIN APPLICATION UI ---
def main():
    st.title("🚀 Pro Options & Greeks Analyzer")
    
    # Initialize session state
    if 'last_fetch_time' not in st.session_state:
        st.session_state.last_fetch_time = None
    if 'run_analysis' not in st.session_state:
        st.session_state.run_analysis = False
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Credentials
        with st.expander("🔐 API Credentials", expanded=True):
            api_key, api_secret = load_credentials()
            session_token = st.text_input("Session Token", type="password", 
                                        help="Get from https://api.icicidirect.com/apiuser/login")
        
        # Symbol Selection
        symbol = st.selectbox("📊 Select Symbol", config.SYMBOLS)
        
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
        show_strategy = st.checkbox("Show Strategy Analysis", value=False)
        
        # Risk Parameters
        st.subheader("⚡ Risk Parameters")
        risk_free_rate = st.number_input("Risk-Free Rate (%)", value=7.0, step=0.1) / 100
        
        # Export Options
        st.subheader("💾 Export Data")
        export_format = st.selectbox("Export Format", ["JSON", "CSV", "Excel"])
    
    # Main Content Area
    if not session_token:
        st.warning("⚠️ Please enter your session token to proceed")
        st.info("Get your session token from: https://api.icicidirect.com/apiuser/login")
        with st.expander("📖 How to get Session Token"):
            st.markdown("""
            1. Visit https://api.icicidirect.com/apiuser/login
            2. Login with your ICICI Direct credentials
            3. Copy the session token from the response
            4. Paste it in the Session Token field
            """)
        return
    
    # Initialize Breeze Connection
    breeze = initialize_breeze(api_key, api_secret, session_token)
    if not breeze:
        return
    
    # Fetch Expiry Dates
    try:
        expiry_map = get_expiry_map(breeze, symbol)
        if not expiry_map:
            st.error("Failed to fetch expiry dates. Please check your connection.")
            return
    except BreezeAPIError as e:
        st.error(str(e))
        return
    
    # Expiry Selection
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        selected_expiry = st.selectbox("📅 Select Expiry", list(expiry_map.keys()))
        st.session_state.selected_display_date = selected_expiry
    
    with col2:
        if st.session_state.last_fetch_time:
            time_diff = (datetime.now() - st.session_state.last_fetch_time).seconds
            st.info(f"Last updated: {st.session_state.last_fetch_time.strftime('%H:%M:%S')} ({time_diff}s ago)")
    
    with col3:
        if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
            st.session_state.run_analysis = True
    
    # Fetch and analyze data
    if st.session_state.run_analysis or auto_refresh:
        try:
            api_expiry_date = expiry_map[selected_expiry]
            raw_data, spot_price = get_options_chain_data_with_retry(breeze, symbol, api_expiry_date)
            
            if raw_data and spot_price:
                full_chain_df = process_and_analyze(raw_data, spot_price, selected_expiry)
                
                if not full_chain_df.empty:
                    # Calculate metrics
                    metrics = calculate_dashboard_metrics(full_chain_df, spot_price)
                    atm_strike = full_chain_df.iloc[(full_chain_df['Strike'] - spot_price).abs().argsort()[:1]]['Strike'].values[0]
                    
                    # Track historical data
                    track_historical_data_efficient(symbol, selected_expiry, metrics)
                    
                    # Display Key Metrics
                    st.subheader("📊 Key Metrics Dashboard")
                    
                    # First row of metrics
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    
                    with col1:
                        st.metric("Spot Price", f"₹{spot_price:,.2f}")
                    with col2:
                        st.metric("ATM Strike", f"₹{atm_strike:,.0f}")
                    with col3:
                        st.metric("Max Pain", f"₹{metrics['max_pain']:,.0f}")
                    with col4:
                        st.metric("PCR", f"{metrics['pcr']:.2f}")
                    with col5:
                        net_oi_delta = f"{metrics['net_oi_change']:+,.0f}"
                        st.metric("Net OI Δ", net_oi_delta)
                    with col6:
                        sentiment_text = "Bullish" if metrics['sentiment'] > 20 else "Bearish" if metrics['sentiment'] < -20 else "Neutral"
                        st.metric("Sentiment", sentiment_text, f"{metrics['sentiment']:.0f}")
                    
                    # Sentiment Gauge
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.plotly_chart(display_sentiment_gauge(metrics['sentiment']), use_container_width=True)
                    
                    with col2:
                        # Support & Resistance Levels
                        st.info(f"**🔴 Key Resistance:** {', '.join(map(str, metrics['resistance']))}")
                        st.success(f"**🟢 Key Support:** {', '.join(map(str, metrics['support']))}")
                        
                        # Additional insights
                        days_to_expiry = (datetime.strptime(selected_expiry, "%d-%b-%Y") - datetime.now()).days
                        st.warning(f"**📅 Days to Expiry:** {days_to_expiry}")
                    
                    # Tabs for different views
                    tabs = ["📊 OI Analysis", "🔥 Heatmap", "😊 IV Analysis", "📈 Volume", "🧮 Greeks", "📉 Strategy", "⏳ History"]
                    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(tabs)
                    
                    with tab1:
                        st.plotly_chart(create_oi_chart(full_chain_df, atm_strike, spot_price, metrics['max_pain']), 
                                      use_container_width=True)
                        
                        # OI Change Analysis
                        oi_change_df = full_chain_df[['Strike', 'Call Chng OI', 'Put Chng OI']].copy()
                        oi_change_df = oi_change_df[(oi_change_df['Call Chng OI'] != 0) | (oi_change_df['Put Chng OI'] != 0)]
                        
                        if not oi_change_df.empty:
                            fig_oi_change = go.Figure()
                            fig_oi_change.add_trace(go.Bar(x=oi_change_df['Strike'], y=oi_change_df['Call Chng OI'], 
                                                          name='Call OI Change', marker_color='red'))
                            fig_oi_change.add_trace(go.Bar(x=oi_change_df['Strike'], y=oi_change_df['Put Chng OI'], 
                                                          name='Put OI Change', marker_color='green'))
                            fig_oi_change.update_layout(title='Open Interest Changes', barmode='group', height=300)
                            st.plotly_chart(fig_oi_change, use_container_width=True)
                    
                    with tab2:
                        st.plotly_chart(create_heatmap(full_chain_df), use_container_width=True)
                    
                    with tab3:
                        if show_iv_smile and 'Call IV' in full_chain_df.columns:
                            iv_chart = create_iv_smile_chart(full_chain_df, spot_price)
                            if iv_chart:
                                st.plotly_chart(iv_chart, use_container_width=True)
                                
                                # IV Statistics
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Avg Call IV", f"{full_chain_df['Call IV'].mean():.1f}%")
                                    st.metric("ATM Call IV", f"{full_chain_df.loc[full_chain_df['Strike'] == atm_strike, 'Call IV'].values[0]:.1f}%")
                                with col2:
                                    st.metric("Avg Put IV", f"{full_chain_df['Put IV'].mean():.1f}%")
                                    st.metric("ATM Put IV", f"{full_chain_df.loc[full_chain_df['Strike'] == atm_strike, 'Put IV'].values[0]:.1f}%")
                            else:
                                st.info("IV Smile chart not available")
                    
                    with tab4:
                        if show_volume:
                            st.plotly_chart(create_volume_profile(full_chain_df), use_container_width=True)
                            
                            # Volume Statistics
                            total_call_vol = full_chain_df['Call Volume'].sum()
                            total_put_vol = full_chain_df['Put Volume'].sum()
                            vol_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else 0
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Call Volume", f"{total_call_vol:,.0f}")
                            with col2:
                                st.metric("Total Put Volume", f"{total_put_vol:,.0f}")
                            with col3:
                                st.metric("Put/Call Volume Ratio", f"{vol_ratio:.2f}")
                    
                     with tab5:
                        if show_greeks and 'call_delta' in full_chain_df.columns:
                            # Greeks visualization options
                            greek_col1, greek_col2 = st.columns(2)
                            with greek_col1:
                                selected_greek = st.selectbox("Select Greek", ["delta", "gamma", "vega", "theta", "rho"])
                            with greek_col2:
                                greek_option_type = st.radio("Option Type", ["Call", "Put"], horizontal=True)
                            
                            # Display Greeks table
                            greeks_cols = ['Strike', 'call_delta', 'call_gamma', 'call_vega', 'call_theta', 'call_rho',
                                         'put_delta', 'put_gamma', 'put_vega', 'put_theta', 'put_rho']
                            available_cols = [col for col in greeks_cols if col in full_chain_df.columns]
                            greeks_df = full_chain_df[available_cols].copy()
                            
                            # Rename columns for display
                            display_names = {
                                'call_delta': 'Call Δ', 'call_gamma': 'Call Γ', 'call_vega': 'Call V', 
                                'call_theta': 'Call Θ', 'call_rho': 'Call ρ',
                                'put_delta': 'Put Δ', 'put_gamma': 'Put Γ', 'put_vega': 'Put V', 
                                'put_theta': 'Put Θ', 'put_rho': 'Put ρ'
                            }
                            greeks_df.rename(columns=display_names, inplace=True)
                            
                            # Filter for near ATM strikes
                            atm_idx = (greeks_df['Strike'] - spot_price).abs().idxmin()
                            start_idx = max(0, atm_idx - 10)
                            end_idx = min(len(greeks_df), atm_idx + 11)
                            
                            # Style the dataframe
                            styled_greeks = greeks_df.iloc[start_idx:end_idx].style.format({
                                'Strike': '{:.0f}',
                                **{col: '{:.4f}' for col in greeks_df.columns if col != 'Strike'}
                            }).background_gradient(subset=[col for col in greeks_df.columns if 'Δ' in col], cmap='RdYlGn')
                            
                            st.dataframe(styled_greeks, use_container_width=True)
                            
                            # Greeks visualization
                            greek_surface = create_greeks_surface(full_chain_df, selected_greek, greek_option_type)
                            if greek_surface:
                                st.plotly_chart(greek_surface, use_container_width=True)
                    
                    with tab6:
                        if show_strategy:
                            st.subheader("Strategy Analysis")
                            
                            # Strategy selector
                            strategy = st.selectbox("Select Strategy", 
                                                  ["Long Straddle", "Short Straddle", "Long Strangle", 
                                                   "Short Strangle", "Bull Call Spread", "Bear Put Spread"])
                            
                            # Display strategy payoff
                            payoff_chart = create_strategy_payoff(full_chain_df, spot_price)
                            st.plotly_chart(payoff_chart, use_container_width=True)
                            
                            # Strategy metrics
                            atm_idx = (full_chain_df['Strike'] - spot_price).abs().idxmin()
                            call_premium = full_chain_df.loc[atm_idx, 'Call LTP']
                            put_premium = full_chain_df.loc[atm_idx, 'Put LTP']
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Premium", f"₹{call_premium + put_premium:.2f}")
                            with col2:
                                upper_be = atm_strike + call_premium + put_premium
                                st.metric("Upper Breakeven", f"₹{upper_be:.2f}")
                            with col3:
                                lower_be = atm_strike - call_premium - put_premium
                                st.metric("Lower Breakeven", f"₹{lower_be:.2f}")
                    
                    with tab7:
                        if 'historical_data' in st.session_state and not st.session_state.historical_data.empty:
                            hist_df = st.session_state.historical_data
                            
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
                            
                            # Max Pain Trend
                            fig2 = go.Figure()
                            fig2.add_trace(go.Scatter(x=hist_df['timestamp'], y=hist_df['max_pain'], 
                                                    mode='lines+markers', name='Max Pain'))
                            fig2.update_layout(
                                title='Max Pain Movement',
                                xaxis_title='Time',
                                yaxis_title='Max Pain',
                                height=300
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                        else:
                            st.info("Historical data will be tracked during this session.")
                    
                    # Options Chain Table
                    st.subheader("📋 Options Chain Data")
                    
                    # Advanced Filters
                    with st.expander("🔍 Advanced Filters", expanded=False):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            strike_range = st.slider("Strike Range", 
                                                   int(full_chain_df['Strike'].min()), 
                                                   int(full_chain_df['Strike'].max()),
                                                   (int(spot_price - 1000), int(spot_price + 1000)))
                        with col2:
                            oi_filter = st.number_input("Min OI Filter", value=0, step=1000)
                        with col3:
                            volume_filter = st.number_input("Min Volume Filter", value=0, step=100)
                        with col4:
                            moneyness = st.selectbox("Moneyness", ["All", "ITM", "ATM", "OTM"])
                    
                    # Apply filters
                    filtered_df = full_chain_df[
                        (full_chain_df['Strike'] >= strike_range[0]) & 
                        (full_chain_df['Strike'] <= strike_range[1]) &
                        ((full_chain_df['Call OI'] >= oi_filter) | (full_chain_df['Put OI'] >= oi_filter)) &
                        ((full_chain_df['Call Volume'] >= volume_filter) | (full_chain_df['Put Volume'] >= volume_filter))
                    ].copy()
                    
                    # Apply moneyness filter
                    if moneyness == "ITM":
                        filtered_df = filtered_df[
                            ((filtered_df['Strike'] < spot_price) & (filtered_df['Put LTP'] > 0)) |
                            ((filtered_df['Strike'] > spot_price) & (filtered_df['Call LTP'] > 0))
                        ]
                    elif moneyness == "ATM":
                        atm_range = config.get_strike_step(symbol) * 2
                        filtered_df = filtered_df[
                            (filtered_df['Strike'] >= spot_price - atm_range) & 
                            (filtered_df['Strike'] <= spot_price + atm_range)
                        ]
                    elif moneyness == "OTM":
                        filtered_df = filtered_df[
                            ((filtered_df['Strike'] > spot_price) & (filtered_df['Put LTP'] > 0)) |
                            ((filtered_df['Strike'] < spot_price) & (filtered_df['Call LTP'] > 0))
                        ]
                    
                    # Display columns
                    display_cols = ['Call OI', 'Call Chng OI', 'Call LTP', 'Call Volume', 'Strike', 
                                  'Put LTP', 'Put Volume', 'Put Chng OI', 'Put OI']
                    
                    if 'Call IV' in filtered_df.columns:
                        display_cols.extend(['Call IV', 'Put IV'])
                    
                    # Add moneyness indicator
                    filtered_df['Moneyness'] = filtered_df.apply(
                        lambda row: 'ITM' if (row['Strike'] < spot_price and row['Put LTP'] > 0) or 
                                           (row['Strike'] > spot_price and row['Call LTP'] > 0)
                        else 'OTM' if (row['Strike'] > spot_price and row['Put LTP'] > 0) or 
                                     (row['Strike'] < spot_price and row['Call LTP'] > 0)
                        else 'ATM', axis=1
                    )
                    
                    # Style the dataframe
                    def highlight_moneyness(row):
                        if row['Moneyness'] == 'ITM':
                            return ['background-color: #e8f5e9'] * len(row)
                        elif row['Moneyness'] == 'ATM':
                            return ['background-color: #fff3e0'] * len(row)
                        else:
                            return [''] * len(row)
                    
                    styled_df = filtered_df[display_cols + ['Moneyness']].style.format({
                        'Call OI': '{:,.0f}',
                        'Call Chng OI': '{:+,.0f}',
                        'Call LTP': '{:,.2f}',
                        'Call Volume': '{:,.0f}',
                        'Strike': '{:,.0f}',
                        'Put LTP': '{:,.2f}',
                        'Put Chng OI': '{:+,.0f}',
                        'Put OI': '{:,.0f}',
                        'Put Volume': '{:,.0f}',
                        'Call IV': '{:.1f}%',
                        'Put IV': '{:.1f}%'
                    }).background_gradient(subset=['Call OI', 'Put OI'], cmap='YlOrRd'
                    ).apply(highlight_moneyness, axis=1)
                    
                    # Display the table
                    st.dataframe(styled_df, use_container_width=True, height=600)
                    
                    # Summary statistics
                    with st.expander("📊 Summary Statistics"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Call Options Summary:**")
                            st.write(f"- Total OI: {filtered_df['Call OI'].sum():,.0f}")
                            st.write(f"- Total Volume: {filtered_df['Call Volume'].sum():,.0f}")
                            st.write(f"- Avg IV: {filtered_df['Call IV'].mean():.1f}%" if 'Call IV' in filtered_df.columns else "")
                            st.write(f"- Max OI Strike: {filtered_df.loc[filtered_df['Call OI'].idxmax(), 'Strike']:,.0f}")
                        
                        with col2:
                            st.write("**Put Options Summary:**")
                            st.write(f"- Total OI: {filtered_df['Put OI'].sum():,.0f}")
                            st.write(f"- Total Volume: {filtered_df['Put Volume'].sum():,.0f}")
                            st.write(f"- Avg IV: {filtered_df['Put IV'].mean():.1f}%" if 'Put IV' in filtered_df.columns else "")
                            st.write(f"- Max OI Strike: {filtered_df.loc[filtered_df['Put OI'].idxmax(), 'Strike']:,.0f}")
                    
                    # Export functionality
                    if st.sidebar.button("📥 Export Data", use_container_width=True):
                        export_df = prepare_export_data(full_chain_df, export_format)
                        if export_df is not None:
                            export_data_dict = {
                                'metadata': {
                                    'symbol': symbol,
                                    'expiry': selected_expiry,
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'spot_price': spot_price,
                                    'metrics': metrics
                                },
                                'chain_data': export_df.to_dict('records')
                            }
                            
                            if export_format == "JSON":
                                json_str = json.dumps(export_data_dict, indent=2, default=str)
                                st.download_button(
                                    label="Download JSON",
                                    data=json_str,
                                    file_name=f"{symbol}_options_chain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json"
                                )
                            elif export_format == "CSV":
                                csv = export_df.to_csv(index=False)
                                st.download_button(
                                    label="Download CSV",
                                    data=csv,
                                    file_name=f"{symbol}_options_chain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                            elif export_format == "Excel":
                                output = BytesIO()
                                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                    export_df.to_excel(writer, sheet_name='Options Chain', index=False)
                                    pd.DataFrame([metrics]).to_excel(writer, sheet_name='Metrics', index=False)
                                    if 'historical_data' in st.session_state and not st.session_state.historical_data.empty:
                                        st.session_state.historical_data.to_excel(
                                            writer, sheet_name='Historical', index=False
                                        )
                                excel_data = output.getvalue()
                                st.download_button(
                                    label="Download Excel",
                                    data=excel_data,
                                    file_name=f"{symbol}_options_chain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                
                else:
                    st.error("No data to display. The options chain might be empty.")
            else:
                st.error("Failed to fetch options data. Please try again.")
                
        except BreezeAPIError as e:
            st.error(str(e))
            logger.error(f"API Error: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            logger.error(f"Unexpected error: {e}", exc_info=True)
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

# --- UNIT TESTS (save as test_options_analyzer.py) ---
def test_calculate_iv():
    """Test IV calculation"""
    iv = calculate_iv('Call', 100, 100, 5, 0.25)
    assert 0 < iv < 2, "IV should be between 0 and 2"

def test_calculate_greeks_vectorized():
    """Test vectorized Greeks calculation"""
    iv_array = np.array([0.2, 0.25, 0.3])
    strikes = np.array([95, 100, 105])
    greeks = calculate_greeks_vectorized(iv_array, 'Call', 100, strikes, 0.25)
    
    assert len(greeks) == len(strikes), "Greeks output should match strikes length"
    assert all(-1 <= greeks['delta'].values) and all(greeks['delta'].values <= 1), "Delta should be between -1 and 1"

def test_validate_option_data():
    """Test option data validation"""
    # Valid data
    valid_df = pd.DataFrame({
        'strike_price': [100, 105, 110],
        'ltp': [5, 3, 1],
        'oi': [1000, 2000, 1500],
        'volume': [100, 200, 150]
    })
    assert validate_option_data(valid_df) == True
    
    # Invalid data - missing columns
    invalid_df = pd.DataFrame({
        'strike_price': [100, 105, 110]
    })
    assert validate_option_data(invalid_df) == False

# Run the application
if __name__ == "__main__":
    main()
    
                        
    
