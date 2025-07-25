import streamlit as st
import pandas as pd
from breeze_connect import BreezeConnect
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta, time as dt_time
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
import time
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import json
import logging
from io import BytesIO
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Deque, Callable
from collections import deque
import threading
from queue import Queue
import warnings
import websocket
import asyncio
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
    
    # OI Flow Analysis Configuration
    OI_FLOW_THRESHOLDS: Dict[str, float] = None
    OI_FLOW_TIMEFRAMES: Dict[str, Dict[str, int]] = None
    
    # Real-time Configuration
    REALTIME_FETCH_INTERVAL: float = 2.0  # seconds
    REALTIME_BUFFER_SIZE: int = 1000
    REALTIME_ALERT_BUFFER_SIZE: int = 100
    
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
        if self.OI_FLOW_THRESHOLDS is None:
            self.OI_FLOW_THRESHOLDS = {
                'large_oi_change': 0.10,
                'unusual_volume': 2.5,
                'rapid_iv_change': 0.05,
                'concentration_threshold': 0.15,
                'sweep_size': 100,
                'institutional_size': 500
            }
        if self.OI_FLOW_TIMEFRAMES is None:
            self.OI_FLOW_TIMEFRAMES = {
                '5min': {'window': 5, 'periods': 12},
                '10min': {'window': 10, 'periods': 18},
                '30min': {'window': 30, 'periods': 16},
                '1hour': {'window': 60, 'periods': 8},
                '2hour': {'window': 120, 'periods': 6},
                'daily': {'window': 390, 'periods': 5}
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
    page_title="Pro Options Analyzer - WebSocket Real-Time", 
    page_icon="🚀", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM EXCEPTIONS ---
class BreezeAPIError(Exception):
    """Custom exception for Breeze API errors"""
    pass

class WebSocketError(Exception):
    """Custom exception for WebSocket errors"""
    pass

# --- OI FLOW ANALYSIS DATA STRUCTURES ---
@dataclass
class OIFootprint:
    """Structure to store OI footprint data"""
    timestamp: datetime
    strike: float
    option_type: str
    oi_change: int
    volume: int
    price_change: float
    bid_ask_spread: float
    iv_change: float
    large_trade_indicator: bool
    aggressor_side: str

@dataclass
class RealTimeAlert:
    """Structure for real-time alerts"""
    timestamp: datetime
    alert_type: str
    strike: float
    option_type: str
    message: str
    severity: str
    data: Dict[str, Any]

# --- TIME-BASED OI TRACKER ---
class TimeBasedOITracker:
    """Track OI changes throughout the trading day with robust error handling"""
    
    def __init__(self):
        self.oi_history = {}  # {timestamp: {strike: {call_oi, put_oi}}}
        self.trading_start = dt_time(9, 15)
        self.trading_end = dt_time(15, 30)
        self.last_update = None
        self.lock = threading.Lock()  # Thread safety
        
    def add_snapshot(self, chain_df: pd.DataFrame, timestamp: datetime) -> bool:
        """Add OI snapshot at given timestamp with error handling"""
        try:
            with self.lock:
                if chain_df.empty:
                    logger.warning("Empty dataframe provided to add_snapshot")
                    return False
                
                snapshot = {}
                required_columns = ['Strike', 'Call OI', 'Put OI']
                
                # Validate columns
                if not all(col in chain_df.columns for col in required_columns):
                    logger.error(f"Missing required columns. Available: {chain_df.columns.tolist()}")
                    return False
                
                for _, row in chain_df.iterrows():
                    try:
                        strike = float(row['Strike'])
                        snapshot[strike] = {
                            'call_oi': int(row['Call OI']) if pd.notna(row['Call OI']) else 0,
                            'put_oi': int(row['Put OI']) if pd.notna(row['Put OI']) else 0,
                            'call_volume': int(row.get('Call Volume', 0)) if pd.notna(row.get('Call Volume', 0)) else 0,
                            'put_volume': int(row.get('Put Volume', 0)) if pd.notna(row.get('Put Volume', 0)) else 0
                        }
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error processing row with strike {row.get('Strike', 'Unknown')}: {e}")
                        continue
                
                if snapshot:
                    self.oi_history[timestamp] = snapshot
                    self.last_update = timestamp
                    logger.info(f"Added OI snapshot at {timestamp} with {len(snapshot)} strikes")
                    return True
                else:
                    logger.warning("No valid data in snapshot")
                    return False
                    
        except Exception as e:
            logger.error(f"Error adding OI snapshot: {e}")
            return False
    
    def get_oi_at_time(self, target_time: datetime) -> Dict:
        """Get OI data at or before target time with error handling"""
        try:
            with self.lock:
                if not self.oi_history:
                    logger.warning("No OI history available")
                    return {}
                
                sorted_times = sorted(self.oi_history.keys())
                
                # Find the latest time <= target_time
                latest_time = None
                for t in sorted_times:
                    if t <= target_time:
                        latest_time = t
                    else:
                        break
                
                if latest_time:
                    return self.oi_history.get(latest_time, {})
                else:
                    logger.info(f"No data available before {target_time}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error getting OI at time: {e}")
            return {}
    
    def calculate_oi_changes(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Calculate OI changes between two times with robust error handling"""
        try:
            with self.lock:
                start_data = self.get_oi_at_time(start_time)
                end_data = self.get_oi_at_time(end_time)
                
                if not start_data and not end_data:
                    logger.warning(f"No data available for time range {start_time} to {end_time}")
                    return pd.DataFrame()
                
                # If no start data, use first available data
                if not start_data and self.oi_history:
                    first_time = min(self.oi_history.keys())
                    start_data = self.oi_history[first_time]
                    logger.info(f"Using first available data from {first_time}")
                
                changes = []
                all_strikes = set()
                
                if start_data:
                    all_strikes.update(start_data.keys())
                if end_data:
                    all_strikes.update(end_data.keys())
                
                for strike in sorted(all_strikes):
                    try:
                        start_call_oi = start_data.get(strike, {}).get('call_oi', 0)
                        end_call_oi = end_data.get(strike, {}).get('call_oi', 0)
                        start_put_oi = start_data.get(strike, {}).get('put_oi', 0)
                        end_put_oi = end_data.get(strike, {}).get('put_oi', 0)
                        
                        changes.append({
                            'Strike': strike,
                            'Call OI': end_call_oi,
                            'Put OI': end_put_oi,
                            'Call OI Change': end_call_oi - start_call_oi,
                            'Put OI Change': end_put_oi - start_put_oi,
                            'Call OI Start': start_call_oi,
                            'Put OI Start': start_put_oi
                        })
                    except Exception as e:
                        logger.warning(f"Error calculating changes for strike {strike}: {e}")
                        continue
                
                df = pd.DataFrame(changes)
                return df if not df.empty else pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error calculating OI changes: {e}")
            return pd.DataFrame()
    
    def clear_old_data(self, hours_to_keep: int = 8):
        """Clear data older than specified hours"""
        try:
            with self.lock:
                if not self.oi_history:
                    return
                
                cutoff_time = datetime.now() - timedelta(hours=hours_to_keep)
                times_to_remove = [t for t in self.oi_history.keys() if t < cutoff_time]
                
                for t in times_to_remove:
                    del self.oi_history[t]
                
                if times_to_remove:
                    logger.info(f"Cleared {len(times_to_remove)} old snapshots")
                    
        except Exception as e:
            logger.error(f"Error clearing old data: {e}")

# --- ENHANCED OI CHART CREATION ---
def create_enhanced_oi_chart(chain_df: pd.DataFrame, 
                            spot_price: float,
                            oi_tracker: Optional[TimeBasedOITracker] = None,
                            selected_time_range: str = "Full Day",
                            custom_start_time: Optional[datetime] = None,
                            custom_end_time: Optional[datetime] = None,
                            show_oi_toggle: bool = True) -> go.Figure:
    """Create enhanced OI distribution chart with time-based analysis and robust error handling"""
    
    try:
        # Input validation
        if chain_df.empty:
            logger.warning("Empty dataframe provided to create_enhanced_oi_chart")
            fig = go.Figure()
            fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        # Determine time range
        now = datetime.now()
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        
        time_ranges = {
            "Last 5 mins": max(now - timedelta(minutes=5), market_open),
            "Last 10 mins": max(now - timedelta(minutes=10), market_open),
            "Last 15 mins": max(now - timedelta(minutes=15), market_open),
            "Last 30 mins": max(now - timedelta(minutes=30), market_open),
            "Last 1 Hr": max(now - timedelta(hours=1), market_open),
            "Last 2 Hrs": max(now - timedelta(hours=2), market_open),
            "Last 3 Hrs": max(now - timedelta(hours=3), market_open),
            "Full Day": market_open
        }
        
        # Use custom time range or preset
        if custom_start_time and custom_end_time:
            start_time = custom_start_time
            end_time = custom_end_time
        else:
            start_time = time_ranges.get(selected_time_range, market_open)
            end_time = now
        
        # Get OI changes for the time range
        if oi_tracker and oi_tracker.oi_history:
            oi_data = oi_tracker.calculate_oi_changes(start_time, end_time)
            
            # If no historical data, fall back to current data
            if oi_data.empty:
                logger.info("No historical data available, using current chain data")
                oi_data = prepare_oi_data_from_chain(chain_df)
        else:
            # Fallback to current chain data
            oi_data = prepare_oi_data_from_chain(chain_df)
        
        if oi_data.empty:
            fig = go.Figure()
            fig.add_annotation(text="No OI data available for selected time range", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        # Create figure
        fig = go.Figure()
        
        # Bar width configuration
        bar_width = 0.35  # Thicker bars as requested
        
        # Add Call OI bars
        if show_oi_toggle:
            # Base OI (start of period)
            base_call_oi = oi_data[oi_data['Call OI Start'] > 0]
            if not base_call_oi.empty:
                fig.add_trace(go.Bar(
                    x=base_call_oi['Strike'],
                    y=base_call_oi['Call OI Start'],
                    name='Call OI (Base)',
                    marker_color='rgba(255, 0, 0, 0.6)',
                    width=bar_width,
                    offset=-bar_width/2,
                    hovertemplate='Strike: %{x}<br>Call OI Base: %{y:,.0f}<extra></extra>',
                    legendgroup='call',
                    showlegend=True
                ))
            
            # OI Changes (positive - hatched pattern)
            positive_call_changes = oi_data[oi_data['Call OI Change'] > 0]
            if not positive_call_changes.empty:
                fig.add_trace(go.Bar(
                    x=positive_call_changes['Strike'],
                    y=positive_call_changes['Call OI Change'],
                    name='Call OI Added',
                    marker=dict(
                        color='rgba(255, 0, 0, 0.8)',
                        pattern=dict(shape="/", size=4, solidity=0.7)
                    ),
                    width=bar_width,
                    offset=-bar_width/2,
                    base=positive_call_changes['Call OI Start'],
                    hovertemplate='Strike: %{x}<br>Call OI Added: %{y:,.0f}<extra></extra>',
                    legendgroup='call',
                    showlegend=True
                ))
            
            # OI Changes (negative - hollow pattern)
            negative_call_changes = oi_data[oi_data['Call OI Change'] < 0]
            if not negative_call_changes.empty:
                fig.add_trace(go.Bar(
                    x=negative_call_changes['Strike'],
                    y=negative_call_changes['Call OI Change'].abs(),
                    name='Call OI Removed',
                    marker=dict(
                        color='rgba(255, 255, 255, 0.8)',
                        line=dict(color='red', width=2)
                    ),
                    width=bar_width,
                    offset=-bar_width/2,
                    base=negative_call_changes['Call OI Start'] + negative_call_changes['Call OI Change'],
                    hovertemplate='Strike: %{x}<br>Call OI Removed: %{y:,.0f}<extra></extra>',
                    legendgroup='call',
                    showlegend=True
                ))
        else:
            # Show only current OI
            current_call_oi = oi_data[oi_data['Call OI'] > 0]
            if not current_call_oi.empty:
                fig.add_trace(go.Bar(
                    x=current_call_oi['Strike'],
                    y=current_call_oi['Call OI'],
                    name='Call OI',
                    marker_color='rgba(255, 0, 0, 0.7)',
                    width=bar_width,
                    offset=-bar_width/2,
                    hovertemplate='Strike: %{x}<br>Call OI: %{y:,.0f}<extra></extra>'
                ))
        
        # Add Put OI bars
        if show_oi_toggle:
            # Base OI (start of period)
            base_put_oi = oi_data[oi_data['Put OI Start'] > 0]
            if not base_put_oi.empty:
                fig.add_trace(go.Bar(
                    x=base_put_oi['Strike'],
                    y=base_put_oi['Put OI Start'],
                    name='Put OI (Base)',
                    marker_color='rgba(0, 255, 0, 0.6)',
                    width=bar_width,
                    offset=bar_width/2,
                    hovertemplate='Strike: %{x}<br>Put OI Base: %{y:,.0f}<extra></extra>',
                    legendgroup='put',
                    showlegend=True
                ))
            
            # OI Changes (positive - hatched pattern)
            positive_put_changes = oi_data[oi_data['Put OI Change'] > 0]
            if not positive_put_changes.empty:
                fig.add_trace(go.Bar(
                    x=positive_put_changes['Strike'],
                    y=positive_put_changes['Put OI Change'],
                    name='Put OI Added',
                    marker=dict(
                        color='rgba(0, 255, 0, 0.8)',
                        pattern=dict(shape="/", size=4, solidity=0.7)
                    ),
                    width=bar_width,
                    offset=bar_width/2,
                    base=positive_put_changes['Put OI Start'],
                    hovertemplate='Strike: %{x}<br>Put OI Added: %{y:,.0f}<extra></extra>',
                    legendgroup='put',
                    showlegend=True
                ))
            
            # OI Changes (negative - hollow pattern)
            negative_put_changes = oi_data[oi_data['Put OI Change'] < 0]
            if not negative_put_changes.empty:
                fig.add_trace(go.Bar(
                    x=negative_put_changes['Strike'],
                    y=negative_put_changes['Put OI Change'].abs(),
                    name='Put OI Removed',
                    marker=dict(
                        color='rgba(255, 255, 255, 0.8)',
                        line=dict(color='green', width=2)
                    ),
                    width=bar_width,
                    offset=bar_width/2,
                    base=negative_put_changes['Put OI Start'] + negative_put_changes['Put OI Change'],
                    hovertemplate='Strike: %{x}<br>Put OI Removed: %{y:,.0f}<extra></extra>',
                    legendgroup='put',
                    showlegend=True
                ))
        else:
            # Show only current OI
            current_put_oi = oi_data[oi_data['Put OI'] > 0]
            if not current_put_oi.empty:
                fig.add_trace(go.Bar(
                    x=current_put_oi['Strike'],
                    y=current_put_oi['Put OI'],
                    name='Put OI',
                    marker_color='rgba(0, 255, 0, 0.7)',
                    width=bar_width,
                    offset=bar_width/2,
                    hovertemplate='Strike: %{x}<br>Put OI: %{y:,.0f}<extra></extra>'
                ))
        
        # Add spot price line
        fig.add_vline(
            x=spot_price, 
            line_width=2, 
            line_dash="solid", 
            line_color="blue",
            annotation_text=f"Spot: {spot_price:.0f}",
            annotation_position="top left"
        )
        
        # Calculate total OI changes safely
        total_call_change = oi_data['Call OI Change'].sum() if 'Call OI Change' in oi_data.columns else 0
        total_put_change = oi_data['Put OI Change'].sum() if 'Put OI Change' in oi_data.columns else 0
        
        # Update layout
        fig.update_layout(
            title={
                'text': f'Open Interest Distribution<br><sub>Time: {start_time.strftime("%H:%M")} - {end_time.strftime("%H:%M")} | '
                       f'Call OI Change: {total_call_change:+,.0f} | Put OI Change: {total_put_change:+,.0f}</sub>',
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title='Strike Price',
            yaxis_title='Open Interest',
            barmode='overlay',
            bargap=0.1,
            height=500,
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='black',
                borderwidth=1
            ),
            xaxis=dict(
                rangeslider=dict(
                    visible=True,
                    thickness=0.05
                ),
                type='linear'
            )
        )
        
        # Add annotations for interpretation
        fig.add_annotation(
            text="🟢 Hatched = OI Added | ⚪ Hollow = OI Removed",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=12)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating enhanced OI chart: {e}")
        # Return empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig

def prepare_oi_data_from_chain(chain_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare OI data from chain dataframe with error handling"""
    try:
        required_cols = ['Strike', 'Call OI', 'Put OI']
        if not all(col in chain_df.columns for col in required_cols):
            logger.error(f"Missing required columns. Available: {chain_df.columns.tolist()}")
            return pd.DataFrame()
        
        oi_data = chain_df[['Strike', 'Call OI', 'Put OI']].copy()
        
        # Handle OI changes if available
        if 'Call Chng OI' in chain_df.columns and 'Put Chng OI' in chain_df.columns:
            oi_data['Call OI Change'] = chain_df['Call Chng OI']
            oi_data['Put OI Change'] = chain_df['Put Chng OI']
            oi_data['Call OI Start'] = oi_data['Call OI'] - oi_data['Call OI Change']
            oi_data['Put OI Start'] = oi_data['Put OI'] - oi_data['Put OI Change']
        else:
            # No change data available
            oi_data['Call OI Change'] = 0
            oi_data['Put OI Change'] = 0
            oi_data['Call OI Start'] = oi_data['Call OI']
            oi_data['Put OI Start'] = oi_data['Put OI']
        
        # Ensure numeric types
        numeric_cols = ['Call OI', 'Put OI', 'Call OI Change', 'Put OI Change', 
                       'Call OI Start', 'Put OI Start']
        for col in numeric_cols:
            if col in oi_data.columns:
                oi_data[col] = pd.to_numeric(oi_data[col], errors='coerce').fillna(0)
        
        return oi_data
        
    except Exception as e:
        logger.error(f"Error preparing OI data from chain: {e}")
        return pd.DataFrame()

# --- ENHANCED OI ANALYSIS TAB ---
def create_enhanced_oi_analysis_tab(chain_df: pd.DataFrame, spot_price: float, symbol: str):
    """Create enhanced OI analysis tab with time-based features and robust error handling"""
    
    try:
        # Initialize OI tracker in session state if not exists
        if 'oi_tracker' not in st.session_state:
            st.session_state.oi_tracker = TimeBasedOITracker()
        
        oi_tracker = st.session_state.oi_tracker
        
        # Add current snapshot to tracker
        snapshot_added = oi_tracker.add_snapshot(chain_df, datetime.now())
        if not snapshot_added:
            st.warning("Failed to add current snapshot to OI tracker")
        
        # Clear old data periodically
        if datetime.now().minute % 30 == 0:  # Every 30 minutes
            oi_tracker.clear_old_data(hours_to_keep=8)
        
        # Time range selection
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            time_range = st.selectbox(
                "Select Time Range",
                ["Last 5 mins", "Last 10 mins", "Last 15 mins", "Last 30 mins", 
                 "Last 1 Hr", "Last 2 Hrs", "Last 3 Hrs", "Full Day", "Custom"],
                index=7,  # Default to Full Day
                key="oi_time_range"
            )
        
        with col2:
            show_oi_toggle = st.checkbox("Show OI Changes", value=True, 
                                        help="Toggle to show OI changes with base OI",
                                        key="show_oi_changes")
        
        with col3:
            if st.button("🔄 Refresh OI Data", key="refresh_oi"):
                st.rerun()
        
        # Custom time range selector
        custom_start = None
        custom_end = None
        if time_range == "Custom":
            col1, col2 = st.columns(2)
            with col1:
                custom_start_time = st.time_input("Start Time", value=dt_time(9, 15), key="oi_start_time")
                custom_start = datetime.combine(datetime.today(), custom_start_time)
            with col2:
                custom_end_time = st.time_input("End Time", value=datetime.now().time(), key="oi_end_time")
                custom_end = datetime.combine(datetime.today(), custom_end_time)
        
        # Create and display enhanced OI chart
        oi_chart = create_enhanced_oi_chart(
            chain_df, 
            spot_price, 
            oi_tracker,
            time_range,
            custom_start,
            custom_end,
            show_oi_toggle
        )
        st.plotly_chart(oi_chart, use_container_width=True, key="main_oi_chart")
        
        # OI Change Analysis Summary
        st.subheader("📊 OI Change Analysis")
        
        # Calculate current session changes
        if oi_tracker.oi_history:
            market_open = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)
            current_changes = oi_tracker.calculate_oi_changes(market_open, datetime.now())
            
            if not current_changes.empty:
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_call_add = current_changes[current_changes['Call OI Change'] > 0]['Call OI Change'].sum()
                    st.metric("Call OI Added", f"{total_call_add:,.0f}", 
                             delta=f"+{total_call_add:,.0f}" if total_call_add > 0 else None)
                
                with col2:
                    total_call_remove = abs(current_changes[current_changes['Call OI Change'] < 0]['Call OI Change'].sum())
                    st.metric("Call OI Removed", f"{total_call_remove:,.0f}", 
                             delta=f"-{total_call_remove:,.0f}" if total_call_remove > 0 else None)
                
                with col3:
                    total_put_add = current_changes[current_changes['Put OI Change'] > 0]['Put OI Change'].sum()
                    st.metric("Put OI Added", f"{total_put_add:,.0f}", 
                             delta=f"+{total_put_add:,.0f}" if total_put_add > 0 else None)
                
                with col4:
                    total_put_remove = abs(current_changes[current_changes['Put OI Change'] < 0]['Put OI Change'].sum())
                    st.metric("Put OI Removed", f"{total_put_remove:,.0f}", 
                             delta=f"-{total_put_remove:,.0f}" if total_put_remove > 0 else None)
                
                # Interpretation guide
                with st.expander("📖 How to Read the OI Change Graph", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("""
                        ### Call Options Interpretation:
                        - **🔴 Solid Red + Hatched**: Call OI increasing
                          - Big players selling calls
                          - Resistance at that strike
                          - Index may not go above
                        
                        - **⚪ Hollow Red Border**: Call OI decreasing
                          - Call sellers buying back
                          - Bullish indication
                          - Resistance weakening
                        """)
                    
                    with col2:
                        st.markdown("""
                        ### Put Options Interpretation:
                        - **🟢 Solid Green + Hatched**: Put OI increasing
                          - Big players selling puts
                          - Support at that strike
                          - Index may not go below
                        
                        - **⚪ Hollow Green Border**: Put OI decreasing
                          - Put sellers buying back
                          - Bearish indication
                          - Support weakening
                        """)
                
                # Top OI changes table
                st.subheader("🎯 Significant OI Changes")
                
                # Find top changes
                top_changes = current_changes.copy()
                top_changes['Call OI Change Abs'] = top_changes['Call OI Change'].abs()
                top_changes['Put OI Change Abs'] = top_changes['Put OI Change'].abs()
                
                # Filter out zero changes
                call_changes_filtered = top_changes[top_changes['Call OI Change Abs'] > 0]
                put_changes_filtered = top_changes[top_changes['Put OI Change Abs'] > 0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Top Call OI Changes:**")
                    if not call_changes_filtered.empty:
                        top_call_changes = call_changes_filtered.nlargest(5, 'Call OI Change Abs')[
                            ['Strike', 'Call OI', 'Call OI Change']
                        ]
                        st.dataframe(
                            top_call_changes.style.format({
                                'Strike': '{:.0f}',
                                'Call OI': '{:,.0f}',
                                'Call OI Change': '{:+,.0f}'
                            }).background_gradient(subset=['Call OI Change'], cmap='RdYlGn'),
                            use_container_width=True,
                            key="top_call_changes"
                        )
                    else:
                        st.info("No significant call OI changes")
                
                with col2:
                    st.write("**Top Put OI Changes:**")
                    if not put_changes_filtered.empty:
                        top_put_changes = put_changes_filtered.nlargest(5, 'Put OI Change Abs')[
                            ['Strike', 'Put OI', 'Put OI Change']
                        ]
                        st.dataframe(
                            top_put_changes.style.format({
                                'Strike': '{:.0f}',
                                'Put OI': '{:,.0f}',
                                'Put OI Change': '{:+,.0f}'
                            }).background_gradient(subset=['Put OI Change'], cmap='RdYlGn'),
                            use_container_width=True,
                            key="top_put_changes"
                        )
                    else:
                        st.info("No significant put OI changes")
                
                # OI Change Distribution
                st.subheader("📈 OI Change Distribution")
                
                # Create subplot for call and put OI change distribution
                fig_dist = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Call OI Changes", "Put OI Changes")
                )
                
                # Call OI change distribution
                call_changes_dist = current_changes[current_changes['Call OI Change'] != 0]
                if not call_changes_dist.empty:
                    fig_dist.add_trace(
                        go.Histogram(
                            x=call_changes_dist['Call OI Change'],
                            name='Call Changes',
                            marker_color='red',
                            opacity=0.7
                        ),
                        row=1, col=1
                    )
                
                # Put OI change distribution
                put_changes_dist = current_changes[current_changes['Put OI Change'] != 0]
                if not put_changes_dist.empty:
                    fig_dist.add_trace(
                        go.Histogram(
                            x=put_changes_dist['Put OI Change'],
                            name='Put Changes',
                            marker_color='green',
                            opacity=0.7
                        ),
                        row=1, col=2
                    )
                
                fig_dist.update_layout(
                    height=300,
                    showlegend=False,
                    title_text="OI Change Distribution Analysis"
                )
                
                st.plotly_chart(fig_dist, use_container_width=True, key="oi_dist_chart")
                
            else:
                st.info("Waiting for more data points to show analysis...")
        else:
            st.info("OI tracking will begin once data is available. Please wait for the next update.")
    
    except Exception as e:
        logger.error(f"Error in enhanced OI analysis tab: {e}")
        st.error(f"Error creating OI analysis: {str(e)}")
        st.info("Please try refreshing the page or check the logs for more details.")

# --- WEBSOCKET DATA STREAMER ---
class WebSocketDataStreamer:
    """WebSocket-based real-time data streaming for options analysis"""
    
    def __init__(self, breeze_connection: BreezeConnect, symbol: str, expiry_date: str):
        self.breeze = breeze_connection
        self.symbol = symbol
        self.expiry_date = expiry_date
        self.is_streaming = False
        self.ws_connected = False
        
        # Data storage
        self.last_data = {}
        self.tick_count = 0
        
        # Real-time buffers
        self.oi_changes_buffer: Deque[Dict] = deque(maxlen=config.REALTIME_BUFFER_SIZE)
        self.price_changes_buffer: Deque[Dict] = deque(maxlen=config.REALTIME_BUFFER_SIZE)
        self.alerts_buffer: Deque[RealTimeAlert] = deque(maxlen=config.REALTIME_ALERT_BUFFER_SIZE)
        
        # Thresholds
        self.oi_change_threshold = 1000
        self.price_change_threshold = 0.05
        self.rapid_change_window = 30
        
        # WebSocket callbacks
        self.on_ticks = self._on_ticks_update
        
        # Strike list to monitor
        self.monitored_strikes = []
        self._initialize_strikes()
        
        # Thread safety
        self.data_lock = threading.Lock()
    
    def _initialize_strikes(self):
        """Initialize the list of strikes to monitor"""
        try:
            # Get current spot price
            spot_data = self.breeze.get_quotes(
                stock_code=self.symbol,
                exchange_code="NSE",
                product_type="cash"
            )
            if spot_data.get('Success'):
                spot_price = float(spot_data['Success'][0]['ltp'])
                
                # Calculate strike range (e.g., +/- 5% from spot)
                strike_step = config.get_strike_step(self.symbol)
                lower_bound = int((spot_price * 0.95) / strike_step) * strike_step
                upper_bound = int((spot_price * 1.05) / strike_step) * strike_step
                
                # Generate strike list
                self.monitored_strikes = list(range(lower_bound, upper_bound + strike_step, strike_step))
                logger.info(f"Monitoring {len(self.monitored_strikes)} strikes from {lower_bound} to {upper_bound}")
        except Exception as e:
            logger.error(f"Error initializing strikes: {e}")
            self.monitored_strikes = []
    
    def start_streaming(self) -> bool:
        """Start WebSocket streaming"""
        try:
            if not self.is_streaming:
                # Connect to WebSocket
                self.breeze.ws_connect()
                self.ws_connected = True
                
                # Set up callbacks
                self.breeze.on_ticks = self.on_ticks
                
                # Subscribe to feeds
                self._subscribe_to_feeds()
                
                self.is_streaming = True
                logger.info(f"WebSocket streaming started for {self.symbol}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to start WebSocket streaming: {e}")
            self.is_streaming = False
            self.ws_connected = False
            raise WebSocketError(f"Failed to start streaming: {e}")
    
    def stop_streaming(self) -> bool:
        """Stop WebSocket streaming"""
        try:
            if self.is_streaming:
                # Unsubscribe from feeds
                self._unsubscribe_from_feeds()
                
                # Disconnect WebSocket
                if self.ws_connected:
                    self.breeze.ws_disconnect()
                    self.ws_connected = False
                
                self.is_streaming = False
                logger.info(f"WebSocket streaming stopped for {self.symbol}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to stop WebSocket streaming: {e}")
            return False
    
    def _subscribe_to_feeds(self):
        """Subscribe to relevant option feeds"""
        try:
            # Subscribe to spot price
            self.breeze.subscribe_feeds(
                stock_code=self.symbol,
                exchange_code="NSE",
                product_type="cash",
                get_exchange_quotes=True,
                get_market_depth=False
            )
            
            # Subscribe to option chains for monitored strikes
            for strike in self.monitored_strikes:
                # Subscribe to calls
                self.breeze.subscribe_feeds(
                    stock_code=self.symbol,
                    exchange_code="NFO",
                    product_type="options",
                    expiry_date=self.expiry_date,
                    strike_price=str(strike),
                    right="Call",
                    get_exchange_quotes=True,
                    get_market_depth=False
                )
                
                # Subscribe to puts
                self.breeze.subscribe_feeds(
                    stock_code=self.symbol,
                    exchange_code="NFO",
                    product_type="options",
                    expiry_date=self.expiry_date,
                    strike_price=str(strike),
                    right="Put",
                    get_exchange_quotes=True,
                    get_market_depth=False
                )
            
            logger.info(f"Subscribed to {len(self.monitored_strikes) * 2 + 1} feeds")
            
        except Exception as e:
            logger.error(f"Error subscribing to feeds: {e}")
            raise WebSocketError(f"Failed to subscribe to feeds: {e}")
    
    def _unsubscribe_from_feeds(self):
        """Unsubscribe from all feeds"""
        try:
            # Unsubscribe from spot
            self.breeze.unsubscribe_feeds(
                stock_code=self.symbol,
                exchange_code="NSE",
                product_type="cash"
            )
            
            # Unsubscribe from options
            for strike in self.monitored_strikes:
                self.breeze.unsubscribe_feeds(
                    stock_code=self.symbol,
                    exchange_code="NFO",
                    product_type="options",
                    expiry_date=self.expiry_date,
                    strike_price=str(strike),
                    right="Call"
                )
                
                self.breeze.unsubscribe_feeds(
                    stock_code=self.symbol,
                    exchange_code="NFO",
                    product_type="options",
                    expiry_date=self.expiry_date,
                    strike_price=str(strike),
                    right="Put"
                )
                
        except Exception as e:
            logger.error(f"Error unsubscribing from feeds: {e}")
    
    def _on_ticks_update(self, ticks):
        """Handle incoming tick data"""
        try:
            with self.data_lock:
                self.tick_count += 1
                
                for tick in ticks:
                    # Parse tick data
                    symbol = tick.get('stock_code', '')
                    product = tick.get('product_type', '')
                    
                    if symbol != self.symbol:
                        continue
                    
                    # Update data based on product type
                    if product == 'cash':
                        self._update_spot_data(tick)
                    elif product == 'options':
                        self._update_option_data(tick)
                
                # Check for significant changes
                self._detect_and_alert_changes()
            
        except Exception as e:
            logger.error(f"Error processing tick data: {e}")
    
    def _update_spot_data(self, tick):
        """Update spot price data"""
        try:
            current_spot = float(tick.get('ltp', 0))
            
            if 'spot' in self.last_data:
                prev_spot = self.last_data['spot']
                change = current_spot - prev_spot
                change_pct = (change / prev_spot) * 100 if prev_spot > 0 else 0
                
                # Store spot change
                self.last_data['spot_change'] = change
                self.last_data['spot_change_pct'] = change_pct
            
            self.last_data['spot'] = current_spot
            self.last_data['spot_timestamp'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating spot data: {e}")
    
    def _update_option_data(self, tick):
        """Update option data and detect changes"""
        try:
            strike = float(tick.get('strike_price', 0))
            right = tick.get('right', '')
            ltp = float(tick.get('ltp', 0))
            oi = int(tick.get('open_interest', 0))
            volume = int(tick.get('volume', 0))
            bid = float(tick.get('best_bid_price', 0))
            ask = float(tick.get('best_ask_price', 0))
            
            # Create unique key
            key = f"{strike}_{right}"
            
            # Check for OI changes
            if key in self.last_data:
                prev_data = self.last_data[key]
                
                # OI change detection
                oi_change = oi - prev_data.get('oi', 0)
                if abs(oi_change) > 0:
                    self.oi_changes_buffer.append({
                        'timestamp': datetime.now(),
                        'strike': strike,
                        'type': right.upper(),
                        'oi_change': oi_change,
                        'current_oi': oi,
                        'prev_oi': prev_data.get('oi', 0),
                        'ltp': ltp,
                        'volume': volume,
                        'bid_ask_spread': ask - bid if ask > 0 and bid > 0 else 0
                    })
                    
                    # Generate alert for large OI changes
                    if abs(oi_change) >= self.oi_change_threshold:
                        self._generate_oi_alert(strike, right, oi_change, oi)
                
                # Price change detection
                prev_ltp = prev_data.get('ltp', 0)
                if prev_ltp > 0:
                    price_change = ltp - prev_ltp
                    price_change_pct = (price_change / prev_ltp) * 100
                    
                    if abs(price_change_pct) > self.price_change_threshold * 100:
                        self.price_changes_buffer.append({
                            'timestamp': datetime.now(),
                            'strike': strike,
                            'type': right.upper(),
                            'price_change': price_change,
                            'price_change_pct': price_change_pct,
                            'current_ltp': ltp,
                            'prev_ltp': prev_ltp
                        })
                        
                        # Generate price alert
                        if abs(price_change_pct) >= 10:
                            self._generate_price_alert(strike, right, price_change_pct)
            
            # Update last data
            self.last_data[key] = {
                'oi': oi,
                'ltp': ltp,
                'volume': volume,
                'bid': bid,
                'ask': ask,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error updating option data: {e}")
    
    def _generate_oi_alert(self, strike: float, option_type: str, oi_change: int, current_oi: int):
        """Generate alert for significant OI change"""
        severity = 'HIGH' if abs(oi_change) >= 2000 else 'MEDIUM'
        
        alert = RealTimeAlert(
            timestamp=datetime.now(),
            alert_type='LARGE_OI_CHANGE',
            strike=strike,
            option_type=option_type.upper(),
            message=f"Large {option_type.upper()} OI change: {oi_change:+,} at strike {strike} (Total: {current_oi:,})",
            severity=severity,
            data={'oi_change': oi_change, 'current_oi': current_oi}
        )
        self.alerts_buffer.append(alert)
    
    def _generate_price_alert(self, strike: float, option_type: str, price_change_pct: float):
        """Generate alert for significant price movement"""
        severity = 'HIGH' if abs(price_change_pct) >= 20 else 'MEDIUM'
        
        alert = RealTimeAlert(
            timestamp=datetime.now(),
            alert_type='UNUSUAL_PRICE_MOVE',
            strike=strike,
            option_type=option_type.upper(),
            message=f"Unusual {option_type.upper()} price move: {price_change_pct:+.1f}% at strike {strike}",
            severity=severity,
            data={'price_change_pct': price_change_pct}
        )
        self.alerts_buffer.append(alert)
    
    def _detect_and_alert_changes(self):
        """Detect patterns and generate additional alerts"""
        try:
            # Detect rapid accumulation
            self._detect_rapid_accumulation()
            
            # Detect sweep patterns
            self._detect_sweep_patterns()
            
            # Detect institutional patterns
            self._detect_institutional_patterns()
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
    
    def _detect_rapid_accumulation(self):
        """Detect rapid OI accumulation in real-time"""
        cutoff_time = datetime.now() - timedelta(seconds=self.rapid_change_window)
        
        # Group recent changes by strike and type
        strike_changes = {}
        for change in self.oi_changes_buffer:
            if change['timestamp'] > cutoff_time:
                key = f"{change['strike']}_{change['type']}"
                if key not in strike_changes:
                    strike_changes[key] = []
                strike_changes[key].append(change)
        
        # Check for rapid accumulation
        for key, changes in strike_changes.items():
            if len(changes) >= 3:  # Multiple changes in short time
                total_change = sum(c['oi_change'] for c in changes)
                if abs(total_change) >= 1500:
                    strike, option_type = key.split('_')
                    
                    # Check if already alerted recently
                    recent_alerts = [a for a in self.alerts_buffer 
                                   if a.alert_type == 'RAPID_ACCUMULATION' 
                                   and a.strike == float(strike)
                                   and a.timestamp > cutoff_time]
                    
                    if not recent_alerts:
                        alert = RealTimeAlert(
                            timestamp=datetime.now(),
                            alert_type='RAPID_ACCUMULATION',
                            strike=float(strike),
                            option_type=option_type,
                            message=f"Rapid {option_type} accumulation: {total_change:+,} in {self.rapid_change_window}s at strike {strike}",
                            severity='HIGH',
                            data={
                                'total_change': total_change,
                                'change_count': len(changes),
                                'timeframe': self.rapid_change_window
                            }
                        )
                        self.alerts_buffer.append(alert)
    
    def _detect_sweep_patterns(self):
        """Detect option sweep patterns from WebSocket data"""
        recent_window = datetime.now() - timedelta(seconds=5)
        
        # Check recent high-volume trades
        for key, data in self.last_data.items():
            if '_' not in key or 'timestamp' not in data:
                continue
                
            if data['timestamp'] > recent_window:
                volume = data.get('volume', 0)
                oi = data.get('oi', 0)
                
                # Sweep detection: high volume relative to OI
                if volume > config.OI_FLOW_THRESHOLDS['sweep_size'] and oi > 0:
                    volume_oi_ratio = volume / oi
                    if volume_oi_ratio > 0.5:  # Volume > 50% of OI
                        strike, option_type = key.split('_')
                        
                        # Check if already alerted
                        recent_sweep_alerts = [a for a in self.alerts_buffer 
                                             if a.alert_type == 'OPTION_SWEEP' 
                                             and a.strike == float(strike)
                                             and a.timestamp > recent_window]
                        
                        if not recent_sweep_alerts:
                            alert = RealTimeAlert(
                                timestamp=datetime.now(),
                                alert_type='OPTION_SWEEP',
                                strike=float(strike),
                                option_type=option_type,
                                message=f"Potential {option_type} sweep detected at strike {strike}: Volume {volume:,} ({volume_oi_ratio:.1%} of OI)",
                                severity='HIGH',
                                data={
                                    'volume': volume,
                                    'oi': oi,
                                    'volume_oi_ratio': volume_oi_ratio
                                }
                            )
                            self.alerts_buffer.append(alert)
    
    def _detect_institutional_patterns(self):
        """Detect institutional trading patterns"""
        recent_window = datetime.now() - timedelta(seconds=10)
        
        for key, data in self.last_data.items():
            if '_' not in key or 'timestamp' not in data:
                continue
            
            if data['timestamp'] > recent_window:
                volume = data.get('volume', 0)
                bid_ask_spread = data.get('ask', 0) - data.get('bid', 0)
                
                # Institutional size detection
                if volume >= config.OI_FLOW_THRESHOLDS['institutional_size']:
                    strike, option_type = key.split('_')
                    
                    # Check spread tightness (institutional tend to get better spreads)
                    tight_spread = bid_ask_spread < (data.get('ltp', 1) * 0.02)  # Less than 2% spread
                    
                    alert = RealTimeAlert(
                        timestamp=datetime.now(),
                        alert_type='INSTITUTIONAL_ACTIVITY',
                        strike=float(strike),
                        option_type=option_type,
                        message=f"Institutional size {option_type} trade at strike {strike}: Volume {volume:,}",
                        severity='MEDIUM',
                        data={
                            'volume': volume,
                            'tight_spread': tight_spread,
                            'spread': bid_ask_spread
                        }
                    )
                    self.alerts_buffer.append(alert)
    
    def get_recent_changes(self, seconds: int = 60) -> Dict[str, List]:
        """Get recent changes within specified time window"""
        try:
            with self.data_lock:
                cutoff_time = datetime.now() - timedelta(seconds=seconds)
                
                recent_oi = [c for c in self.oi_changes_buffer if c['timestamp'] > cutoff_time]
                recent_price = [c for c in self.price_changes_buffer if c['timestamp'] > cutoff_time]
                recent_alerts = [a for a in self.alerts_buffer if a.timestamp > cutoff_time]
                
                return {
                    'oi_changes': recent_oi,
                    'price_changes': recent_price,
                    'alerts': recent_alerts
                }
        except Exception as e:
            logger.error(f"Error getting recent changes: {e}")
            return {'oi_changes': [], 'price_changes': [], 'alerts': []}
    
    def add_strike_to_monitor(self, strike: float):
        """Dynamically add a strike to monitor"""
        if strike not in self.monitored_strikes:
            self.monitored_strikes.append(strike)
            
            if self.is_streaming and self.ws_connected:
                # Subscribe to the new strike
                try:
                    # Subscribe to call
                    self.breeze.subscribe_feeds(
                        stock_code=self.symbol,
                        exchange_code="NFO",
                        product_type="options",
                        expiry_date=self.expiry_date,
                        strike_price=str(strike),
                        right="Call",
                        get_exchange_quotes=True,
                        get_market_depth=False
                    )
                    
                    # Subscribe to put
                    self.breeze.subscribe_feeds(
                        stock_code=self.symbol,
                        exchange_code="NFO",
                        product_type="options",
                        expiry_date=self.expiry_date,
                        strike_price=str(strike),
                        right="Put",
                        get_exchange_quotes=True,
                        get_market_depth=False
                    )
                    
                    logger.info(f"Added strike {strike} to monitoring")
                except Exception as e:
                    logger.error(f"Error adding strike {strike}: {e}")
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get WebSocket connection status"""
        with self.data_lock:
            return {
                'is_streaming': self.is_streaming,
                'ws_connected': self.ws_connected,
                'tick_count': self.tick_count,
                'monitored_strikes': len(self.monitored_strikes),
                'alerts_count': len(self.alerts_buffer),
                'buffer_size': len(self.oi_changes_buffer),
                'last_update': self.last_data.get('spot_timestamp', None)
            }
    
    def get_current_snapshot(self) -> Dict[str, Any]:
        """Get current market snapshot from WebSocket data"""
        with self.data_lock:
            snapshot = {
                'spot_price': self.last_data.get('spot', 0),
                'spot_change': self.last_data.get('spot_change', 0),
                'spot_change_pct': self.last_data.get('spot_change_pct', 0),
                'options_data': {},
                'timestamp': datetime.now()
            }
            
            # Add option data
            for key, data in self.last_data.items():
                if '_' in key and isinstance(data, dict):
                    strike, option_type = key.split('_')
                    if strike not in snapshot['options_data']:
                        snapshot['options_data'][strike] = {}
                    snapshot['options_data'][strike][option_type] = data
            
            return snapshot

# --- ENHANCED OI FLOW ANALYZER WITH WEBSOCKET ---
class RealTimeOIFlowAnalyzer:
    """Enhanced OI Flow Analyzer with WebSocket real-time capabilities"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.thresholds = config.OI_FLOW_THRESHOLDS
        self.timeframes = config.OI_FLOW_TIMEFRAMES
        self.footprint_buffer: Deque[OIFootprint] = deque(maxlen=10000)
        self.alert_history: List[Dict] = []
        
        # WebSocket streamer instead of polling
        self.streamer: Optional[WebSocketDataStreamer] = None
        self.is_real_time_enabled = False
    
    def start_real_time_analysis(self, breeze, symbol: str, expiry_date: str) -> bool:
        """Start WebSocket-based real-time analysis"""
        try:
            if self.streamer:
                self.streamer.stop_streaming()
            
            # Create WebSocket streamer
            self.streamer = WebSocketDataStreamer(breeze, symbol, expiry_date)
            success = self.streamer.start_streaming()
            
            if success:
                self.is_real_time_enabled = True
                logger.info(f"WebSocket real-time analysis started for {symbol}")
            
            return success
        except WebSocketError as e:
            logger.error(f"WebSocket error: {e}")
            st.error(f"Failed to start real-time streaming: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to start WebSocket analysis: {e}")
            st.error("Failed to start real-time analysis. Please check your connection.")
            return False
    
    def stop_real_time_analysis(self) -> bool:
        """Stop WebSocket-based real-time analysis"""
        try:
            if self.streamer:
                success = self.streamer.stop_streaming()
                if success:
                    self.is_real_time_enabled = False
                    logger.info("WebSocket real-time analysis stopped")
                return success
            return True
        except Exception as e:
            logger.error(f"Failed to stop WebSocket analysis: {e}")
            return False
    
    def get_real_time_status(self) -> Dict[str, Any]:
        """Get current WebSocket status"""
        try:
            if not self.streamer:
                return {
                    'is_streaming': False,
                    'ws_connected': False,
                    'tick_count': 0,
                    'alerts_count': 0,
                    'buffer_size': 0,
                    'last_update': None
                }
            
            return self.streamer.get_connection_status()
        except Exception as e:
            logger.error(f"Error getting WebSocket status: {e}")
            return {
                'is_streaming': False,
                'ws_connected': False,
                'tick_count': 0,
                'alerts_count': 0,
                'buffer_size': 0,
                'last_update': None
            }
    
    def get_real_time_data(self, seconds: int = 60) -> Dict[str, Any]:
        """Get real-time data for specified time window"""
        try:
            if not self.streamer:
                return {'oi_changes': [], 'price_changes': [], 'alerts': []}
            
            return self.streamer.get_recent_changes(seconds)
        except Exception as e:
            logger.error(f"Error getting real-time data: {e}")
            return {'oi_changes': [], 'price_changes': [], 'alerts': []}
    
    def get_current_market_snapshot(self) -> Dict[str, Any]:
        """Get current market snapshot from WebSocket"""
        try:
            if not self.streamer:
                return {}
            
            return self.streamer.get_current_snapshot()
        except Exception as e:
            logger.error(f"Error getting market snapshot: {e}")
            return {}

    # Include all the original methods from EnhancedOIFlowAnalyzer
    def analyze_oi_flow_patterns(self, chain_df: pd.DataFrame, 
                                spot_price: float,
                                timeframe: str = '5min') -> Dict[str, Any]:
        """Core function to analyze OI flow patterns"""
        try:
            # Ensure we have required columns
            if not self._validate_dataframe(chain_df):
                logger.warning("Invalid dataframe for OI flow analysis")
                return self._empty_analysis_results()
            
            analysis_results = {
                'footprints': [],
                'signals': [],
                'manipulation_alerts': [],
                'institutional_activity': [],
                'market_regime': None,
                'key_levels': {}
            }
            
            # 1. Detect Large OI Changes
            footprints = self._detect_oi_footprints(chain_df, spot_price)
            analysis_results['footprints'] = footprints
            
            # 2. Identify Unusual Activity
            unusual_patterns = self._identify_unusual_patterns(chain_df, footprints)
            
            # 3. Detect Manipulation
            manipulation_signals = self._detect_manipulation_patterns(
                chain_df, footprints, unusual_patterns
            )
            analysis_results['manipulation_alerts'] = manipulation_signals
            
            # 4. Track Institutional Flow
            institutional_flow = self._track_institutional_flow(chain_df, timeframe)
            analysis_results['institutional_activity'] = institutional_flow
            
            # 5. Generate Trading Signals
            signals = self._generate_oi_based_signals(
                chain_df, footprints, institutional_flow, spot_price
            )
            analysis_results['signals'] = signals
            
            # 6. Identify Key Levels
            key_levels = self._identify_oi_based_levels(chain_df, spot_price)
            analysis_results['key_levels'] = key_levels
            
            # 7. Determine Market Regime
            regime = self._determine_market_regime(chain_df, footprints)
            analysis_results['market_regime'] = regime
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in OI flow analysis: {e}")
            return self._empty_analysis_results()
    
    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        """Validate dataframe has required columns"""
        required_cols = ['Strike', 'Call OI', 'Put OI', 'Call Chng OI', 
                        'Put Chng OI', 'Call LTP', 'Put LTP', 'Call Volume', 'Put Volume']
        return all(col in df.columns for col in required_cols)
    
    def _empty_analysis_results(self) -> Dict[str, Any]:
        """Return empty analysis results structure"""
        return {
            'footprints': [],
            'signals': [],
            'manipulation_alerts': [],
            'institutional_activity': [],
            'market_regime': 'UNKNOWN',
            'key_levels': {}
        }
    
    def _detect_oi_footprints(self, chain_df: pd.DataFrame, 
                             spot_price: float) -> List[OIFootprint]:
        """Detect significant OI changes"""
        footprints = []
        
        try:
            for _, row in chain_df.iterrows():
                # Call options
                if row['Call OI'] > 0 and abs(row['Call Chng OI']) > 0:
                    oi_change_pct = abs(row['Call Chng OI']) / row['Call OI']
                    
                    if oi_change_pct > self.thresholds['large_oi_change']:
                        is_large = row['Call Volume'] > self.thresholds['institutional_size']
                        
                        # Calculate price change
                        price_change = 0
                        if 'Call Prev Close' in chain_df.columns:
                            price_change = row['Call LTP'] - row['Call Prev Close']
                        
                        # Calculate IV change
                        iv_change = 0
                        if 'Call IV' in chain_df.columns and 'Call Prev IV' in chain_df.columns:
                            iv_change = row['Call IV'] - row.get('Call Prev IV', row['Call IV'])
                        
                        footprint = OIFootprint(
                            timestamp=datetime.now(),
                            strike=row['Strike'],
                            option_type='CALL',
                            oi_change=int(row['Call Chng OI']),
                            volume=int(row['Call Volume']),
                            price_change=price_change,
                            bid_ask_spread=0,  # Can be added if bid/ask data available
                            iv_change=iv_change,
                            large_trade_indicator=is_large,
                            aggressor_side='BUY' if row['Call Chng OI'] > 0 else 'SELL'
                        )
                        footprints.append(footprint)
                        self.footprint_buffer.append(footprint)
                
                # Put options
                if row['Put OI'] > 0 and abs(row['Put Chng OI']) > 0:
                    oi_change_pct = abs(row['Put Chng OI']) / row['Put OI']
                    
                    if oi_change_pct > self.thresholds['large_oi_change']:
                        is_large = row['Put Volume'] > self.thresholds['institutional_size']
                        
                        price_change = 0
                        if 'Put Prev Close' in chain_df.columns:
                            price_change = row['Put LTP'] - row['Put Prev Close']
                        
                        iv_change = 0
                        if 'Put IV' in chain_df.columns and 'Put Prev IV' in chain_df.columns:
                            iv_change = row['Put IV'] - row.get('Put Prev IV', row['Put IV'])
                        
                        footprint = OIFootprint(
                            timestamp=datetime.now(),
                            strike=row['Strike'],
                            option_type='PUT',
                            oi_change=int(row['Put Chng OI']),
                            volume=int(row['Put Volume']),
                            price_change=price_change,
                            bid_ask_spread=0,
                            iv_change=iv_change,
                            large_trade_indicator=is_large,
                            aggressor_side='BUY' if row['Put Chng OI'] > 0 else 'SELL'
                        )
                        footprints.append(footprint)
                        self.footprint_buffer.append(footprint)
        except Exception as e:
            logger.error(f"Error detecting OI footprints: {e}")
        
        return footprints
    
    def _identify_unusual_patterns(self, chain_df: pd.DataFrame, 
                                 footprints: List[OIFootprint]) -> Dict[str, Any]:
        """Identify unusual patterns"""
        try:
            patterns = {
                'sweeps': self._detect_option_sweeps(chain_df),
                'blocks': self._detect_block_trades(chain_df),
                'synthetic_positions': self._detect_synthetic_positions(chain_df),
                'pin_attempts': self._detect_pin_attempts(chain_df, footprints)
            }
            return patterns
        except Exception as e:
            logger.error(f"Error identifying unusual patterns: {e}")
            return {'sweeps': [], 'blocks': [], 'synthetic_positions': [], 'pin_attempts': []}
    
    def _detect_option_sweeps(self, chain_df: pd.DataFrame) -> List[Dict]:
        """Detect option sweeps"""
        sweeps = []
        
        try:
            for _, row in chain_df.iterrows():
                # Call sweeps
                if row['Call Volume'] > self.thresholds['sweep_size']:
                    if row['Call Volume'] > row['Call OI'] * 0.5:  # Volume > 50% of OI
                        sweeps.append({
                            'type': 'CALL_SWEEP',
                            'strike': row['Strike'],
                            'volume': row['Call Volume'],
                            'premium': row['Call LTP'] * row['Call Volume'] * 100,
                            'direction': 'BULLISH'
                        })
                
                # Put sweeps
                if row['Put Volume'] > self.thresholds['sweep_size']:
                    if row['Put Volume'] > row['Put OI'] * 0.5:
                        sweeps.append({
                            'type': 'PUT_SWEEP',
                            'strike': row['Strike'],
                            'volume': row['Put Volume'],
                            'premium': row['Put LTP'] * row['Put Volume'] * 100,
                            'direction': 'BEARISH'
                        })
        except Exception as e:
            logger.error(f"Error detecting option sweeps: {e}")
        
        return sweeps
    
    def _detect_block_trades(self, chain_df: pd.DataFrame) -> List[Dict]:
        """Detect block trades"""
        blocks = []
        
        try:
            # Define block trade threshold
            block_threshold = self.thresholds['institutional_size'] * 2
            
            for _, row in chain_df.iterrows():
                if row['Call Volume'] > block_threshold:
                    blocks.append({
                        'type': 'CALL_BLOCK',
                        'strike': row['Strike'],
                        'size': row['Call Volume'],
                        'premium': row['Call LTP'] * row['Call Volume'] * 100
                    })
                
                if row['Put Volume'] > block_threshold:
                    blocks.append({
                        'type': 'PUT_BLOCK',
                        'strike': row['Strike'],
                        'size': row['Put Volume'],
                        'premium': row['Put LTP'] * row['Put Volume'] * 100
                    })
        except Exception as e:
            logger.error(f"Error detecting block trades: {e}")
        
        return blocks
    
    def _detect_synthetic_positions(self, chain_df: pd.DataFrame) -> List[Dict]:
        """Detect synthetic positions"""
        synthetics = []
        
        try:
            for _, row in chain_df.iterrows():
                # Synthetic long (Long Call + Short Put at same strike)
                if row['Call Chng OI'] > 100 and row['Put Chng OI'] < -100:
                    if abs(row['Call Chng OI']) == abs(row['Put Chng OI']):
                        synthetics.append({
                            'type': 'SYNTHETIC_LONG',
                            'strike': row['Strike'],
                            'size': abs(row['Call Chng OI'])
                        })
                
                # Synthetic short (Short Call + Long Put at same strike)
                elif row['Call Chng OI'] < -100 and row['Put Chng OI'] > 100:
                    if abs(row['Call Chng OI']) == abs(row['Put Chng OI']):
                        synthetics.append({
                            'type': 'SYNTHETIC_SHORT',
                            'strike': row['Strike'],
                            'size': abs(row['Put Chng OI'])
                        })
        except Exception as e:
            logger.error(f"Error detecting synthetic positions: {e}")
        
        return synthetics
    
    def _detect_pin_attempts(self, chain_df: pd.DataFrame, 
                           footprints: List[OIFootprint]) -> List[Dict]:
        """Detect potential pin attempts"""
        pin_attempts = []
        
        try:
            # Find strikes with highest OI
            max_call_oi_strike = chain_df.loc[chain_df['Call OI'].idxmax(), 'Strike']
            max_put_oi_strike = chain_df.loc[chain_df['Put OI'].idxmax(), 'Strike']
            
            # Check for concentrated activity around these strikes
            for strike in [max_call_oi_strike, max_put_oi_strike]:
                strike_footprints = [fp for fp in footprints if fp.strike == strike]
                
                if len(strike_footprints) > 3:  # Multiple transactions
                    pin_attempts.append({
                        'strike': strike,
                        'activity_count': len(strike_footprints),
                        'net_oi_change': sum(fp.oi_change for fp in strike_footprints)
                    })
        except Exception as e:
            logger.error(f"Error detecting pin attempts: {e}")
        
        return pin_attempts
    
    def _detect_manipulation_patterns(self, chain_df: pd.DataFrame,
                                    footprints: List[OIFootprint],
                                    unusual_patterns: Dict) -> List[Dict]:
        """Detect potential manipulation patterns"""
        alerts = []
        
        try:
            # Group footprints by strike
            strike_footprints = {}
            for fp in footprints:
                if fp.strike not in strike_footprints:
                    strike_footprints[fp.strike] = []
                strike_footprints[fp.strike].append(fp)
            
            for strike, fps in strike_footprints.items():
                if len(fps) >= 2:
                    # Check for pump and dump pattern
                    initial_buildup = sum(fp.oi_change for fp in fps if fp.oi_change > 0)
                    subsequent_unwind = sum(fp.oi_change for fp in fps if fp.oi_change < 0)
                    
                    if initial_buildup > 1000 and abs(subsequent_unwind) > initial_buildup * 0.5:
                        alerts.append({
                            'type': 'PUMP_DUMP_ALERT',
                            'strike': strike,
                            'severity': 'HIGH',
                            'buildup_size': initial_buildup,
                            'unwind_size': abs(subsequent_unwind),
                            'recommendation': 'AVOID - Potential manipulation detected'
                        })
        except Exception as e:
            logger.error(f"Error detecting manipulation patterns: {e}")
        
        return alerts
    
    def _track_institutional_flow(self, chain_df: pd.DataFrame, 
                                timeframe: str) -> List[Dict]:
        """Track institutional flow patterns"""
        institutional_flows = []
        
        try:
            # Size thresholds by timeframe
            size_thresholds = {
                '5min': 100,
                '10min': 200,
                '30min': 500,
                '1hour': 1000,
                '2hour': 2000,
                'daily': 5000
            }
            
            threshold = size_thresholds.get(timeframe, 500)
            
            for _, row in chain_df.iterrows():
                # Call side
                if row['Call Volume'] > threshold or abs(row['Call Chng OI']) > threshold:
                    flow_type = self._classify_institutional_flow(row, 'CALL')
                    if flow_type:
                        institutional_flows.append({
                            'strike': row['Strike'],
                            'type': 'CALL',
                            'flow_type': flow_type,
                            'size': max(row['Call Volume'], abs(row['Call Chng OI'])),
                            'direction': 'LONG' if row['Call Chng OI'] > 0 else 'SHORT',
                            'premium_involved': row['Call LTP'] * row['Call Volume'] * 100
                        })
                
                # Put side
                if row['Put Volume'] > threshold or abs(row['Put Chng OI']) > threshold:
                    flow_type = self._classify_institutional_flow(row, 'PUT')
                    if flow_type:
                        institutional_flows.append({
                            'strike': row['Strike'],
                            'type': 'PUT',
                            'flow_type': flow_type,
                            'size': max(row['Put Volume'], abs(row['Put Chng OI'])),
                            'direction': 'LONG' if row['Put Chng OI'] > 0 else 'SHORT',
                            'premium_involved': row['Put LTP'] * row['Put Volume'] * 100
                        })
        except Exception as e:
            logger.error(f"Error tracking institutional flow: {e}")
        
        return institutional_flows
    
    def _classify_institutional_flow(self, row: pd.Series, option_type: str) -> Optional[str]:
        """Classify the type of institutional flow"""
        try:
            if option_type == 'CALL':
                oi_change = row['Call Chng OI']
                price_change = row.get('Call Price Change', 0)
                
                # If we don't have price change, estimate from current data
                if price_change == 0 and 'Call Prev Close' in row:
                    price_change = row['Call LTP'] - row['Call Prev Close']
            else:
                oi_change = row['Put Chng OI']
                price_change = row.get('Put Price Change', 0)
                
                if price_change == 0 and 'Put Prev Close' in row:
                    price_change = row['Put LTP'] - row['Put Prev Close']
            
            # Classification
            if oi_change > 0 and price_change > 0:
                return 'LONG_BUILDUP'
            elif oi_change > 0 and price_change < 0:
                return 'SHORT_BUILDUP'
            elif oi_change < 0 and price_change > 0:
                return 'SHORT_COVERING'
            elif oi_change < 0 and price_change < 0:
                return 'LONG_UNWINDING'
            else:
                return None
        except Exception as e:
            logger.error(f"Error classifying institutional flow: {e}")
            return None
    
    def _generate_oi_based_signals(self, chain_df: pd.DataFrame,
                                  footprints: List[OIFootprint],
                                  institutional_flow: List[Dict],
                                  spot_price: float) -> List[Dict]:
        """Generate actionable trading signals"""
        signals = []
        
        try:
            # 1. Breakout/Breakdown Signals
            call_oi_sorted = chain_df.nlargest(5, 'Call OI')
            for _, row in call_oi_sorted.iterrows():
                if row['Strike'] > spot_price and row['Strike'] < spot_price * 1.02:
                    if row['Call Chng OI'] < -1000:  # Significant unwinding
                        signals.append({
                            'type': 'BREAKOUT',
                            'strike': row['Strike'],
                            'action': 'BUY',
                            'target': row['Strike'] * 1.01,
                            'stop_loss': spot_price * 0.995,
                            'strength': 0.8,
                            'reason': f"Heavy call unwinding at resistance {row['Strike']}"
                        })
            
            # 2. Support Signals
            put_oi_sorted = chain_df.nlargest(5, 'Put OI')
            for _, row in put_oi_sorted.iterrows():
                if row['Strike'] < spot_price and row['Strike'] > spot_price * 0.98:
                    if row['Put Chng OI'] < -1000:  # Significant unwinding
                        signals.append({
                            'type': 'BREAKDOWN',
                            'strike': row['Strike'],
                            'action': 'SELL',
                            'target': row['Strike'] * 0.99,
                            'stop_loss': spot_price * 1.005,
                            'strength': 0.8,
                            'reason': f"Heavy put unwinding at support {row['Strike']}"
                        })
            
            # 3. Institutional Flow Signals
            for flow in institutional_flow:
                if flow['flow_type'] == 'LONG_BUILDUP' and flow['type'] == 'CALL':
                    signals.append({
                        'type': 'INSTITUTIONAL_LONG',
                        'strike': flow['strike'],
                        'action': 'BUY',
                        'strength': 0.7,
                        'reason': f"Institutional call buying at {flow['strike']}"
                    })
            
            # Sort by strength
            signals = sorted(signals, key=lambda x: x['strength'], reverse=True)
        except Exception as e:
            logger.error(f"Error generating OI-based signals: {e}")
        
        return signals[:10]
    
    def _identify_oi_based_levels(self, chain_df: pd.DataFrame, 
                                 spot_price: float) -> Dict[str, List[float]]:
        """Identify key support/resistance levels from OI"""
        levels = {
            'resistance': [],
            'support': [],
            'max_pain': 0
        }
        
        try:
            # Resistance levels (high call OI)
            call_oi_sorted = chain_df.nlargest(5, 'Call OI')
            levels['resistance'] = call_oi_sorted[
                call_oi_sorted['Strike'] > spot_price
            ]['Strike'].tolist()[:3]
            
            # Support levels (high put OI)
            put_oi_sorted = chain_df.nlargest(5, 'Put OI')
            levels['support'] = put_oi_sorted[
                put_oi_sorted['Strike'] < spot_price
            ]['Strike'].tolist()[:3]
        except Exception as e:
            logger.error(f"Error identifying OI-based levels: {e}")
        
        return levels
    
    def _determine_market_regime(self, chain_df: pd.DataFrame, 
                               footprints: List[OIFootprint]) -> str:
        """Determine current market regime"""
        try:
            # Count bullish vs bearish footprints
            bullish_count = sum(1 for fp in footprints 
                               if fp.aggressor_side == 'BUY' and fp.option_type == 'CALL')
            bearish_count = sum(1 for fp in footprints 
                               if fp.aggressor_side == 'BUY' and fp.option_type == 'PUT')
            
            # Calculate PCR
            total_put_oi = chain_df['Put OI'].sum()
            total_call_oi = chain_df['Call OI'].sum()
            pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1
            
            # Determine regime
            if bullish_count > bearish_count * 1.5 and pcr < 0.8:
                return 'STRONGLY_BULLISH'
            elif bullish_count > bearish_count and pcr < 1:
                return 'BULLISH'
            elif bearish_count > bullish_count * 1.5 and pcr > 1.2:
                return 'STRONGLY_BEARISH'
            elif bearish_count > bullish_count and pcr > 1:
                return 'BEARISH'
            else:
                return 'NEUTRAL'
        except Exception as e:
            logger.error(f"Error determining market regime: {e}")
            return 'UNKNOWN'

# --- HELPER & SETUP FUNCTIONS ---
def load_credentials() -> Tuple[str, str]:
    """Load API credentials from secrets or environment"""
    try:
        if 'BREEZE_API_KEY' in st.secrets:
            return st.secrets["BREEZE_API_KEY"], st.secrets["BREEZE_API_SECRET"]
        else:
            load_dotenv()
            return os.getenv("BREEZE_API_KEY"), os.getenv("BREEZE_API_SECRET")
    except Exception as e:
        logger.error(f"Error loading credentials: {e}")
        st.error("Failed to load API credentials")
        return None, None

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
    try:
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
    except Exception as e:
        logger.error(f"Error normalizing column names: {e}")
        return df

def validate_option_data(df: pd.DataFrame) -> bool:
    """Validate option chain data integrity"""
    try:
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
    except Exception as e:
        logger.error(f"Error validating option data: {e}")
        return False

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
    try:
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
        
        return results.round(4)
    except Exception as e:
        logger.error(f"Error calculating Greeks: {e}")
        return pd.DataFrame(columns=['delta', 'gamma', 'vega', 'theta', 'rho']).fillna(0)

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
    """Process raw options data and calculate Greeks with robust error handling"""
    try:
        if not raw_data:
            st.warning("No options data received.")
            return pd.DataFrame()
        
        df = pd.DataFrame(raw_data)
        
        # Normalize column names first
        df = normalize_column_names(df)
        
        # Validate data after normalization
        if not validate_option_data(df):
            return pd.DataFrame()
        
        # Convert to numeric with error handling
        numeric_columns = ['oi', 'oi_change', 'ltp', 'volume', 'strike_price']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Separate calls and puts
        calls = df[df['right'].str.upper() == 'CALL'].copy()
        puts = df[df['right'].str.upper() == 'PUT'].copy()
        
        if calls.empty and puts.empty:
            st.warning("No valid call or put data found")
            return pd.DataFrame()
        
        # Merge into chain
        chain = pd.merge(calls, puts, on="strike_price", suffixes=('_call', '_put'), how="outer")
        chain = chain.sort_values("strike_price").fillna(0)
        
        # Calculate Time to Expiry in years
        try:
            expiry_dt = datetime.strptime(expiry_date, "%d-%b-%Y")
            t = max((expiry_dt - datetime.now()).total_seconds() / (365 * 24 * 3600), 0)
        except ValueError:
            logger.warning(f"Could not parse expiry date: {expiry_date}")
            t = 0.1  # Default to ~36 days
        
        if t > 0:
            # Vectorized IV calculation with error handling
            try:
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
            except Exception as e:
                logger.error(f"Error calculating IV: {e}")
                chain['Call IV'] = 0
                chain['Put IV'] = 0
            
            # Calculate Greeks using vectorized function
            try:
                strikes = chain['strike_price'].values
                call_ivs = chain['Call IV'].values / 100
                put_ivs = chain['Put IV'].values / 100
                
                call_greeks = calculate_greeks_vectorized(call_ivs, 'Call', spot_price, strikes, t)
                put_greeks = calculate_greeks_vectorized(put_ivs, 'Put', spot_price, strikes, t)
                
                # Add Greeks to chain
                chain = pd.concat([chain, 
                                  call_greeks.add_prefix('call_'), 
                                  put_greeks.add_prefix('put_')], axis=1)
            except Exception as e:
                logger.error(f"Error calculating Greeks: {e}")
                # Add empty Greeks columns
                greek_cols = ['delta', 'gamma', 'vega', 'theta', 'rho']
                for prefix in ['call_', 'put_']:
                    for col in greek_cols:
                        chain[f"{prefix}{col}"] = 0
        
        # Rename columns for display
        chain.rename(columns={
            'oi_call': 'Call OI', 'oi_change_call': 'Call Chng OI', 'ltp_call': 'Call LTP',
            'strike_price': 'Strike', 'ltp_put': 'Put LTP', 'oi_change_put': 'Put Chng OI',
            'oi_put': 'Put OI', 'volume_call': 'Call Volume', 'volume_put': 'Put Volume'
        }, inplace=True)
        
        # Add update timestamp for OI tracker
        if 'oi_tracker' in st.session_state:
            st.session_state.oi_tracker.add_snapshot(chain, datetime.now())
        
        return chain
    except Exception as e:
        logger.error(f"Error processing and analyzing data: {e}")
        st.error(f"Error processing data: {e}")
        return pd.DataFrame()

def calculate_dashboard_metrics(chain_df: pd.DataFrame, spot_price: float) -> Dict[str, Any]:
    """Calculate key metrics from options chain with robust error handling"""
    try:
        if chain_df.empty:
            logger.warning("Empty dataframe provided to calculate_dashboard_metrics")
            return {
                'max_pain': 0, 'resistance': [], 'support': [], 'pcr': 0,
                'net_oi_change': 0, 'sentiment': 0, 'total_call_oi': 0, 'total_put_oi': 0
            }
        
        # Vectorized Max Pain calculation
        strikes = chain_df['Strike'].values
        call_oi = chain_df['Call OI'].values
        put_oi = chain_df['Put OI'].values
        
        # Handle edge cases
        if len(strikes) == 0:
            max_pain = 0
        else:
            try:
                strike_matrix = strikes[:, np.newaxis]
                call_pain = np.sum(np.maximum(strike_matrix - strikes, 0) * call_oi, axis=1)
                put_pain = np.sum(np.maximum(strikes - strike_matrix, 0) * put_oi, axis=1)
                total_pain = call_pain + put_pain
                max_pain = strikes[np.argmin(total_pain)] if len(total_pain) > 0 else 0
            except Exception as e:
                logger.error(f"Error calculating max pain: {e}")
                max_pain = 0
        
        # PCR and other metrics
        total_call_oi = chain_df['Call OI'].sum()
        total_put_oi = chain_df['Put OI'].sum()
        pcr = round(total_put_oi / total_call_oi if total_call_oi > 0 else 0, 2)
        net_oi_change = chain_df['Put Chng OI'].sum() - chain_df['Call Chng OI'].sum()
        
        # Enhanced Sentiment Score with error handling
        sentiment_score = 0
        
        try:
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
            if max_pain > 0:
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
                try:
                    atm_idx = (chain_df['Strike'] - spot_price).abs().idxmin()
                    if 0 < atm_idx < len(chain_df) - 1:
                        call_iv_skew = chain_df.loc[atm_idx, 'Call IV'] - chain_df['Call IV'].mean()
                        put_iv_skew = chain_df.loc[atm_idx, 'Put IV'] - chain_df['Put IV'].mean()
                        if put_iv_skew > call_iv_skew:
                            sentiment_score += 10
                        else:
                            sentiment_score -= 10
                except Exception as e:
                    logger.warning(f"Error in IV skew analysis: {e}")
        except Exception as e:
            logger.error(f"Error calculating sentiment score: {e}")
        
        # Get resistance and support levels
        try:
            resistance = chain_df.nlargest(3, 'Call OI')['Strike'].tolist()
            support = chain_df.nlargest(3, 'Put OI')['Strike'].tolist()
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            resistance = []
            support = []
        
        return {
            'max_pain': max_pain,
            'resistance': resistance,
            'support': support,
            'pcr': pcr,
            'net_oi_change': net_oi_change,
            'sentiment': max(-100, min(100, sentiment_score)),
            'total_call_oi': total_call_oi,
            'total_put_oi': total_put_oi
        }
    except Exception as e:
        logger.error(f"Error calculating dashboard metrics: {e}")
        return {
            'max_pain': 0,
            'resistance': [],
            'support': [],
            'pcr': 0,
            'net_oi_change': 0,
            'sentiment': 0,
            'total_call_oi': 0,
            'total_put_oi': 0
        }

# --- VISUALIZATION FUNCTIONS ---
def create_iv_smile_chart(chain_df: pd.DataFrame, spot_price: float) -> Optional[go.Figure]:
    """Create IV smile chart with error handling"""
    try:
        iv_data = []
        
        # Collect IV data
        for _, row in chain_df.iterrows():
            try:
                if row.get('Call IV', 0) > 0:
                    iv_data.append({'Strike': row['Strike'], 'IV': row['Call IV'], 'Type': 'Call'})
                if row.get('Put IV', 0) > 0:
                    iv_data.append({'Strike': row['Strike'], 'IV': row['Put IV'], 'Type': 'Put'})
            except Exception as e:
                logger.warning(f"Error processing IV data for strike {row.get('Strike', 'Unknown')}: {e}")
                continue
        
        if not iv_data:
            logger.info("No IV data available for chart")
            return None
        
        iv_df = pd.DataFrame(iv_data)
        
        fig = go.Figure()
        
        # Add traces for each option type
        for option_type in ['Call', 'Put']:
            data = iv_df[iv_df['Type'] == option_type]
            if not data.empty:
                color = 'red' if option_type == 'Call' else 'green'
                fig.add_trace(go.Scatter(
                    x=data['Strike'], 
                    y=data['IV'],
                    mode='lines+markers',
                    name=f'{option_type} IV',
                    line=dict(width=2, color=color),
                    marker=dict(size=6),
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
    except Exception as e:
        logger.error(f"Error creating IV smile chart: {e}")
        return None

def create_volume_profile(chain_df: pd.DataFrame) -> go.Figure:
    """Create volume profile chart with error handling"""
    try:
        fig = go.Figure()
        
        # Check if volume columns exist
        if 'Call Volume' not in chain_df.columns or 'Put Volume' not in chain_df.columns:
            fig.add_annotation(text="Volume data not available", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        # Filter out zero volumes for cleaner chart
        volume_data = chain_df[(chain_df['Call Volume'] > 0) | (chain_df['Put Volume'] > 0)]
        
        if volume_data.empty:
            fig.add_annotation(text="No volume data available", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        fig.add_trace(go.Bar(
            x=volume_data['Strike'],
            y=volume_data['Call Volume'],
            name='Call Volume',
            marker_color='rgba(239, 83, 80, 0.7)',
            hovertemplate='Strike: %{x}<br>Call Volume: %{y:,.0f}<extra></extra>'
        ))
        
        fig.add_trace(go.Bar(
            x=volume_data['Strike'],
            y=volume_data['Put Volume'],
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
    except Exception as e:
        logger.error(f"Error creating volume profile: {e}")
        return go.Figure()

def display_sentiment_gauge(sentiment_score: float) -> go.Figure:
    """Create sentiment gauge chart with error handling"""
    try:
        # Ensure sentiment score is within bounds
        sentiment_score = max(-100, min(100, sentiment_score))
        
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
    except Exception as e:
        logger.error(f"Error creating sentiment gauge: {e}")
        return go.Figure()

def create_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create premium heatmap with error handling"""
    try:
        if df.empty or 'Strike' not in df.columns:
            fig = go.Figure()
            fig.add_annotation(text="No data available for heatmap", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        # Check if required columns exist
        required_cols = ['Call LTP', 'Put LTP']
        if not all(col in df.columns for col in required_cols):
            fig = go.Figure()
            fig.add_annotation(text="Premium data not available", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        heat_df = df.set_index('Strike')[['Call LTP', 'Put LTP']].sort_index(ascending=False)
        
        # Filter out rows with all zeros
        heat_df = heat_df[(heat_df['Call LTP'] > 0) | (heat_df['Put LTP'] > 0)]
        
        if heat_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No premium data available", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
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
    except Exception as e:
        logger.error(f"Error creating heatmap: {e}")
        return go.Figure()

def create_greeks_surface(chain_df: pd.DataFrame, greek: str, option_type: str) -> Optional[go.Figure]:
    """Create 3D surface plot for Greeks with error handling"""
    try:
        greek_col = f"{option_type.lower()}_{greek}"
        if greek_col not in chain_df.columns:
            logger.info(f"Greek column {greek_col} not found in dataframe")
            return None
        
        # Filter out zero values
        data_subset = chain_df[chain_df[greek_col] != 0]
        if data_subset.empty:
            logger.info(f"No {greek} data available for {option_type}")
            return None
        
        # Create meshgrid for surface plot
        strikes = data_subset['Strike'].values
        greek_values = data_subset[greek_col].values
        
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
    except Exception as e:
        logger.error(f"Error creating Greeks surface: {e}")
        return None

def track_historical_data_efficient(symbol: str, expiry: str, metrics: Dict[str, Any]) -> None:
    """Efficient historical data tracking with compression and error handling"""
    try:
        if 'historical_data' not in st.session_state:
            st.session_state.historical_data = pd.DataFrame()
        
        # Validate metrics data
        if not isinstance(metrics, dict) or not metrics:
            logger.warning("Invalid metrics data provided for historical tracking")
            return
        
        new_row = pd.DataFrame([{
            'timestamp': datetime.now(),
            'symbol': symbol,
            'expiry': expiry,
            **metrics
        }])
        
        # Append and trim to max records
        st.session_state.historical_data = pd.concat([
            st.session_state.historical_data, 
            new_row
        ], ignore_index=True).tail(config.MAX_HISTORICAL_RECORDS)
        
        logger.info(f"Added historical data point. Total records: {len(st.session_state.historical_data)}")
        
    except Exception as e:
        logger.error(f"Error tracking historical data: {e}")

def prepare_export_data(df: pd.DataFrame, format_type: str) -> Optional[pd.DataFrame]:
    """Prepare and validate data for export with error handling"""
    try:
        if df.empty:
            st.error("No data to export")
            return None
        
        # Create a copy to avoid modifying original
        export_df = df.copy()
        
        # Remove any infinite or NaN values
        export_df = export_df.replace([np.inf, -np.inf], np.nan)
        export_df = export_df.fillna(0)
        
        # Format based on export type
        if format_type == "Excel":
            # Ensure numeric columns are properly formatted
            numeric_cols = export_df.select_dtypes(include=[np.number]).columns
            export_df[numeric_cols] = export_df[numeric_cols].round(2)
        
        return export_df
    except Exception as e:
        logger.error(f"Error preparing export data: {e}")
        return None

def create_strategy_payoff(chain_df: pd.DataFrame, spot_price: float) -> go.Figure:
    """Create strategy payoff diagram with error handling"""
    try:
        if chain_df.empty or 'Strike' not in chain_df.columns:
            fig = go.Figure()
            fig.add_annotation(text="No data available for strategy analysis", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        strikes = chain_df['Strike'].values
        
        # Find ATM strike
        atm_idx = (chain_df['Strike'] - spot_price).abs().idxmin()
        atm_strike = chain_df.loc[atm_idx, 'Strike']
        
        # Get premiums
        call_premium = chain_df.loc[atm_idx, 'Call LTP'] if 'Call LTP' in chain_df.columns else 0
        put_premium = chain_df.loc[atm_idx, 'Put LTP'] if 'Put LTP' in chain_df.columns else 0
        
        if call_premium == 0 and put_premium == 0:
            fig = go.Figure()
            fig.add_annotation(text="No premium data available for strategy analysis", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        # Calculate payoff (Long Straddle example)
        spot_range = np.linspace(strikes.min(), strikes.max(), 100)
        straddle_payoff = (np.maximum(spot_range - atm_strike, 0) + 
                          np.maximum(atm_strike - spot_range, 0) - 
                          (call_premium + put_premium))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=spot_range,
            y=straddle_payoff,
            mode='lines',
            name='Long Straddle',
            line=dict(width=3),
            hovertemplate='Spot: %{x:.0f}<br>P&L: %{y:.2f}<extra></extra>'
        ))
        
        # Add breakeven lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Breakeven")
        fig.add_vline(x=spot_price, line_dash="dash", line_color="blue", annotation_text="Current Spot")
        
        fig.update_layout(
            title=f'Long Straddle Payoff (Strike: {atm_strike})',
            xaxis_title='Spot Price at Expiry',
            yaxis_title='Profit/Loss',
            height=400,
            hovermode='x unified'
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating strategy payoff: {e}")
        return go.Figure()

# --- REAL-TIME DASHBOARD FUNCTIONS ---
def create_real_time_dashboard() -> None:
    """Create real-time monitoring dashboard with WebSocket support and error handling"""
    try:
        st.subheader("🔴 LIVE - Real-Time WebSocket Monitor")
        
        # Initialize analyzer if not exists
        if 'rt_analyzer' not in st.session_state:
            st.session_state.rt_analyzer = RealTimeOIFlowAnalyzer(config)
        
        rt_analyzer = st.session_state.rt_analyzer
        
        # Get required data
        breeze = st.session_state.get('breeze_connection')
        symbol = st.session_state.get('current_symbol')
        expiry = st.session_state.get('current_expiry')
        
        # Check data availability
        data_ready = all([breeze, symbol, expiry])
        
        # Real-time status
        col1, col2, col3, col4 = st.columns(4)
        
        status = rt_analyzer.get_real_time_status() if rt_analyzer else {
            'is_streaming': False, 'ws_connected': False, 'tick_count': 0, 
            'alerts_count': 0, 'buffer_size': 0, 'last_update': None
        }
        
        with col1:
            if status['is_streaming'] and status['ws_connected']:
                st.success("🟢 WEBSOCKET LIVE")
                st.metric("Ticks Received", f"{status['tick_count']:,}")
            elif data_ready:
                st.warning("⚪ READY TO START")
                st.metric("Status", "Ready")
            else:
                st.info("⚪ WAITING FOR DATA")
                st.metric("Status", "Waiting")
        
        with col2:
            st.metric("Live Alerts", status['alerts_count'])
            if status.get('last_update'):
                time_diff = (datetime.now() - status['last_update']).seconds
                st.caption(f"Last update: {time_diff}s ago")
            else:
                st.caption("No updates yet")
        
        with col3:
            if status['ws_connected']:
                st.success("✅ WebSocket Connected")
            else:
                st.error("❌ WebSocket Disconnected")
            st.metric("Strikes Monitored", status.get('monitored_strikes', 0))
        
        with col4:
            st.metric("Data Buffer", status['buffer_size'])
            current_time = datetime.now().strftime("%H:%M:%S")
            st.caption(f"Time: {current_time}")
        
        # Control buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if data_ready:
                if not status['is_streaming']:
                    if st.button("🚀 Start WebSocket", type="primary", use_container_width=True):
                        try:
                            with st.spinner("Connecting to WebSocket..."):
                                success = rt_analyzer.start_real_time_analysis(breeze, symbol, expiry)
                            if success:
                                st.success("WebSocket streaming started!")
                                st.rerun()
                            else:
                                st.error("Failed to start WebSocket streaming")
                        except WebSocketError as e:
                            st.error(f"WebSocket Error: {e}")
                        except Exception as e:
                            st.error(f"Error starting real-time: {e}")
                            logger.error(f"Real-time start error: {e}")
                else:
                    st.success("🟢 STREAMING ACTIVE")
            else:
                st.button("🚀 Start WebSocket", disabled=True, use_container_width=True)
                st.caption("Load options data first")
        
        with col2:
            if st.button("🔴 Stop Streaming", use_container_width=True):
                if rt_analyzer:
                    try:
                        success = rt_analyzer.stop_real_time_analysis()
                        if success:
                            st.info("WebSocket streaming stopped")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error stopping real-time: {e}")
        
        with col3:
            if st.button("🗑️ Clear Buffers", use_container_width=True):
                if rt_analyzer and rt_analyzer.streamer:
                    try:
                        rt_analyzer.streamer.oi_changes_buffer.clear()
                        rt_analyzer.streamer.price_changes_buffer.clear()
                        rt_analyzer.streamer.alerts_buffer.clear()
                        st.info("Buffers cleared")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing buffers: {e}")
        
        # Real-time data display
        if rt_analyzer and status['is_streaming'] and status['ws_connected']:
            # Get real-time data
            rt_data = rt_analyzer.get_real_time_data(60)  # Last 60 seconds
            
            # Live Market Snapshot
            if rt_analyzer.streamer:
                snapshot = rt_analyzer.get_current_market_snapshot()
                if snapshot.get('spot_price'):
                    st.subheader("📈 Live Market Snapshot")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Spot Price", f"₹{snapshot['spot_price']:,.2f}",
                                 f"{snapshot['spot_change_pct']:+.2f}%")
                    
                    with col2:
                        total_call_oi_change = sum(c['oi_change'] for c in rt_data['oi_changes'] 
                                                 if c['type'] == 'CALL')
                        st.metric("Call OI Change (1m)", f"{total_call_oi_change:+,}")
                    
                    with col3:
                        total_put_oi_change = sum(c['oi_change'] for c in rt_data['oi_changes'] 
                                                if c['type'] == 'PUT')
                        st.metric("Put OI Change (1m)", f"{total_put_oi_change:+,}")
            
            # Real-time alerts
            if rt_data['alerts']:
                st.subheader("🚨 Live Alerts (Last 60 seconds)")
                
                # Filter alerts by severity
                severity_filter = st.selectbox("Filter by Severity", ["ALL", "HIGH", "MEDIUM", "LOW"])
                
                filtered_alerts = rt_data['alerts']
                if severity_filter != "ALL":
                    filtered_alerts = [a for a in filtered_alerts if a.severity == severity_filter]
                
                # Display recent alerts
                for alert in reversed(filtered_alerts[-10:]):  # Show last 10 alerts
                    timestamp = alert.timestamp.strftime("%H:%M:%S")
                    
                    if alert.severity == 'HIGH':
                        st.error(f"🔥 {timestamp} - {alert.message}")
                    elif alert.severity == 'MEDIUM':
                        st.warning(f"⚠️ {timestamp} - {alert.message}")
                    else:
                        st.info(f"ℹ️ {timestamp} - {alert.message}")
            
            # Real-time OI changes visualization
            if rt_data['oi_changes']:
                st.subheader("📊 Live OI Flow (Last 60 seconds)")
                
                # Aggregate OI changes by strike
                call_changes = {}
                put_changes = {}
                
                for change in rt_data['oi_changes']:
                    strike = change['strike']
                    if change['type'] == 'CALL':
                        if strike not in call_changes:
                            call_changes[strike] = 0
                        call_changes[strike] += change['oi_change']
                    else:
                        if strike not in put_changes:
                            put_changes[strike] = 0
                        put_changes[strike] += change['oi_change']
                
                if call_changes or put_changes:
                    # Create real-time chart
                    fig = go.Figure()
                    
                    if call_changes:
                        sorted_strikes = sorted(call_changes.keys())
                        fig.add_trace(go.Bar(
                            x=sorted_strikes,
                            y=[call_changes[s] for s in sorted_strikes],
                            name='Call OI Changes',
                            marker_color='red',
                            opacity=0.7
                        ))
                    
                    if put_changes:
                        sorted_strikes = sorted(put_changes.keys())
                        fig.add_trace(go.Bar(
                            x=sorted_strikes,
                            y=[put_changes[s] for s in sorted_strikes],
                            name='Put OI Changes',
                            marker_color='green',
                            opacity=0.7
                        ))
                    
                    fig.update_layout(
                        title="Real-Time OI Changes (WebSocket Feed)",
                        xaxis_title="Strike Price",
                        yaxis_title="OI Change",
                        height=400,
                        barmode='group'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Real-time statistics
            st.subheader("📊 Live Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if rt_data['oi_changes']:
                    total_oi_changes = len(rt_data['oi_changes'])
                    st.metric("OI Updates (1m)", total_oi_changes)
            
            with col2:
                if rt_data['price_changes']:
                    significant_moves = len([p for p in rt_data['price_changes'] 
                                           if abs(p['price_change_pct']) > 5])
                    st.metric("Significant Moves", significant_moves)
            
            with col3:
                high_alerts = len([a for a in rt_data['alerts'] if a.severity == 'HIGH'])
                st.metric("High Alerts", high_alerts)
            
            with col4:
                sweep_alerts = len([a for a in rt_data['alerts'] if a.alert_type == 'OPTION_SWEEP'])
                st.metric("Sweep Alerts", sweep_alerts)
        
        else:
            st.info("Start WebSocket streaming to see live data")
            st.markdown("""
            ### 🚀 WebSocket Streaming Benefits:
            - **Real-time tick data** - No polling delays
            - **No API rate limits** - Single persistent connection
            - **Lower latency** - Instant market updates
            - **Professional grade** - Same technology used by trading systems
            """)
    
    except Exception as e:
        logger.error(f"Error in real-time dashboard: {e}")
        st.error(f"Error creating real-time dashboard: {str(e)}")

# --- OI FLOW INTEGRATION FUNCTIONS ---
def create_oi_flow_dashboard(analyzer: RealTimeOIFlowAnalyzer, 
                           analysis_results: Dict[str, Any],
                           chain_df: pd.DataFrame,
                           spot_price: float) -> None:
    """Create OI flow dashboard integrated with existing UI"""
    
    try:
        # Header metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_footprints = len(analysis_results.get('footprints', []))
            st.metric("OI Footprints", total_footprints)
        
        with col2:
            inst_flows = len(analysis_results.get('institutional_activity', []))
            st.metric("Institutional Flows", inst_flows)
        
        with col3:
            signals = analysis_results.get('signals', [])
            active_signals = len([s for s in signals if s.get('strength', 0) > 0.7])
            st.metric("Active Signals", active_signals)
        
        with col4:
            regime = analysis_results.get('market_regime', 'UNKNOWN')
            regime_color = "🟢" if "BULLISH" in regime else "🔴" if "BEARISH" in regime else "⚪"
            st.metric("Market Regime", f"{regime_color} {regime}")
        
        # Alerts
        manipulation_alerts = analysis_results.get('manipulation_alerts', [])
        if manipulation_alerts:
            st.error("⚠️ Manipulation Alerts Detected!")
            for alert in manipulation_alerts:
                st.warning(f"{alert.get('type', 'Unknown')}: Strike {alert.get('strike', 'N/A')} - {alert.get('recommendation', 'No recommendation')}")
        
        # Create visualizations
        _create_oi_footprint_chart(analysis_results.get('footprints', []), chain_df, spot_price)
        
        # Signals table
        if signals:
            st.subheader("📊 Trading Signals")
            try:
                signals_df = pd.DataFrame(signals)
                if 'strength' in signals_df.columns:
                    st.dataframe(
                        signals_df.style.background_gradient(subset=['strength'], cmap='RdYlGn'),
                        use_container_width=True
                    )
                else:
                    st.dataframe(signals_df, use_container_width=True)
            except Exception as e:
                logger.error(f"Error displaying signals: {e}")
                st.error("Error displaying trading signals")
    
    except Exception as e:
        logger.error(f"Error in OI flow dashboard: {e}")
        st.error(f"Error creating OI flow dashboard: {str(e)}")

def _create_oi_footprint_chart(footprints: List[OIFootprint], 
                              chain_df: pd.DataFrame,
                              spot_price: float) -> None:
    """Create OI footprint visualization with error handling"""
    try:
        if not footprints:
            st.info("No significant OI footprints detected")
            return
        
        if chain_df.empty or 'Strike' not in chain_df.columns:
            st.error("Invalid chain data for footprint analysis")
            return
        
        # Prepare data
        strikes = sorted(chain_df['Strike'].unique())
        call_footprint_data = {strike: 0 for strike in strikes}
        put_footprint_data = {strike: 0 for strike in strikes}
        
        for fp in footprints:
            if fp.strike in strikes:
                if fp.option_type == 'CALL':
                    call_footprint_data[fp.strike] += fp.oi_change
                else:
                    put_footprint_data[fp.strike] += fp.oi_change
        
        # Create figure
        fig = go.Figure()
        
        # Add call footprints
        call_strikes = [k for k, v in call_footprint_data.items() if v != 0]
        if call_strikes:
            fig.add_trace(go.Bar(
                x=call_strikes,
                y=[call_footprint_data[s] for s in call_strikes],
                name='Call OI Changes',
                marker_color='rgba(239, 83, 80, 0.7)',
                hovertemplate='Strike: %{x}<br>Call OI Change: %{y:,.0f}<extra></extra>'
            ))
        
        # Add put footprints
        put_strikes = [k for k, v in put_footprint_data.items() if v != 0]
        if put_strikes:
            fig.add_trace(go.Bar(
                x=put_strikes,
                y=[put_footprint_data[s] for s in put_strikes],
                name='Put OI Changes',
                marker_color='rgba(46, 125, 50, 0.7)',
                hovertemplate='Strike: %{x}<br>Put OI Change: %{y:,.0f}<extra></extra>'
            ))
        
        # Add spot price line
        fig.add_vline(x=spot_price, line_width=2, line_dash="solid", 
                      line_color="blue", annotation_text="Spot")
        
        fig.update_layout(
            title='OI Footprint Analysis',
            xaxis_title='Strike Price',
            yaxis_title='OI Change',
            barmode='group',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        logger.error(f"Error creating OI footprint chart: {e}")
        st.error("Failed to create OI footprint chart")

# --- MAIN APPLICATION UI ---
def main():
    """Main application function with comprehensive error handling"""
    try:
        st.title("🚀 Pro Options & Greeks Analyzer - WebSocket Real-Time Edition")
        
        # Initialize session state
        if 'last_fetch_time' not in st.session_state:
            st.session_state.last_fetch_time = None
        if 'run_analysis' not in st.session_state:
            st.session_state.run_analysis = False
        if 'theme' not in st.session_state:
            st.session_state.theme = 'light'
        if 'real_time_enabled' not in st.session_state:
            st.session_state.real_time_enabled = False
        if 'chain_data_loaded' not in st.session_state:
            st.session_state.chain_data_loaded = False
        
        # Sidebar Configuration
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # API Credentials
            with st.expander("🔐 API Credentials", expanded=True):
                api_key, api_secret = load_credentials()
                if not api_key or not api_secret:
                    st.error("API credentials not found. Please check your configuration.")
                    return
                session_token = st.text_input("Session Token", type="password", 
                                            help="Get from https://api.icicidirect.com/apiuser/login")
            
            # Symbol Selection
            symbol = st.selectbox("📊 Select Symbol", config.SYMBOLS)
            st.session_state.current_symbol = symbol
            
            # Mode Selection
            st.subheader("🔄 Data Mode")
            data_mode = st.radio(
                "Select Data Mode",
                ["Static (Manual Refresh)", "Auto-Refresh", "WebSocket Streaming"],
                help="WebSocket provides real-time tick data without API limits"
            )
            
            # Configure based on mode
            if data_mode == "Auto-Refresh":
                refresh_interval = st.slider("Refresh Interval (seconds)", 10, 300, 60)
                st_autorefresh(interval=refresh_interval * 1000, key="datarefresh")
            elif data_mode == "WebSocket Streaming":
                st.session_state.real_time_enabled = True
                
                # Auto-initialize analyzer
                if 'rt_analyzer' not in st.session_state:
                    try:
                        st.session_state.rt_analyzer = RealTimeOIFlowAnalyzer(config)
                        logger.info("Real-time analyzer initialized")
                    except Exception as e:
                        st.error(f"Failed to initialize analyzer: {e}")
                        logger.error(f"Analyzer initialization error: {e}")
                
                # Auto-trigger data fetch if not loaded
                if not st.session_state.chain_data_loaded:
                    st.session_state.run_analysis = True
                
                st.info("🌐 WebSocket mode - No API rate limits!")
            else:
                st.session_state.real_time_enabled = False
            
            # Real-time Status (WebSocket)
            if st.session_state.real_time_enabled:
                st.subheader("🌐 WebSocket Status")
                
                if 'rt_analyzer' in st.session_state:
                    rt_analyzer = st.session_state.rt_analyzer
                    
                    # Check if we have the required connection info
                    breeze = st.session_state.get('breeze_connection')
                    symbol = st.session_state.get('current_symbol')
                    expiry = st.session_state.get('current_expiry')
                    
                    if rt_analyzer:
                        status = rt_analyzer.get_real_time_status()
                        if status['is_streaming'] and status['ws_connected']:
                            st.success("🟢 WEBSOCKET LIVE")
                            st.metric("Ticks", f"{status['tick_count']:,}")
                            st.metric("Alerts", status['alerts_count'])
                        elif all([breeze, symbol, expiry]):
                            st.warning("⚪ READY TO START")
                            if st.button("🚀 Quick Start", key="sidebar_start", use_container_width=True):
                                with st.spinner("Connecting..."):
                                    success = rt_analyzer.start_real_time_analysis(breeze, symbol, expiry)
                                if success:
                                    st.success("Started!")
                                    st.rerun()
                                else:
                                    st.error("Failed to connect")
                        else:
                            st.info("⚪ WAITING FOR DATA")
                            st.caption("Load options data first")
                    else:
                        st.error("❌ INITIALIZATION FAILED")
                else:
                    st.error("❌ Analyzer not initialized")
            
            # Display Settings
            st.subheader("📈 Display Options")
            show_greeks = st.checkbox("Show Greeks", value=True)
            show_iv_smile = st.checkbox("Show IV Smile", value=True)
            show_volume = st.checkbox("Show Volume Profile", value=True)
            show_strategy = st.checkbox("Show Strategy Analysis", value=False)
            show_real_time_dashboard = st.checkbox("Show Real-Time Dashboard", value=st.session_state.real_time_enabled)
            
            # Risk Parameters
            st.subheader("⚡ Risk Parameters")
            risk_free_rate = st.number_input("Risk-Free Rate (%)", value=7.0, step=0.1) / 100
            
            # Export Options
            st.subheader("💾 Export Data")
            export_format = st.selectbox("Export Format", ["JSON", "CSV", "Excel"])
            
            # Quick Stats
            if 'oi_analysis_results' in st.session_state:
                st.subheader("🔍 Quick Stats")
                results = st.session_state.oi_analysis_results
                st.metric("Footprints", len(results.get('footprints', [])))
                st.metric("Alerts", len(results.get('manipulation_alerts', [])))
                st.metric("Regime", results.get('market_regime', 'Unknown'))
                
                signals = results.get('signals', [])
                if signals:
                    st.write("**Top Signal:**")
                    top_signal = signals[0]
                    st.info(f"{top_signal.get('action', 'N/A')} @ {top_signal.get('strike', 'N/A')}")
        
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
        
        # Store breeze connection for real-time use
        st.session_state.breeze_connection = breeze
        
        # Initialize real-time analyzer if real-time mode is enabled
        if st.session_state.real_time_enabled and 'rt_analyzer' not in st.session_state:
            try:
                st.session_state.rt_analyzer = RealTimeOIFlowAnalyzer(config)
                logger.info("Real-time analyzer initialized in main")
            except Exception as e:
                st.error(f"Failed to initialize real-time analyzer: {e}")
                logger.error(f"RT analyzer init error: {e}")
        
        # Fetch Expiry Dates
        try:
            expiry_map = get_expiry_map(breeze, symbol)
            if not expiry_map:
                st.error("Failed to fetch expiry dates. Please check your connection.")
                return
        except BreezeAPIError as e:
            st.error(str(e))
            return
        except Exception as e:
            st.error(f"Error fetching expiry dates: {e}")
            return
        
        # Expiry Selection and Data Mode Display
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            selected_expiry = st.selectbox("📅 Select Expiry", list(expiry_map.keys()))
            st.session_state.selected_display_date = selected_expiry
            st.session_state.current_expiry = expiry_map[selected_expiry]
        
        with col2:
            # Show data mode and freshness
            if data_mode == "WebSocket Streaming":
                rt_analyzer = st.session_state.get('rt_analyzer')
                if rt_analyzer and rt_analyzer.get_real_time_status()['is_streaming']:
                    st.success("🌐 WEBSOCKET LIVE STREAMING")
                else:
                    st.warning("⚪ WEBSOCKET NOT ACTIVE")
            elif st.session_state.last_fetch_time:
                time_diff = (datetime.now() - st.session_state.last_fetch_time).seconds
                st.info(f"Last updated: {st.session_state.last_fetch_time.strftime('%H:%M:%S')} ({time_diff}s ago)")
        
        with col3:
            if data_mode == "Static (Manual Refresh)":
                if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
                    st.session_state.run_analysis = True
            elif data_mode == "WebSocket Streaming":
                # Add a manual refresh button for WebSocket mode too
                if st.button("📊 Load Chain", type="primary", use_container_width=True):
                    st.session_state.run_analysis = True
                    st.session_state.chain_data_loaded = False
        
        # Real-Time Dashboard (if enabled and before main analysis)
        if show_real_time_dashboard and st.session_state.real_time_enabled:
            create_real_time_dashboard()
            st.markdown("---")
        
        # Fetch and analyze data - Updated condition for WebSocket mode
        if (st.session_state.run_analysis or 
            data_mode in ["Auto-Refresh", "WebSocket Streaming"] or 
            (not st.session_state.chain_data_loaded and data_mode == "WebSocket Streaming")):
            
            try:
                api_expiry_date = expiry_map[selected_expiry]
                raw_data, spot_price = get_options_chain_data_with_retry(breeze, symbol, api_expiry_date)
                
                if raw_data and spot_price:
                    full_chain_df = process_and_analyze(raw_data, spot_price, selected_expiry)
                    
                    if not full_chain_df.empty:
                        # Mark data as loaded
                        st.session_state.chain_data_loaded = True
                        st.session_state.run_analysis = False  # Reset the flag
                        
                        # Calculate metrics
                        metrics = calculate_dashboard_metrics(full_chain_df, spot_price)
                        atm_strike = full_chain_df.iloc[(full_chain_df['Strike'] - spot_price).abs().argsort()[:1]]['Strike'].values[0]
                        
                        # Track historical data
                        track_historical_data_efficient(symbol, selected_expiry, metrics)
                        
                        # Display Key Metrics
                        st.subheader("📊 Key Metrics Dashboard")
                        
                        # Data mode indicator
                        if data_mode == "WebSocket Streaming":
                            rt_analyzer = st.session_state.get('rt_analyzer')
                            if rt_analyzer and rt_analyzer.get_real_time_status()['is_streaming']:
                                st.success("🌐 WEBSOCKET LIVE - Real-time tick data streaming with no API limits")
                            else:
                                st.warning("⚪ Static snapshot - Start WebSocket streaming for live tick updates")
                        elif data_mode == "Auto-Refresh":
                            st.info(f"🔄 AUTO-REFRESH - Updates every {refresh_interval} seconds")
                        else:
                            st.info("📷 STATIC MODE - Manual refresh required")
                        
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
                        
                        # Tabs for different views - Replace the first tab with enhanced OI analysis
                        tabs = ["📊 OI Analysis", "🔥 Heatmap", "😊 IV Analysis", "📈 Volume", 
                                "🧮 Greeks", "📉 Strategy", "⏳ History", "🔍 OI Flow"]
                        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(tabs)
                        
                        with tab1:  # Enhanced OI Analysis tab
                            create_enhanced_oi_analysis_tab(full_chain_df, spot_price, symbol)
                        
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
                        
                        with tab8:  # OI Flow Analysis Tab
                            st.subheader("🔍 Advanced OI Flow Analysis")
                            
                            # Initialize OI analyzer
                            if 'oi_analyzer' not in st.session_state:
                                st.session_state.oi_analyzer = RealTimeOIFlowAnalyzer(config)
                            
                            oi_analyzer = st.session_state.oi_analyzer
                            
                            # Analysis controls
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                analysis_timeframe = st.selectbox(
                                    "Analysis Timeframe",
                                    ["5min", "10min", "30min", "1hour", "2hour", "daily"],
                                    key="oi_flow_timeframe"
                                )
                            
                            with col2:
                                if st.button("🔄 Analyze OI Flow", use_container_width=True):
                                    # Run OI flow analysis
                                    with st.spinner("Analyzing OI flow patterns..."):
                                        oi_analysis_results = oi_analyzer.analyze_oi_flow_patterns(
                                            full_chain_df, spot_price, analysis_timeframe
                                        )
                                        
                                        # Store results in session state
                                        st.session_state.oi_analysis_results = oi_analysis_results
                                        st.session_state.oi_analysis_timestamp = datetime.now()
                            
                            # Display results if available
                            if 'oi_analysis_results' in st.session_state:
                                results = st.session_state.oi_analysis_results
                                
                                # Show analysis timestamp
                                if 'oi_analysis_timestamp' in st.session_state:
                                    time_diff = (datetime.now() - st.session_state.oi_analysis_timestamp).seconds
                                    st.info(f"Analysis performed {time_diff} seconds ago")
                                
                                # Create dashboard
                                create_oi_flow_dashboard(oi_analyzer, results, full_chain_df, spot_price)
                                
                                # Institutional Activity
                                if results.get('institutional_activity'):
                                    st.subheader("🏢 Institutional Activity")
                                    inst_df = pd.DataFrame(results['institutional_activity'])
                                    
                                    # Group by flow type
                                    flow_summary = inst_df.groupby('flow_type').agg({
                                        'size': 'sum',
                                        'premium_involved': 'sum'
                                    }).round(0)
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.dataframe(flow_summary, use_container_width=True)
                                    
                                    with col2:
                                        # Pie chart of flow types
                                        fig = go.Figure(data=[go.Pie(
                                            labels=flow_summary.index,
                                            values=flow_summary['size'],
                                            hole=0.3
                                        )])
                                        fig.update_layout(
                                            title="Institutional Flow Distribution",
                                            height=300
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                
                                # Key Levels
                                if results.get('key_levels'):
                                    st.subheader("🎯 Key OI-Based Levels")
                                    levels = results['key_levels']
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if levels.get('resistance'):
                                            st.error(f"🔴 Resistance: {', '.join(map(str, levels['resistance']))}")
                                    
                                    with col2:
                                        if levels.get('support'):
                                            st.success(f"🟢 Support: {', '.join(map(str, levels['support']))}")
                                
                                # Export OI Analysis
                                if st.button("📥 Export OI Flow Analysis"):
                                    try:
                                        export_data = {
                                            'timestamp': datetime.now().isoformat(),
                                            'symbol': symbol,
                                            'expiry': selected_expiry,
                                            'spot_price': spot_price,
                                            'timeframe': analysis_timeframe,
                                            'analysis_results': {
                                                'footprints': [
                                                    {
                                                        'timestamp': fp.timestamp.isoformat(),
                                                        'strike': fp.strike,
                                                        'option_type': fp.option_type,
                                                        'oi_change': fp.oi_change,
                                                        'volume': fp.volume,
                                                        'large_trade': fp.large_trade_indicator,
                                                        'aggressor': fp.aggressor_side
                                                    } for fp in results['footprints']
                                                ],
                                                'signals': results['signals'],
                                                'alerts': results['manipulation_alerts'],
                                                'institutional_activity': results['institutional_activity'],
                                                'market_regime': results['market_regime']
                                            }
                                        }
                                        
                                        st.download_button(
                                            "Download OI Analysis JSON",
                                            data=json.dumps(export_data, indent=2),
                                            file_name=f"oi_flow_analysis_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                            mime="application/json"
                                        )
                                    except Exception as e:
                                        logger.error(f"Error exporting OI analysis: {e}")
                                        st.error("Failed to export OI analysis")
                        
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
                        try:
                            filtered_df['Moneyness'] = filtered_df.apply(
                                lambda row: 'ITM' if (row['Strike'] < spot_price and row['Put LTP'] > 0) or 
                                                   (row['Strike'] > spot_price and row['Call LTP'] > 0)
                                else 'OTM' if (row['Strike'] > spot_price and row['Put LTP'] > 0) or 
                                             (row['Strike'] < spot_price and row['Call LTP'] > 0)
                                else 'ATM', axis=1
                            )
                        except Exception as e:
                            logger.error(f"Error calculating moneyness: {e}")
                            filtered_df['Moneyness'] = 'Unknown'
                        
                        # Style the dataframe
                        def highlight_moneyness(row):
                            if row['Moneyness'] == 'ITM':
                                return ['background-color: #e8f5e9'] * len(row)
                            elif row['Moneyness'] == 'ATM':
                                return ['background-color: #fff3e0'] * len(row)
                            else:
                                return [''] * len(row)
                        
                        try:
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
                        except Exception as e:
                            logger.error(f"Error styling dataframe: {e}")
                            st.dataframe(filtered_df[display_cols + ['Moneyness']], use_container_width=True, height=600)
                        
                        # Summary statistics
                        with st.expander("📊 Summary Statistics"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Call Options Summary:**")
                                st.write(f"- Total OI: {filtered_df['Call OI'].sum():,.0f}")
                                st.write(f"- Total Volume: {filtered_df['Call Volume'].sum():,.0f}")
                                if 'Call IV' in filtered_df.columns:
                                    st.write(f"- Avg IV: {filtered_df['Call IV'].mean():.1f}%")
                                if not filtered_df.empty:
                                    st.write(f"- Max OI Strike: {filtered_df.loc[filtered_df['Call OI'].idxmax(), 'Strike']:,.0f}")
                            
                            with col2:
                                st.write("**Put Options Summary:**")
                                st.write(f"- Total OI: {filtered_df['Put OI'].sum():,.0f}")
                                st.write(f"- Total Volume: {filtered_df['Put Volume'].sum():,.0f}")
                                if 'Put IV' in filtered_df.columns:
                                    st.write(f"- Avg IV: {filtered_df['Put IV'].mean():.1f}%")
                                if not filtered_df.empty:
                                    st.write(f"- Max OI Strike: {filtered_df.loc[filtered_df['Put OI'].idxmax(), 'Strike']:,.0f}")
                        
                        # Export functionality
                        if st.sidebar.button("📥 Export Data", use_container_width=True):
                            export_df = prepare_export_data(full_chain_df, export_format)
                            if export_df is not None:
                                try:
                                    export_data_dict = {
                                        'metadata': {
                                            'symbol': symbol,
                                            'expiry': selected_expiry,
                                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                            'spot_price': spot_price,
                                            'metrics': metrics,
                                            'data_mode': data_mode
                                        },
                                        'chain_data': export_df.to_dict('records')
                                    }
                                    
                                    # Add real-time data if available
                                    if st.session_state.real_time_enabled and 'rt_analyzer' in st.session_state:
                                        rt_analyzer = st.session_state.rt_analyzer
                                        if rt_analyzer.streamer:
                                            rt_data = rt_analyzer.get_real_time_data(300)  # Last 5 minutes
                                            export_data_dict['real_time_data'] = {
                                                'oi_changes': rt_data['oi_changes'],
                                                'price_changes': rt_data['price_changes'],
                                                'alerts': [
                                                    {
                                                        'timestamp': alert.timestamp.isoformat(),
                                                        'type': alert.alert_type,
                                                        'strike': alert.strike,
                                                        'option_type': alert.option_type,
                                                        'message': alert.message,
                                                        'severity': alert.severity
                                                    } for alert in rt_data['alerts']
                                                ]
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
                                            
                                            # Add OI analysis results if available
                                            if 'oi_analysis_results' in st.session_state:
                                                results = st.session_state.oi_analysis_results
                                                if results.get('institutional_activity'):
                                                    pd.DataFrame(results['institutional_activity']).to_excel(
                                                        writer, sheet_name='Institutional Flow', index=False
                                                    )
                                                if results.get('signals'):
                                                    pd.DataFrame(results['signals']).to_excel(
                                                        writer, sheet_name='Trading Signals', index=False
                                                    )
                                        
                                        excel_data = output.getvalue()
                                        st.download_button(
                                            label="Download Excel",
                                            data=excel_data,
                                            file_name=f"{symbol}_options_chain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                        )
                                except Exception as e:
                                    logger.error(f"Error exporting data: {e}")
                                    st.error(f"Failed to export data: {e}")
                    
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
            st.info("👆 Select data mode and refresh to load the options chain")
            
            # Add informative content for new users
            with st.expander("📚 Quick Start Guide", expanded=True):
                st.markdown("""
                ### Getting Started:
                
                1. **Enter Session Token** - Get it from the ICICI Direct API login page
                2. **Select Symbol** - Choose from NIFTY, BANKNIFTY, etc.
                3. **Choose Data Mode**:
                   - **Static**: Manual refresh only
                   - **Auto-Refresh**: Updates at set intervals
                   - **WebSocket Streaming**: Real-time tick data with no API limits! 🚀
                4. **Click Load Chain** to fetch initial data
                
                ### Key Features:
                
                - **Real-time WebSocket streaming** - Professional-grade live data
                - **Enhanced OI Analysis** - Time-based OI tracking with visual indicators
                - **Greeks Calculation** - Full options analytics
                - **Manipulation Detection** - AI-powered alerts
                - **Historical Tracking** - Monitor trends over time
                
                ### WebSocket Advantages:
                
                - ✅ No API rate limits
                - ✅ True real-time tick data
                - ✅ Lower latency
                - ✅ Professional trading quality
                """)
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center'>
                <p>Built with ❤️ using Streamlit | Data from ICICI Direct Breeze API</p>
                <p style='font-size: 0.8em; color: gray;'>
                    🌐 <strong>WebSocket Edition</strong> - Professional real-time streaming with enhanced OI analysis!<br>
                    Disclaimer: This tool is for educational purposes only. 
                    Please do your own research before making any trading decisions.
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    except Exception as e:
        logger.error(f"Fatal error in main function: {e}", exc_info=True)
        st.error(f"A fatal error occurred: {e}")
        st.info("Please refresh the page and try again. Check the logs for more details.")

# Run the application
if __name__ == "__main__":
    main()
