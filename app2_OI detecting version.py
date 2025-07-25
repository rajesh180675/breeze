import streamlit as st
import pandas as pd
from breeze_connect import BreezeConnect
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
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
from typing import Dict, List, Tuple, Optional, Any, Deque
from collections import deque
import threading
from queue import Queue
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
    page_title="Pro Options Analyzer - Real-Time", 
    page_icon="🚀", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM EXCEPTIONS ---
class BreezeAPIError(Exception):
    """Custom exception for Breeze API errors"""
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

# --- REAL-TIME DATA STREAMER ---
class RealTimeDataStreamer:
    """Real-time data streaming for options analysis"""
    
    def __init__(self, breeze_connection, symbol: str, expiry_date: str):
        self.breeze = breeze_connection
        self.symbol = symbol
        self.expiry_date = expiry_date
        self.is_streaming = False
        self.streaming_thread = None
        
        # Data storage
        self.data_queue = Queue(maxsize=100)
        self.last_data = None
        self.tick_count = 0
        
        # Real-time buffers
        self.oi_changes_buffer: Deque[Dict] = deque(maxlen=config.REALTIME_BUFFER_SIZE)
        self.price_changes_buffer: Deque[Dict] = deque(maxlen=config.REALTIME_BUFFER_SIZE)
        self.alerts_buffer: Deque[RealTimeAlert] = deque(maxlen=config.REALTIME_ALERT_BUFFER_SIZE)
        
        # Thresholds
        self.oi_change_threshold = 1000  # Minimum OI change to trigger alert
        self.price_change_threshold = 0.05  # 5% price change threshold
        self.rapid_change_window = 30  # 30 seconds for rapid change detection
        
    def start_streaming(self):
        """Start real-time data streaming"""
        try:
            if not self.is_streaming:
                self.is_streaming = True
                self.streaming_thread = threading.Thread(target=self._stream_data, daemon=True)
                self.streaming_thread.start()
                logger.info(f"Real-time streaming started for {self.symbol}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            self.is_streaming = False
            return False
    
    def stop_streaming(self):
        """Stop real-time data streaming"""
        try:
            if self.is_streaming:
                self.is_streaming = False
                if self.streaming_thread:
                    self.streaming_thread.join(timeout=5)
                logger.info(f"Real-time streaming stopped for {self.symbol}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to stop streaming: {e}")
            return False
    
    def _stream_data(self):
        """Main streaming loop"""
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.is_streaming:
            try:
                # Fetch current snapshot
                current_data = self._fetch_current_snapshot()
                
                if current_data:
                    # Reset error counter on successful fetch
                    consecutive_errors = 0
                    
                    # Detect changes if we have previous data
                    if self.last_data:
                        changes = self._detect_changes(current_data)
                        if changes:
                            # Add timestamp and tick number
                            changes['timestamp'] = datetime.now()
                            changes['tick_number'] = self.tick_count
                            
                            # Store in queue for UI processing
                            if not self.data_queue.full():
                                self.data_queue.put(changes)
                            
                            # Update buffers and generate alerts
                            self._update_buffers(changes)
                            self._generate_real_time_alerts(changes)
                            
                            self.tick_count += 1
                    
                    # Update last data
                    self.last_data = current_data
                else:
                    consecutive_errors += 1
                    logger.warning(f"Failed to fetch data, consecutive errors: {consecutive_errors}")
                
                # Break if too many consecutive errors
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"Too many consecutive errors ({consecutive_errors}), stopping stream")
                    self.is_streaming = False
                    break
                
                # Sleep between fetches
                time.sleep(config.REALTIME_FETCH_INTERVAL)
                
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Streaming error: {e}")
                if consecutive_errors >= max_consecutive_errors:
                    self.is_streaming = False
                    break
                time.sleep(5)  # Wait longer on error
    
    def _fetch_current_snapshot(self) -> Optional[Dict]:
        """Fetch current options data snapshot"""
        try:
            # Fetch call options
            call_data = self.breeze.get_option_chain_quotes(
                stock_code=self.symbol,
                exchange_code="NFO",
                product_type="options",
                right="Call",
                expiry_date=self.expiry_date
            )
            
            # Fetch put options
            put_data = self.breeze.get_option_chain_quotes(
                stock_code=self.symbol,
                exchange_code="NFO",
                product_type="options",
                right="Put",
                expiry_date=self.expiry_date
            )
            
            # Fetch spot price
            spot_data = self.breeze.get_quotes(
                stock_code=self.symbol,
                exchange_code="NSE",
                product_type="cash"
            )
            
            # Check if all requests succeeded
            if (call_data.get('Success') and put_data.get('Success') and 
                spot_data.get('Success')):
                
                return {
                    'calls': call_data['Success'],
                    'puts': put_data['Success'],
                    'spot': float(spot_data['Success'][0]['ltp']),
                    'timestamp': datetime.now()
                }
            
        except Exception as e:
            logger.error(f"Error fetching snapshot: {e}")
        
        return None
    
    def _detect_changes(self, current_data: Dict) -> Optional[Dict]:
        """Detect changes from previous snapshot"""
        try:
            if not self.last_data:
                return None
            
            changes = {
                'oi_changes': [],
                'price_changes': [],
                'volume_changes': [],
                'spot_change': current_data['spot'] - self.last_data['spot'],
                'spot_change_pct': ((current_data['spot'] - self.last_data['spot']) / self.last_data['spot']) * 100
            }
            
            # Process calls
            for current_call in current_data['calls']:
                prev_call = self._find_previous_option(self.last_data['calls'], current_call['strike_price'])
                
                if prev_call:
                    # OI changes
                    oi_change = current_call['open_interest'] - prev_call['open_interest']
                    if abs(oi_change) > 0:
                        changes['oi_changes'].append({
                            'strike': current_call['strike_price'],
                            'type': 'CALL',
                            'oi_change': oi_change,
                            'current_oi': current_call['open_interest'],
                            'prev_oi': prev_call['open_interest'],
                            'ltp': current_call['ltp'],
                            'volume': current_call.get('volume', 0)
                        })
                    
                    # Price changes
                    price_change = current_call['ltp'] - prev_call['ltp']
                    price_change_pct = (price_change / prev_call['ltp']) * 100 if prev_call['ltp'] > 0 else 0
                    
                    if abs(price_change_pct) > self.price_change_threshold * 100:
                        changes['price_changes'].append({
                            'strike': current_call['strike_price'],
                            'type': 'CALL',
                            'price_change': price_change,
                            'price_change_pct': price_change_pct,
                            'current_ltp': current_call['ltp'],
                            'prev_ltp': prev_call['ltp']
                        })
            
            # Process puts (similar logic)
            for current_put in current_data['puts']:
                prev_put = self._find_previous_option(self.last_data['puts'], current_put['strike_price'])
                
                if prev_put:
                    # OI changes
                    oi_change = current_put['open_interest'] - prev_put['open_interest']
                    if abs(oi_change) > 0:
                        changes['oi_changes'].append({
                            'strike': current_put['strike_price'],
                            'type': 'PUT',
                            'oi_change': oi_change,
                            'current_oi': current_put['open_interest'],
                            'prev_oi': prev_put['open_interest'],
                            'ltp': current_put['ltp'],
                            'volume': current_put.get('volume', 0)
                        })
                    
                    # Price changes
                    price_change = current_put['ltp'] - prev_put['ltp']
                    price_change_pct = (price_change / prev_put['ltp']) * 100 if prev_put['ltp'] > 0 else 0
                    
                    if abs(price_change_pct) > self.price_change_threshold * 100:
                        changes['price_changes'].append({
                            'strike': current_put['strike_price'],
                            'type': 'PUT',
                            'price_change': price_change,
                            'price_change_pct': price_change_pct,
                            'current_ltp': current_put['ltp'],
                            'prev_ltp': prev_put['ltp']
                        })
            
            # Return changes only if significant
            if (changes['oi_changes'] or changes['price_changes'] or 
                abs(changes['spot_change']) > 1):
                return changes
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting changes: {e}")
            return None
    
    def _find_previous_option(self, prev_options: List[Dict], strike: float) -> Optional[Dict]:
        """Find previous option data for given strike"""
        try:
            for option in prev_options:
                if option['strike_price'] == strike:
                    return option
            return None
        except Exception as e:
            logger.error(f"Error finding previous option: {e}")
            return None
    
    def _update_buffers(self, changes: Dict):
        """Update real-time data buffers"""
        try:
            timestamp = changes['timestamp']
            
            # Update OI changes buffer
            for oi_change in changes['oi_changes']:
                self.oi_changes_buffer.append({
                    'timestamp': timestamp,
                    'strike': oi_change['strike'],
                    'type': oi_change['type'],
                    'oi_change': oi_change['oi_change'],
                    'volume': oi_change['volume'],
                    'ltp': oi_change['ltp']
                })
            
            # Update price changes buffer
            for price_change in changes['price_changes']:
                self.price_changes_buffer.append({
                    'timestamp': timestamp,
                    'strike': price_change['strike'],
                    'type': price_change['type'],
                    'price_change_pct': price_change['price_change_pct'],
                    'current_ltp': price_change['current_ltp']
                })
        except Exception as e:
            logger.error(f"Error updating buffers: {e}")
    
    def _generate_real_time_alerts(self, changes: Dict):
        """Generate real-time alerts based on changes"""
        try:
            timestamp = changes['timestamp']
            
            # Large OI change alerts
            for oi_change in changes['oi_changes']:
                if abs(oi_change['oi_change']) >= self.oi_change_threshold:
                    severity = 'HIGH' if abs(oi_change['oi_change']) >= 2000 else 'MEDIUM'
                    
                    alert = RealTimeAlert(
                        timestamp=timestamp,
                        alert_type='LARGE_OI_CHANGE',
                        strike=oi_change['strike'],
                        option_type=oi_change['type'],
                        message=f"Large {oi_change['type']} OI change: {oi_change['oi_change']:+,} at strike {oi_change['strike']}",
                        severity=severity,
                        data=oi_change
                    )
                    self.alerts_buffer.append(alert)
            
            # Unusual price movement alerts
            for price_change in changes['price_changes']:
                if abs(price_change['price_change_pct']) >= 10:  # 10% or more
                    severity = 'HIGH' if abs(price_change['price_change_pct']) >= 20 else 'MEDIUM'
                    
                    alert = RealTimeAlert(
                        timestamp=timestamp,
                        alert_type='UNUSUAL_PRICE_MOVE',
                        strike=price_change['strike'],
                        option_type=price_change['type'],
                        message=f"Unusual {price_change['type']} price move: {price_change['price_change_pct']:+.1f}% at strike {price_change['strike']}",
                        severity=severity,
                        data=price_change
                    )
                    self.alerts_buffer.append(alert)
            
            # Rapid accumulation detection
            rapid_changes = self._detect_rapid_accumulation()
            for rapid_change in rapid_changes:
                alert = RealTimeAlert(
                    timestamp=timestamp,
                    alert_type='RAPID_ACCUMULATION',
                    strike=rapid_change['strike'],
                    option_type=rapid_change['type'],
                    message=f"Rapid {rapid_change['type']} accumulation: {rapid_change['total_change']:+,} in {rapid_change['timeframe']}s at strike {rapid_change['strike']}",
                    severity='HIGH',
                    data=rapid_change
                )
                self.alerts_buffer.append(alert)
        except Exception as e:
            logger.error(f"Error generating alerts: {e}")
    
    def _detect_rapid_accumulation(self) -> List[Dict]:
        """Detect rapid OI accumulation patterns"""
        try:
            cutoff_time = datetime.now() - timedelta(seconds=self.rapid_change_window)
            
            # Group recent OI changes by strike and type
            strike_changes = {}
            for change in self.oi_changes_buffer:
                if change['timestamp'] > cutoff_time:
                    key = f"{change['strike']}_{change['type']}"
                    if key not in strike_changes:
                        strike_changes[key] = []
                    strike_changes[key].append(change)
            
            rapid_changes = []
            for key, changes in strike_changes.items():
                if len(changes) >= 3:  # At least 3 changes in the window
                    total_change = sum(c['oi_change'] for c in changes)
                    if abs(total_change) >= 1500:  # Significant total change
                        strike, option_type = key.split('_')
                        rapid_changes.append({
                            'strike': float(strike),
                            'type': option_type,
                            'total_change': total_change,
                            'change_count': len(changes),
                            'timeframe': self.rapid_change_window
                        })
            
            return rapid_changes
        except Exception as e:
            logger.error(f"Error detecting rapid accumulation: {e}")
            return []
    
    def get_recent_changes(self, seconds: int = 60) -> Dict[str, List]:
        """Get recent changes within specified time window"""
        try:
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

# --- ENHANCED OI FLOW ANALYZER WITH REAL-TIME ---
class RealTimeOIFlowAnalyzer:
    """Enhanced OI Flow Analyzer with real-time capabilities"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.thresholds = config.OI_FLOW_THRESHOLDS
        self.timeframes = config.OI_FLOW_TIMEFRAMES
        self.footprint_buffer: Deque[OIFootprint] = deque(maxlen=10000)
        self.alert_history: List[Dict] = []
        
        # Real-time components
        self.streamer: Optional[RealTimeDataStreamer] = None
        self.is_real_time_enabled = False
    
    def start_real_time_analysis(self, breeze, symbol: str, expiry_date: str) -> bool:
        """Start real-time analysis"""
        try:
            if self.streamer:
                self.streamer.stop_streaming()
            
            self.streamer = RealTimeDataStreamer(breeze, symbol, expiry_date)
            success = self.streamer.start_streaming()
            
            if success:
                self.is_real_time_enabled = True
                logger.info(f"Real-time analysis started for {symbol}")
            
            return success
        except Exception as e:
            logger.error(f"Failed to start real-time analysis: {e}")
            return False
    
    def stop_real_time_analysis(self) -> bool:
        """Stop real-time analysis"""
        try:
            if self.streamer:
                success = self.streamer.stop_streaming()
                if success:
                    self.is_real_time_enabled = False
                    logger.info("Real-time analysis stopped")
                return success
            return True
        except Exception as e:
            logger.error(f"Failed to stop real-time analysis: {e}")
            return False
    
    def get_real_time_status(self) -> Dict[str, Any]:
        """Get current real-time status"""
        try:
            if not self.streamer:
                return {
                    'is_streaming': False,
                    'tick_count': 0,
                    'alerts_count': 0,
                    'buffer_size': 0
                }
            
            return {
                'is_streaming': self.streamer.is_streaming,
                'tick_count': self.streamer.tick_count,
                'alerts_count': len(self.streamer.alerts_buffer),
                'buffer_size': len(self.streamer.oi_changes_buffer)
            }
        except Exception as e:
            logger.error(f"Error getting real-time status: {e}")
            return {
                'is_streaming': False,
                'tick_count': 0,
                'alerts_count': 0,
                'buffer_size': 0
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
    """Process raw options data and calculate Greeks"""
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
    except Exception as e:
        logger.error(f"Error processing and analyzing data: {e}")
        st.error(f"Error processing data: {e}")
        return pd.DataFrame()

def calculate_dashboard_metrics(chain_df: pd.DataFrame, spot_price: float) -> Dict[str, Any]:
    """Calculate key metrics from options chain"""
    try:
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

def create_oi_chart(chain_df: pd.DataFrame, atm_strike: float, spot_price: float, 
                   max_pain: Optional[float] = None) -> go.Figure:
    """Create Open Interest distribution chart"""
    try:
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
    except Exception as e:
        logger.error(f"Error creating OI chart: {e}")
        return go.Figure()

def create_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create premium heatmap"""
    try:
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
    except Exception as e:
        logger.error(f"Error creating heatmap: {e}")
        return go.Figure()

def create_iv_smile_chart(chain_df: pd.DataFrame, spot_price: float) -> Optional[go.Figure]:
    """Create IV smile chart"""
    try:
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
    except Exception as e:
        logger.error(f"Error creating IV smile chart: {e}")
        return None

def create_volume_profile(chain_df: pd.DataFrame) -> go.Figure:
    """Create volume profile chart"""
    try:
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
    except Exception as e:
        logger.error(f"Error creating volume profile: {e}")
        return go.Figure()

def display_sentiment_gauge(sentiment_score: float) -> go.Figure:
    """Create sentiment gauge chart"""
    try:
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

def create_greeks_surface(chain_df: pd.DataFrame, greek: str, option_type: str) -> go.Figure:
    """Create 3D surface plot for Greeks"""
    try:
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
    except Exception as e:
        logger.error(f"Error creating Greeks surface: {e}")
        return None

def track_historical_data_efficient(symbol: str, expiry: str, metrics: Dict[str, Any]) -> None:
    """Efficient historical data tracking with compression"""
    try:
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
    except Exception as e:
        logger.error(f"Error tracking historical data: {e}")

def prepare_export_data(df: pd.DataFrame, format_type: str) -> Optional[pd.DataFrame]:
    """Prepare and validate data for export"""
    try:
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
    except Exception as e:
        logger.error(f"Error preparing export data: {e}")
        return None

def create_strategy_payoff(chain_df: pd.DataFrame, spot_price: float) -> go.Figure:
    """Create strategy payoff diagram"""
    try:
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
    except Exception as e:
        logger.error(f"Error creating strategy payoff: {e}")
        return go.Figure()

# --- REAL-TIME DASHBOARD FUNCTIONS ---
def create_real_time_dashboard() -> None:
    """Create real-time monitoring dashboard"""
    st.subheader("🔴 LIVE - Real-Time OI Flow Monitor")
    
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
        'is_streaming': False, 'tick_count': 0, 'alerts_count': 0, 'buffer_size': 0
    }
    
    with col1:
        if status['is_streaming']:
            st.success("🟢 LIVE STREAMING")
            st.metric("Ticks Processed", status['tick_count'])
        elif data_ready:
            st.warning("⚪ READY TO START")
            st.metric("Status", "Ready")
        else:
            st.info("⚪ WAITING FOR DATA")
            st.metric("Status", "Waiting")
    
    with col2:
        st.metric("Live Alerts", status['alerts_count'])
        if data_ready:
            st.success("✅ Data Ready")
        else:
            st.error("❌ Data Missing")
    
    with col3:
        current_time = datetime.now().strftime("%H:%M:%S")
        st.metric("Current Time", current_time)
    
    with col4:
        st.metric("Buffer Size", status['buffer_size'])
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if data_ready:
            if not status['is_streaming']:
                if st.button("🟢 Start Real-Time", type="primary", use_container_width=True):
                    try:
                        success = rt_analyzer.start_real_time_analysis(breeze, symbol, expiry)
                        if success:
                            st.success("Real-time analysis started!")
                            st.rerun()
                        else:
                            st.error("Failed to start real-time analysis")
                    except Exception as e:
                        st.error(f"Error starting real-time: {e}")
                        logger.error(f"Real-time start error: {e}")
            else:
                st.success("🟢 STREAMING ACTIVE")
        else:
            st.button("🟢 Start Real-Time", disabled=True, use_container_width=True)
            st.caption("Load options data first")
    
    with col2:
        if st.button("🔴 Stop Real-Time", use_container_width=True):
            if rt_analyzer:
                try:
                    success = rt_analyzer.stop_real_time_analysis()
                    if success:
                        st.info("Real-time analysis stopped")
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
    
    # Debug information (optional - remove in production)
    with st.expander("🔧 Debug Info", expanded=False):
        st.write("**Session State:**")
        st.write(f"- RT Enabled: {st.session_state.get('real_time_enabled', False)}")
        st.write(f"- RT Analyzer: {'✅' if 'rt_analyzer' in st.session_state else '❌'}")
        st.write(f"- Breeze Connection: {'✅' if breeze else '❌'}")
        st.write(f"- Symbol: {symbol if symbol else '❌'}")
        st.write(f"- Expiry: {expiry if expiry else '❌'}")
        
        if rt_analyzer:
            st.write(f"- Analyzer Status: {status}")
        else:
            st.write("- Analyzer Status: NOT AVAILABLE")
    
    # Real-time alerts display
    if rt_analyzer and status['is_streaming']:
        # Get real-time data
        rt_data = rt_analyzer.get_real_time_data(60)  # Last 60 seconds
        
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
            st.subheader("📊 Live OI Changes (Last 60 seconds)")
            
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
                    fig.add_trace(go.Bar(
                        x=list(call_changes.keys()),
                        y=list(call_changes.values()),
                        name='Call OI Changes',
                        marker_color='red',
                        opacity=0.7
                    ))
                
                if put_changes:
                    fig.add_trace(go.Bar(
                        x=list(put_changes.keys()),
                        y=list(put_changes.values()),
                        name='Put OI Changes',
                        marker_color='green',
                        opacity=0.7
                    ))
                
                fig.update_layout(
                    title="Real-Time OI Changes (Last 60 seconds)",
                    xaxis_title="Strike Price",
                    yaxis_title="OI Change",
                    height=400,
                    barmode='group'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Real-time statistics
        col1, col2 = st.columns(2)
        
        with col1:
            if rt_data['oi_changes']:
                total_call_changes = sum(c['oi_change'] for c in rt_data['oi_changes'] if c['type'] == 'CALL')
                total_put_changes = sum(c['oi_change'] for c in rt_data['oi_changes'] if c['type'] == 'PUT')
                
                st.metric("Total Call OI Change", f"{total_call_changes:+,}")
                st.metric("Total Put OI Change", f"{total_put_changes:+,}")
        
        with col2:
            if rt_data['price_changes']:
                avg_call_change = np.mean([c['price_change_pct'] for c in rt_data['price_changes'] if c['type'] == 'CALL'] or [0])
                avg_put_change = np.mean([c['price_change_pct'] for c in rt_data['price_changes'] if c['type'] == 'PUT'] or [0])
                
                st.metric("Avg Call Price Change", f"{avg_call_change:+.1f}%")
                st.metric("Avg Put Price Change", f"{avg_put_change:+.1f}%")
    
    else:
        st.info("Start real-time monitoring to see live data")

# --- OI FLOW INTEGRATION FUNCTIONS ---
def create_oi_flow_dashboard(analyzer: RealTimeOIFlowAnalyzer, 
                           analysis_results: Dict[str, Any],
                           chain_df: pd.DataFrame,
                           spot_price: float) -> None:
    """Create OI flow dashboard integrated with existing UI"""
    
    # Header metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_footprints = len(analysis_results['footprints'])
        st.metric("OI Footprints", total_footprints)
    
    with col2:
        inst_flows = len(analysis_results['institutional_activity'])
        st.metric("Institutional Flows", inst_flows)
    
    with col3:
        active_signals = len([s for s in analysis_results['signals'] if s['strength'] > 0.7])
        st.metric("Active Signals", active_signals)
    
    with col4:
        regime = analysis_results['market_regime']
        regime_color = "🟢" if "BULLISH" in regime else "🔴" if "BEARISH" in regime else "⚪"
        st.metric("Market Regime", f"{regime_color} {regime}")
    
    # Alerts
    if analysis_results['manipulation_alerts']:
        st.error("⚠️ Manipulation Alerts Detected!")
        for alert in analysis_results['manipulation_alerts']:
            st.warning(f"{alert['type']}: Strike {alert['strike']} - {alert['recommendation']}")
    
    # Create visualizations
    _create_oi_footprint_chart(analysis_results['footprints'], chain_df, spot_price)
    
    # Signals table
    if analysis_results['signals']:
        st.subheader("📊 Trading Signals")
        signals_df = pd.DataFrame(analysis_results['signals'])
        st.dataframe(
            signals_df.style.background_gradient(subset=['strength'], cmap='RdYlGn'),
            use_container_width=True
        )

def _create_oi_footprint_chart(footprints: List[OIFootprint], 
                              chain_df: pd.DataFrame,
                              spot_price: float) -> None:
    """Create OI footprint visualization"""
    if not footprints:
        st.info("No significant OI footprints detected")
        return
    
    try:
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
        fig.add_trace(go.Bar(
            x=list(call_footprint_data.keys()),
            y=list(call_footprint_data.values()),
            name='Call OI Changes',
            marker_color='rgba(239, 83, 80, 0.7)',
            hovertemplate='Strike: %{x}<br>Call OI Change: %{y:,.0f}<extra></extra>'
        ))
        
        # Add put footprints
        fig.add_trace(go.Bar(
            x=list(put_footprint_data.keys()),
            y=list(put_footprint_data.values()),
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
    st.title("🚀 Pro Options & Greeks Analyzer - Real-Time Edition")
    
    # Initialize session state
    if 'last_fetch_time' not in st.session_state:
        st.session_state.last_fetch_time = None
    if 'run_analysis' not in st.session_state:
        st.session_state.run_analysis = False
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'
    if 'real_time_enabled' not in st.session_state:
        st.session_state.real_time_enabled = False
    
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
            ["Static (Manual Refresh)", "Auto-Refresh", "Real-Time Streaming"],
            help="Choose how you want to receive data"
        )
        
        # Configure based on mode
        if data_mode == "Auto-Refresh":
            refresh_interval = st.slider("Refresh Interval (seconds)", 10, 300, 60)
            st_autorefresh(interval=refresh_interval * 1000, key="datarefresh")
        elif data_mode == "Real-Time Streaming":
            st.session_state.real_time_enabled = True
            
            # Auto-initialize analyzer
            if 'rt_analyzer' not in st.session_state:
                try:
                    st.session_state.rt_analyzer = RealTimeOIFlowAnalyzer(config)
                    logger.info("Real-time analyzer initialized")
                except Exception as e:
                    st.error(f"Failed to initialize analyzer: {e}")
                    logger.error(f"Analyzer initialization error: {e}")
            
            rt_fetch_interval = st.slider("Fetch Interval (seconds)", 1, 10, 2)
            config.REALTIME_FETCH_INTERVAL = rt_fetch_interval
        else:
            st.session_state.real_time_enabled = False
        
        # Real-time Status
        if st.session_state.real_time_enabled:
            st.subheader("🔴 Real-Time Status")
            
            # Initialize analyzer if not exists and we have the required data
            if 'rt_analyzer' not in st.session_state:
                st.session_state.rt_analyzer = RealTimeOIFlowAnalyzer(config)
            
            rt_analyzer = st.session_state.rt_analyzer
            
            # Check if we have the required connection info
            breeze = st.session_state.get('breeze_connection')
            symbol = st.session_state.get('current_symbol')
            expiry = st.session_state.get('current_expiry')
            
            if rt_analyzer:
                status = rt_analyzer.get_real_time_status()
                if status['is_streaming']:
                    st.success("🟢 STREAMING LIVE")
                    st.metric("Ticks", status['tick_count'])
                    st.metric("Alerts", status['alerts_count'])
                elif all([breeze, symbol, expiry]):
                    st.warning("⚪ READY TO START")
                    if st.button("🚀 Quick Start", key="sidebar_start", use_container_width=True):
                        success = rt_analyzer.start_real_time_analysis(breeze, symbol, expiry)
                        if success:
                            st.success("Started!")
                            st.rerun()
                        else:
                            st.error("Failed to start")
                else:
                    st.info("⚪ WAITING FOR DATA")
                    st.caption("Load options data first")
            else:
                st.error("❌ INITIALIZATION FAILED")
        
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
            st.metric("Footprints", len(results['footprints']))
            st.metric("Alerts", len(results['manipulation_alerts']))
            st.metric("Regime", results['market_regime'])
            
            if results['signals']:
                st.write("**Top Signal:**")
                top_signal = results['signals'][0]
                st.info(f"{top_signal['action']} @ {top_signal.get('strike', 'N/A')}")
    
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
    
    # Expiry Selection
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        selected_expiry = st.selectbox("📅 Select Expiry", list(expiry_map.keys()))
        st.session_state.selected_display_date = selected_expiry
        st.session_state.current_expiry = expiry_map[selected_expiry]
    
    with col2:
        # Show data mode and freshness
        if data_mode == "Real-Time Streaming":
            rt_analyzer = st.session_state.get('rt_analyzer')
            if rt_analyzer and rt_analyzer.get_real_time_status()['is_streaming']:
                st.success("🔴 LIVE DATA STREAMING")
            else:
                st.warning("⚪ REAL-TIME NOT ACTIVE")
        elif st.session_state.last_fetch_time:
            time_diff = (datetime.now() - st.session_state.last_fetch_time).seconds
            st.info(f"Last updated: {st.session_state.last_fetch_time.strftime('%H:%M:%S')} ({time_diff}s ago)")
    
    with col3:
        if data_mode == "Static (Manual Refresh)":
            if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
                st.session_state.run_analysis = True
    
    # Real-Time Dashboard (if enabled and before main analysis)
    if show_real_time_dashboard and st.session_state.real_time_enabled:
        create_real_time_dashboard()
        st.markdown("---")
    
    # Fetch and analyze data
    if st.session_state.run_analysis or data_mode == "Auto-Refresh" or data_mode == "Static (Manual Refresh)":
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
                    
                    # Data mode indicator
                    if data_mode == "Real-Time Streaming":
                        rt_analyzer = st.session_state.get('rt_analyzer')
                        if rt_analyzer and rt_analyzer.get_real_time_status()['is_streaming']:
                            st.success("🔴 LIVE DATA - Real-time streaming active with tick-by-tick updates")
                        else:
                            st.warning("⚪ Static snapshot - Start real-time streaming for live updates")
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
                    
                    # Tabs for different views
                    tabs = ["📊 OI Analysis", "🔥 Heatmap", "😊 IV Analysis", "📈 Volume", 
                            "🧮 Greeks", "📉 Strategy", "⏳ History", "🔍 OI Flow"]
                    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(tabs)
                    
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
                            if results['institutional_activity']:
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
                            if results['key_levels']:
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
                                            if results['institutional_activity']:
                                                pd.DataFrame(results['institutional_activity']).to_excel(
                                                    writer, sheet_name='Institutional Flow', index=False
                                                )
                                            if results['signals']:
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
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built with ❤️ using Streamlit | Data from ICICI Direct Breeze API</p>
            <p style='font-size: 0.8em; color: gray;'>
                🔴 <strong>Real-Time Edition</strong> - Now with tick-by-tick streaming capabilities!<br>
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
