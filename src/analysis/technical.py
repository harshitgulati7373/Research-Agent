"""
Technical Analysis Module

This module provides comprehensive technical analysis capabilities including:
- Moving averages and trend analysis
- Momentum indicators (RSI, MACD, Stochastic)
- Volume analysis
- Support and resistance levels
- Chart patterns
- Technical signals generation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    
from src.data_sources.base import StockData


logger = logging.getLogger(__name__)


@dataclass
class TechnicalSignal:
    """Signal from technical analysis."""
    signal: str  # 'BUY', 'SELL', 'HOLD'
    strength: float  # 0.0 to 1.0
    reason: str
    indicators: Dict[str, float]


@dataclass
class TechnicalAnalysis:
    """Complete technical analysis result."""
    symbol: str
    analysis_date: datetime
    overall_signal: TechnicalSignal
    trend_analysis: Dict[str, Any]
    momentum_analysis: Dict[str, Any]
    volume_analysis: Dict[str, Any]
    support_resistance: Dict[str, Any]
    pattern_analysis: Dict[str, Any]
    score: float  # Overall score 0-100


class TechnicalAnalyzer:
    """Comprehensive technical analysis engine."""
    
    def __init__(self):
        self.trend_weights = {
            'sma_trend': 0.30,
            'ema_trend': 0.25,
            'macd_trend': 0.25,
            'adx_trend': 0.20
        }
        
        self.momentum_weights = {
            'rsi': 0.35,
            'macd': 0.30,
            'stochastic': 0.20,
            'williams_r': 0.15
        }
        
        self.volume_weights = {
            'volume_trend': 0.40,
            'obv': 0.30,
            'volume_price_trend': 0.30
        }
        
    async def analyze(self, stock_data: StockData) -> TechnicalAnalysis:
        """
        Perform comprehensive technical analysis.
        
        Args:
            stock_data: StockData object containing price data
            
        Returns:
            TechnicalAnalysis object with complete analysis
        """
        if stock_data.price_data is None or stock_data.price_data.empty:
            raise ValueError(f"No price data available for {stock_data.symbol}")
            
        # Calculate technical indicators
        enriched_data = self._calculate_indicators(stock_data.price_data)
        
        # Perform individual analyses
        trend_analysis = self._analyze_trend(enriched_data)
        momentum_analysis = self._analyze_momentum(enriched_data)
        volume_analysis = self._analyze_volume(enriched_data)
        support_resistance = self._analyze_support_resistance(enriched_data)
        pattern_analysis = self._analyze_patterns(enriched_data)
        
        # Calculate overall score and signal
        overall_score = self._calculate_overall_score(
            trend_analysis, momentum_analysis, volume_analysis,
            support_resistance, pattern_analysis
        )
        
        overall_signal = self._generate_overall_signal(overall_score, enriched_data)
        
        return TechnicalAnalysis(
            symbol=stock_data.symbol,
            analysis_date=datetime.now(),
            overall_signal=overall_signal,
            trend_analysis=trend_analysis,
            momentum_analysis=momentum_analysis,
            volume_analysis=volume_analysis,
            support_resistance=support_resistance,
            pattern_analysis=pattern_analysis,
            score=overall_score
        )
        
    def _calculate_indicators(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        df = price_data.copy()
        
        # Ensure we have required columns
        if 'close' not in df.columns:
            raise ValueError("Price data must contain 'close' column")
            
        # Basic price info
        high = df['high'] if 'high' in df.columns else df['close']
        low = df['low'] if 'low' in df.columns else df['close']
        volume = df['volume'] if 'volume' in df.columns else pd.Series(index=df.index, data=0)
        
        # Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(high, low, df['close'])
        
        # Williams %R
        df['williams_r'] = self._calculate_williams_r(high, low, df['close'])
        
        # ADX (Average Directional Index)
        df['adx'] = self._calculate_adx(high, low, df['close'])
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
        
        # Volume indicators
        df['obv'] = self._calculate_obv(df['close'], volume)
        df['volume_sma'] = volume.rolling(window=20).mean()
        
        # Price momentum
        df['momentum'] = df['close'].pct_change(periods=10)
        df['rate_of_change'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
        
        # Average True Range
        df['atr'] = self._calculate_atr(high, low, df['close'])
        
        # Commodity Channel Index
        df['cci'] = self._calculate_cci(high, low, df['close'])
        
        return df
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        if HAS_TALIB:
            return talib.RSI(prices.values, timeperiod=period)
        else:
            # Manual RSI calculation
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, 
                            close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic oscillator."""
        if HAS_TALIB:
            stoch_k, stoch_d = talib.STOCH(high.values, low.values, close.values,
                                         fastk_period=period, slowk_period=3, slowd_period=3)
            return pd.Series(stoch_k, index=close.index), pd.Series(stoch_d, index=close.index)
        else:
            # Manual calculation
            lowest_low = low.rolling(window=period).min()
            highest_high = high.rolling(window=period).max()
            
            stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
            stoch_d = stoch_k.rolling(window=3).mean()
            
            return stoch_k, stoch_d
            
    def _calculate_williams_r(self, high: pd.Series, low: pd.Series, 
                            close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        if HAS_TALIB:
            return talib.WILLR(high.values, low.values, close.values, timeperiod=period)
        else:
            # Manual calculation
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
            
            williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
            
            return williams_r
            
    def _calculate_adx(self, high: pd.Series, low: pd.Series, 
                      close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate ADX (Average Directional Index)."""
        if HAS_TALIB:
            return talib.ADX(high.values, low.values, close.values, timeperiod=period)
        else:
            # Simplified ADX calculation
            # This is a basic implementation; full ADX is more complex
            tr = pd.concat([
                high - low,
                abs(high - close.shift(1)),
                abs(low - close.shift(1))
            ], axis=1).max(axis=1)
            
            atr = tr.rolling(window=period).mean()
            return atr  # Simplified version
            
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, 
                                 std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        if HAS_TALIB:
            upper, middle, lower = talib.BBANDS(prices.values, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
            return (pd.Series(upper, index=prices.index), 
                   pd.Series(middle, index=prices.index), 
                   pd.Series(lower, index=prices.index))
        else:
            # Manual calculation
            middle = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            
            return upper, middle, lower
            
    def _calculate_obv(self, prices: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume."""
        if HAS_TALIB:
            return talib.OBV(prices.values, volume.values)
        else:
            # Manual calculation
            obv = pd.Series(index=prices.index, dtype=float)
            obv.iloc[0] = volume.iloc[0]
            
            for i in range(1, len(prices)):
                if prices.iloc[i] > prices.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
                elif prices.iloc[i] < prices.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
                    
            return obv
            
    def _calculate_atr(self, high: pd.Series, low: pd.Series, 
                      close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        if HAS_TALIB:
            return talib.ATR(high.values, low.values, close.values, timeperiod=period)
        else:
            # Manual calculation
            tr = pd.concat([
                high - low,
                abs(high - close.shift(1)),
                abs(low - close.shift(1))
            ], axis=1).max(axis=1)
            
            atr = tr.rolling(window=period).mean()
            
            return atr
            
    def _calculate_cci(self, high: pd.Series, low: pd.Series, 
                      close: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index."""
        if HAS_TALIB:
            return talib.CCI(high.values, low.values, close.values, timeperiod=period)
        else:
            # Manual calculation
            typical_price = (high + low + close) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            
            cci = (typical_price - sma_tp) / (0.015 * mad)
            
            return cci
            
    def _analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend indicators."""
        latest = data.iloc[-1]
        
        trend_scores = {}
        
        # SMA Trend Analysis
        sma_trend_score = 0
        if not pd.isna(latest['sma_20']) and not pd.isna(latest['sma_50']) and not pd.isna(latest['sma_200']):
            if latest['close'] > latest['sma_20'] > latest['sma_50'] > latest['sma_200']:
                sma_trend_score = 90  # Strong uptrend
            elif latest['close'] > latest['sma_20'] > latest['sma_50']:
                sma_trend_score = 70  # Moderate uptrend
            elif latest['close'] > latest['sma_20']:
                sma_trend_score = 60  # Weak uptrend
            elif latest['close'] < latest['sma_20'] < latest['sma_50'] < latest['sma_200']:
                sma_trend_score = 10  # Strong downtrend
            elif latest['close'] < latest['sma_20'] < latest['sma_50']:
                sma_trend_score = 30  # Moderate downtrend
            else:
                sma_trend_score = 40  # Weak downtrend
        else:
            sma_trend_score = 50
            
        trend_scores['sma_trend'] = sma_trend_score
        
        # EMA Trend Analysis
        ema_trend_score = 0
        if not pd.isna(latest['ema_12']) and not pd.isna(latest['ema_26']) and not pd.isna(latest['ema_50']):
            if latest['close'] > latest['ema_12'] > latest['ema_26'] > latest['ema_50']:
                ema_trend_score = 90
            elif latest['close'] > latest['ema_12'] > latest['ema_26']:
                ema_trend_score = 70
            elif latest['close'] > latest['ema_12']:
                ema_trend_score = 60
            elif latest['close'] < latest['ema_12'] < latest['ema_26'] < latest['ema_50']:
                ema_trend_score = 10
            elif latest['close'] < latest['ema_12'] < latest['ema_26']:
                ema_trend_score = 30
            else:
                ema_trend_score = 40
        else:
            ema_trend_score = 50
            
        trend_scores['ema_trend'] = ema_trend_score
        
        # MACD Trend Analysis
        macd_trend_score = 0
        if not pd.isna(latest['macd']) and not pd.isna(latest['macd_signal']):
            if latest['macd'] > latest['macd_signal'] and latest['macd'] > 0:
                macd_trend_score = 80  # Bullish above zero
            elif latest['macd'] > latest['macd_signal'] and latest['macd'] < 0:
                macd_trend_score = 60  # Bullish below zero
            elif latest['macd'] < latest['macd_signal'] and latest['macd'] > 0:
                macd_trend_score = 40  # Bearish above zero
            else:
                macd_trend_score = 20  # Bearish below zero
        else:
            macd_trend_score = 50
            
        trend_scores['macd_trend'] = macd_trend_score
        
        # ADX Trend Strength
        adx_trend_score = 0
        if not pd.isna(latest['adx']):
            if latest['adx'] > 25:
                adx_trend_score = 80  # Strong trend
            elif latest['adx'] > 20:
                adx_trend_score = 60  # Moderate trend
            else:
                adx_trend_score = 40  # Weak trend
        else:
            adx_trend_score = 50
            
        trend_scores['adx_trend'] = adx_trend_score
        
        # Calculate weighted trend score
        weighted_score = sum(
            trend_scores.get(metric, 50) * weight
            for metric, weight in self.trend_weights.items()
        )
        
        return {
            'scores': trend_scores,
            'weighted_score': weighted_score,
            'indicators': {
                'sma_20': latest.get('sma_20'),
                'sma_50': latest.get('sma_50'),
                'sma_200': latest.get('sma_200'),
                'ema_12': latest.get('ema_12'),
                'ema_26': latest.get('ema_26'),
                'ema_50': latest.get('ema_50'),
                'macd': latest.get('macd'),
                'macd_signal': latest.get('macd_signal'),
                'adx': latest.get('adx')
            },
            'signal': self._score_to_signal(weighted_score)
        }
        
    def _analyze_momentum(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze momentum indicators."""
        latest = data.iloc[-1]
        
        momentum_scores = {}
        
        # RSI Analysis
        rsi_score = 0
        if not pd.isna(latest['rsi']):
            if latest['rsi'] < 30:
                rsi_score = 80  # Oversold - potential buy
            elif latest['rsi'] < 40:
                rsi_score = 60  # Approaching oversold
            elif latest['rsi'] > 70:
                rsi_score = 20  # Overbought - potential sell
            elif latest['rsi'] > 60:
                rsi_score = 40  # Approaching overbought
            else:
                rsi_score = 50  # Neutral
        else:
            rsi_score = 50
            
        momentum_scores['rsi'] = rsi_score
        
        # MACD Momentum
        macd_momentum_score = 0
        if not pd.isna(latest['macd_histogram']):
            # Look at MACD histogram trend
            recent_histogram = data['macd_histogram'].tail(3)
            if recent_histogram.iloc[-1] > recent_histogram.iloc[-2] > recent_histogram.iloc[-3]:
                macd_momentum_score = 80  # Increasing momentum
            elif recent_histogram.iloc[-1] > recent_histogram.iloc[-2]:
                macd_momentum_score = 60  # Improving momentum
            elif recent_histogram.iloc[-1] < recent_histogram.iloc[-2] < recent_histogram.iloc[-3]:
                macd_momentum_score = 20  # Decreasing momentum
            elif recent_histogram.iloc[-1] < recent_histogram.iloc[-2]:
                macd_momentum_score = 40  # Weakening momentum
            else:
                macd_momentum_score = 50
        else:
            macd_momentum_score = 50
            
        momentum_scores['macd'] = macd_momentum_score
        
        # Stochastic Analysis
        stoch_score = 0
        if not pd.isna(latest['stoch_k']) and not pd.isna(latest['stoch_d']):
            if latest['stoch_k'] < 20 and latest['stoch_d'] < 20:
                stoch_score = 80  # Oversold
            elif latest['stoch_k'] > 80 and latest['stoch_d'] > 80:
                stoch_score = 20  # Overbought
            elif latest['stoch_k'] > latest['stoch_d']:
                stoch_score = 60  # Bullish crossover
            else:
                stoch_score = 40  # Bearish crossover
        else:
            stoch_score = 50
            
        momentum_scores['stochastic'] = stoch_score
        
        # Williams %R Analysis
        williams_score = 0
        if not pd.isna(latest['williams_r']):
            if latest['williams_r'] < -80:
                williams_score = 80  # Oversold
            elif latest['williams_r'] > -20:
                williams_score = 20  # Overbought
            else:
                williams_score = 50  # Neutral
        else:
            williams_score = 50
            
        momentum_scores['williams_r'] = williams_score
        
        # Calculate weighted momentum score
        weighted_score = sum(
            momentum_scores.get(metric, 50) * weight
            for metric, weight in self.momentum_weights.items()
        )
        
        return {
            'scores': momentum_scores,
            'weighted_score': weighted_score,
            'indicators': {
                'rsi': latest.get('rsi'),
                'macd_histogram': latest.get('macd_histogram'),
                'stoch_k': latest.get('stoch_k'),
                'stoch_d': latest.get('stoch_d'),
                'williams_r': latest.get('williams_r')
            },
            'signal': self._score_to_signal(weighted_score)
        }
        
    def _analyze_volume(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume indicators."""
        latest = data.iloc[-1]
        
        volume_scores = {}
        
        # Volume Trend Analysis
        volume_trend_score = 0
        if 'volume' in data.columns and not pd.isna(latest['volume']) and not pd.isna(latest['volume_sma']):
            if latest['volume'] > latest['volume_sma'] * 1.5:
                volume_trend_score = 80  # High volume
            elif latest['volume'] > latest['volume_sma'] * 1.2:
                volume_trend_score = 60  # Above average volume
            elif latest['volume'] < latest['volume_sma'] * 0.8:
                volume_trend_score = 40  # Below average volume
            else:
                volume_trend_score = 50  # Average volume
        else:
            volume_trend_score = 50
            
        volume_scores['volume_trend'] = volume_trend_score
        
        # OBV Analysis
        obv_score = 0
        if not pd.isna(latest['obv']):
            # Check OBV trend over last 5 days
            obv_trend = data['obv'].tail(5)
            if obv_trend.iloc[-1] > obv_trend.iloc[-5]:
                obv_score = 70  # OBV increasing
            elif obv_trend.iloc[-1] < obv_trend.iloc[-5]:
                obv_score = 30  # OBV decreasing
            else:
                obv_score = 50  # OBV stable
        else:
            obv_score = 50
            
        volume_scores['obv'] = obv_score
        
        # Volume Price Trend
        vpt_score = 0
        if 'volume' in data.columns and len(data) > 1:
            price_change = data['close'].pct_change()
            volume_weighted_price = (price_change * data['volume']).rolling(window=10).mean()
            
            if not pd.isna(volume_weighted_price.iloc[-1]):
                if volume_weighted_price.iloc[-1] > 0:
                    vpt_score = 70  # Positive volume-weighted price trend
                elif volume_weighted_price.iloc[-1] < 0:
                    vpt_score = 30  # Negative volume-weighted price trend
                else:
                    vpt_score = 50
            else:
                vpt_score = 50
        else:
            vpt_score = 50
            
        volume_scores['volume_price_trend'] = vpt_score
        
        # Calculate weighted volume score
        weighted_score = sum(
            volume_scores.get(metric, 50) * weight
            for metric, weight in self.volume_weights.items()
        )
        
        return {
            'scores': volume_scores,
            'weighted_score': weighted_score,
            'indicators': {
                'volume': latest.get('volume'),
                'volume_sma': latest.get('volume_sma'),
                'obv': latest.get('obv')
            },
            'signal': self._score_to_signal(weighted_score)
        }
        
    def _analyze_support_resistance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze support and resistance levels."""
        
        # Calculate recent highs and lows
        recent_data = data.tail(50)  # Last 50 periods
        current_price = data['close'].iloc[-1]
        
        # Find pivot points
        highs = recent_data['high'].rolling(window=5, center=True).max()
        lows = recent_data['low'].rolling(window=5, center=True).min()
        
        # Identify support and resistance levels
        resistance_levels = []
        support_levels = []
        
        for i in range(2, len(recent_data) - 2):
            # Check if it's a pivot high (resistance)
            if (recent_data['high'].iloc[i] == highs.iloc[i] and 
                recent_data['high'].iloc[i] > recent_data['high'].iloc[i-1] and
                recent_data['high'].iloc[i] > recent_data['high'].iloc[i-2] and
                recent_data['high'].iloc[i] > recent_data['high'].iloc[i+1] and
                recent_data['high'].iloc[i] > recent_data['high'].iloc[i+2]):
                resistance_levels.append(recent_data['high'].iloc[i])
                
            # Check if it's a pivot low (support)
            if (recent_data['low'].iloc[i] == lows.iloc[i] and 
                recent_data['low'].iloc[i] < recent_data['low'].iloc[i-1] and
                recent_data['low'].iloc[i] < recent_data['low'].iloc[i-2] and
                recent_data['low'].iloc[i] < recent_data['low'].iloc[i+1] and
                recent_data['low'].iloc[i] < recent_data['low'].iloc[i+2]):
                support_levels.append(recent_data['low'].iloc[i])
        
        # Find nearest support and resistance
        resistance_above = [r for r in resistance_levels if r > current_price]
        support_below = [s for s in support_levels if s < current_price]
        
        nearest_resistance = min(resistance_above) if resistance_above else None
        nearest_support = max(support_below) if support_below else None
        
        # Calculate support/resistance strength
        sr_score = 50  # Default neutral
        
        if nearest_resistance and nearest_support:
            # Distance to support/resistance
            distance_to_resistance = (nearest_resistance - current_price) / current_price
            distance_to_support = (current_price - nearest_support) / current_price
            
            if distance_to_support < 0.02:  # Within 2% of support
                sr_score = 30  # Near support, potential bounce
            elif distance_to_resistance < 0.02:  # Within 2% of resistance
                sr_score = 70  # Near resistance, potential reversal
            elif distance_to_support < distance_to_resistance:
                sr_score = 60  # Closer to support
            else:
                sr_score = 40  # Closer to resistance
        
        return {
            'score': sr_score,
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'current_price': current_price,
            'signal': self._score_to_signal(sr_score)
        }
        
    def _analyze_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze chart patterns."""
        
        pattern_scores = {}
        
        # Bollinger Bands Pattern
        bb_score = 50
        if not pd.isna(data['bb_upper'].iloc[-1]) and not pd.isna(data['bb_lower'].iloc[-1]):
            current_price = data['close'].iloc[-1]
            bb_upper = data['bb_upper'].iloc[-1]
            bb_lower = data['bb_lower'].iloc[-1]
            bb_middle = data['bb_middle'].iloc[-1]
            
            if current_price <= bb_lower:
                bb_score = 80  # Oversold - potential buy
            elif current_price >= bb_upper:
                bb_score = 20  # Overbought - potential sell
            elif current_price > bb_middle:
                bb_score = 60  # Above middle band
            else:
                bb_score = 40  # Below middle band
                
        pattern_scores['bollinger_bands'] = bb_score
        
        # Moving Average Convergence Pattern
        ma_convergence_score = 50
        if (not pd.isna(data['sma_20'].iloc[-1]) and 
            not pd.isna(data['sma_50'].iloc[-1]) and
            len(data) >= 10):
            
            # Check if short MA is converging with long MA
            ma_diff_current = data['sma_20'].iloc[-1] - data['sma_50'].iloc[-1]
            ma_diff_prev = data['sma_20'].iloc[-5] - data['sma_50'].iloc[-5]
            
            if abs(ma_diff_current) < abs(ma_diff_prev):
                if ma_diff_current > 0:
                    ma_convergence_score = 60  # Bullish convergence
                else:
                    ma_convergence_score = 40  # Bearish convergence
            else:
                if ma_diff_current > ma_diff_prev:
                    ma_convergence_score = 70  # Bullish divergence
                else:
                    ma_convergence_score = 30  # Bearish divergence
                    
        pattern_scores['ma_convergence'] = ma_convergence_score
        
        # Price momentum pattern
        momentum_pattern_score = 50
        if len(data) >= 10:
            recent_returns = data['close'].pct_change().tail(10)
            if recent_returns.sum() > 0.05:  # 5% gain in last 10 periods
                momentum_pattern_score = 70  # Strong upward momentum
            elif recent_returns.sum() < -0.05:  # 5% loss in last 10 periods
                momentum_pattern_score = 30  # Strong downward momentum
                
        pattern_scores['momentum_pattern'] = momentum_pattern_score
        
        # Overall pattern score
        pattern_score = np.mean(list(pattern_scores.values()))
        
        return {
            'scores': pattern_scores,
            'weighted_score': pattern_score,
            'signal': self._score_to_signal(pattern_score)
        }
        
    def _calculate_overall_score(self, trend_analysis: Dict[str, Any],
                               momentum_analysis: Dict[str, Any],
                               volume_analysis: Dict[str, Any],
                               support_resistance: Dict[str, Any],
                               pattern_analysis: Dict[str, Any]) -> float:
        """Calculate overall technical score."""
        
        # Weights for different analysis categories
        category_weights = {
            'trend': 0.35,
            'momentum': 0.30,
            'volume': 0.15,
            'support_resistance': 0.10,
            'patterns': 0.10
        }
        
        overall_score = (
            trend_analysis['weighted_score'] * category_weights['trend'] +
            momentum_analysis['weighted_score'] * category_weights['momentum'] +
            volume_analysis['weighted_score'] * category_weights['volume'] +
            support_resistance['score'] * category_weights['support_resistance'] +
            pattern_analysis['weighted_score'] * category_weights['patterns']
        )
        
        return overall_score
        
    def _generate_overall_signal(self, score: float, data: pd.DataFrame) -> TechnicalSignal:
        """Generate overall technical signal based on score."""
        
        latest = data.iloc[-1]
        
        if score >= 70:
            signal = 'BUY'
            strength = min(1.0, score / 100)
            reason = "Strong bullish technical signals with good momentum and trend"
        elif score >= 55:
            signal = 'BUY'
            strength = min(0.8, score / 100)
            reason = "Moderately bullish technical signals"
        elif score >= 45:
            signal = 'HOLD'
            strength = min(0.6, score / 100)
            reason = "Mixed technical signals with no clear direction"
        elif score >= 30:
            signal = 'SELL'
            strength = min(0.4, score / 100)
            reason = "Bearish technical signals with weakening momentum"
        else:
            signal = 'SELL'
            strength = min(0.2, score / 100)
            reason = "Strong bearish technical signals"
            
        return TechnicalSignal(
            signal=signal,
            strength=strength,
            reason=reason,
            indicators={
                'overall_score': score,
                'current_price': latest['close'],
                'rsi': latest.get('rsi'),
                'macd': latest.get('macd'),
                'volume': latest.get('volume')
            }
        )
        
    def _score_to_signal(self, score: float) -> str:
        """Convert numeric score to signal."""
        if score >= 60:
            return 'BUY'
        elif score >= 40:
            return 'HOLD'
        else:
            return 'SELL' 