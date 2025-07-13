"""
Unit tests for analysis modules.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

from src.analysis.fundamental import FundamentalAnalyzer, FundamentalAnalysis, FundamentalSignal
from src.analysis.technical import TechnicalAnalyzer, TechnicalAnalysis, TechnicalSignal
from src.data_sources.base import StockData


class TestFundamentalAnalyzer:
    """Test fundamental analysis functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.analyzer = FundamentalAnalyzer()
        
    def create_mock_stock_data(self, fundamental_data=None, financial_statements=None):
        """Create mock stock data for testing."""
        price_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [99, 100, 101],
            'close': [104, 105, 106],
            'volume': [1000, 1100, 1200]
        })
        
        if fundamental_data is None:
            fundamental_data = {
                'pe_ratio': 25.5,
                'price_to_book': 3.2,
                'price_to_sales': 2.1,
                'peg_ratio': 1.1,
                'enterprise_to_ebitda': 14.5,
                'profit_margin': 0.15,
                'operating_margin': 0.12,
                'return_on_equity': 0.18,
                'return_on_assets': 0.08,
                'current_ratio': 1.5,
                'debt_to_equity': 0.4,
                'revenue_growth': 0.10,
                'earnings_growth': 0.08,
                'beta': 1.2,
                'market_cap': 2000000000
            }
        
        return StockData(
            symbol="AAPL",
            price_data=price_data,
            fundamental_data=fundamental_data,
            financial_statements=financial_statements
        )
        
    @pytest.mark.asyncio
    async def test_analyze_success(self):
        """Test successful fundamental analysis."""
        stock_data = self.create_mock_stock_data()
        
        result = await self.analyzer.analyze(stock_data)
        
        assert isinstance(result, FundamentalAnalysis)
        assert result.symbol == "AAPL"
        assert result.overall_signal is not None
        assert result.score >= 0 and result.score <= 100
        assert result.valuation_analysis is not None
        assert result.profitability_analysis is not None
        assert result.financial_health_analysis is not None
        assert result.growth_analysis is not None
        
    @pytest.mark.asyncio
    async def test_analyze_no_fundamental_data(self):
        """Test analysis with no fundamental data."""
        stock_data = self.create_mock_stock_data(fundamental_data={})
        
        with pytest.raises(ValueError, match="No fundamental data available"):
            await self.analyzer.analyze(stock_data)
            
    def test_analyze_valuation_good_metrics(self):
        """Test valuation analysis with good metrics."""
        stock_data = self.create_mock_stock_data({
            'pe_ratio': 15.0,  # Good PE
            'price_to_book': 2.0,  # Good P/B
            'price_to_sales': 1.5,  # Good P/S
            'peg_ratio': 0.8,  # Good PEG
            'enterprise_to_ebitda': 8.0  # Good EV/EBITDA
        })
        
        result = self.analyzer._analyze_valuation(stock_data)
        
        assert result['weighted_score'] > 60  # Should be good score
        assert result['signal'] in ['BUY', 'HOLD']
        assert 'pe_ratio' in result['scores']
        assert result['scores']['pe_ratio'] > 60
        
    def test_analyze_valuation_poor_metrics(self):
        """Test valuation analysis with poor metrics."""
        stock_data = self.create_mock_stock_data({
            'pe_ratio': 50.0,  # Poor PE
            'price_to_book': 8.0,  # Poor P/B
            'price_to_sales': 5.0,  # Poor P/S
            'peg_ratio': 3.0,  # Poor PEG
            'enterprise_to_ebitda': 25.0  # Poor EV/EBITDA
        })
        
        result = self.analyzer._analyze_valuation(stock_data)
        
        assert result['weighted_score'] < 50  # Should be poor score
        assert result['signal'] in ['SELL', 'HOLD']
        
    def test_analyze_profitability_good_metrics(self):
        """Test profitability analysis with good metrics."""
        stock_data = self.create_mock_stock_data({
            'profit_margin': 0.25,  # Good profit margin
            'operating_margin': 0.20,  # Good operating margin
            'return_on_equity': 0.20,  # Good ROE
            'return_on_assets': 0.12   # Good ROA
        })
        
        result = self.analyzer._analyze_profitability(stock_data)
        
        assert result['weighted_score'] > 60
        assert result['signal'] in ['BUY', 'HOLD']
        assert all(score > 60 for score in result['scores'].values())
        
    def test_analyze_profitability_poor_metrics(self):
        """Test profitability analysis with poor metrics."""
        stock_data = self.create_mock_stock_data({
            'profit_margin': 0.02,  # Poor profit margin
            'operating_margin': 0.01,  # Poor operating margin
            'return_on_equity': 0.03,  # Poor ROE
            'return_on_assets': 0.01   # Poor ROA
        })
        
        result = self.analyzer._analyze_profitability(stock_data)
        
        assert result['weighted_score'] < 50
        assert result['signal'] in ['SELL', 'HOLD']
        
    def test_analyze_financial_health_good_metrics(self):
        """Test financial health analysis with good metrics."""
        stock_data = self.create_mock_stock_data({
            'current_ratio': 2.5,  # Good current ratio
            'debt_to_equity': 0.2,  # Good debt/equity
            'quick_ratio': 1.5     # Good quick ratio
        })
        
        result = self.analyzer._analyze_financial_health(stock_data)
        
        assert result['weighted_score'] > 60
        assert result['signal'] in ['BUY', 'HOLD']
        
    def test_analyze_financial_health_poor_metrics(self):
        """Test financial health analysis with poor metrics."""
        stock_data = self.create_mock_stock_data({
            'current_ratio': 0.8,  # Poor current ratio
            'debt_to_equity': 2.0,  # Poor debt/equity
            'quick_ratio': 0.3     # Poor quick ratio
        })
        
        result = self.analyzer._analyze_financial_health(stock_data)
        
        assert result['weighted_score'] < 50
        assert result['signal'] in ['SELL', 'HOLD']
        
    def test_analyze_growth_good_metrics(self):
        """Test growth analysis with good metrics."""
        stock_data = self.create_mock_stock_data({
            'revenue_growth': 0.25,  # Good revenue growth
            'earnings_growth': 0.20  # Good earnings growth
        })
        
        result = self.analyzer._analyze_growth(stock_data)
        
        assert result['weighted_score'] > 60
        assert result['signal'] in ['BUY', 'HOLD']
        
    def test_score_to_signal_mapping(self):
        """Test score to signal mapping."""
        assert self.analyzer._score_to_signal(80) == 'BUY'
        assert self.analyzer._score_to_signal(60) == 'HOLD'
        assert self.analyzer._score_to_signal(30) == 'SELL'


class TestTechnicalAnalyzer:
    """Test technical analysis functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.analyzer = TechnicalAnalyzer()
        
    def create_mock_price_data(self, days=50):
        """Create mock price data for testing."""
        dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
        
        # Create realistic price data with some trend
        base_price = 100
        prices = []
        volumes = []
        
        for i in range(days):
            # Add some trend and noise
            trend = i * 0.1  # Slight upward trend
            noise = np.random.normal(0, 2)
            price = base_price + trend + noise
            
            prices.append({
                'open': price + np.random.normal(0, 0.5),
                'high': price + abs(np.random.normal(2, 1)),
                'low': price - abs(np.random.normal(2, 1)),
                'close': price,
                'volume': 1000 + np.random.normal(0, 200)
            })
            
        df = pd.DataFrame(prices, index=dates)
        return df
        
    def create_mock_stock_data(self, days=50):
        """Create mock stock data for testing."""
        price_data = self.create_mock_price_data(days)
        
        return StockData(
            symbol="AAPL",
            price_data=price_data,
            fundamental_data={"pe_ratio": 25.5}
        )
        
    @pytest.mark.asyncio
    async def test_analyze_success(self):
        """Test successful technical analysis."""
        stock_data = self.create_mock_stock_data()
        
        result = await self.analyzer.analyze(stock_data)
        
        assert isinstance(result, TechnicalAnalysis)
        assert result.symbol == "AAPL"
        assert result.overall_signal is not None
        assert result.score >= 0 and result.score <= 100
        assert result.trend_analysis is not None
        assert result.momentum_analysis is not None
        assert result.volume_analysis is not None
        assert result.support_resistance is not None
        assert result.pattern_analysis is not None
        
    @pytest.mark.asyncio
    async def test_analyze_no_price_data(self):
        """Test analysis with no price data."""
        stock_data = StockData(symbol="AAPL", price_data=None)
        
        with pytest.raises(ValueError, match="No price data available"):
            await self.analyzer.analyze(stock_data)
            
    @pytest.mark.asyncio
    async def test_analyze_empty_price_data(self):
        """Test analysis with empty price data."""
        stock_data = StockData(symbol="AAPL", price_data=pd.DataFrame())
        
        with pytest.raises(ValueError, match="No price data available"):
            await self.analyzer.analyze(stock_data)
            
    def test_calculate_indicators(self):
        """Test technical indicators calculation."""
        stock_data = self.create_mock_stock_data(100)  # More days for better indicators
        
        enriched_data = self.analyzer._calculate_indicators(stock_data.price_data)
        
        # Check that indicators are calculated
        assert 'sma_20' in enriched_data.columns
        assert 'sma_50' in enriched_data.columns
        assert 'ema_12' in enriched_data.columns
        assert 'ema_26' in enriched_data.columns
        assert 'macd' in enriched_data.columns
        assert 'macd_signal' in enriched_data.columns
        assert 'rsi' in enriched_data.columns
        assert 'obv' in enriched_data.columns
        
        # Check that indicators have reasonable values
        assert not enriched_data['sma_20'].isna().all()
        assert not enriched_data['rsi'].isna().all()
        assert enriched_data['rsi'].max() <= 100
        assert enriched_data['rsi'].min() >= 0
        
    def test_calculate_rsi_manual(self):
        """Test RSI calculation without TA-Lib."""
        prices = pd.Series([100, 101, 99, 102, 98, 103, 97, 104, 96, 105])
        
        rsi = self.analyzer._calculate_rsi(prices, period=5)
        
        # RSI should be between 0 and 100
        assert rsi.min() >= 0
        assert rsi.max() <= 100
        assert not rsi.isna().all()
        
    def test_calculate_stochastic_manual(self):
        """Test Stochastic calculation without TA-Lib."""
        high = pd.Series([105, 106, 104, 107, 103, 108, 102, 109, 101, 110])
        low = pd.Series([99, 100, 98, 101, 97, 102, 96, 103, 95, 104])
        close = pd.Series([104, 105, 103, 106, 102, 107, 101, 108, 100, 109])
        
        stoch_k, stoch_d = self.analyzer._calculate_stochastic(high, low, close)
        
        # Stochastic should be between 0 and 100
        assert stoch_k.min() >= 0
        assert stoch_k.max() <= 100
        assert stoch_d.min() >= 0
        assert stoch_d.max() <= 100
        
    def test_analyze_trend_bullish(self):
        """Test trend analysis with bullish trend."""
        # Create data with clear upward trend
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        prices = [100 + i * 0.5 for i in range(100)]  # Strong upward trend
        
        data = pd.DataFrame({
            'close': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'volume': [1000] * 100
        }, index=dates)
        
        # Add calculated indicators
        enriched_data = self.analyzer._calculate_indicators(data)
        
        trend_analysis = self.analyzer._analyze_trend(enriched_data)
        
        assert trend_analysis['weighted_score'] > 50  # Should be bullish
        assert trend_analysis['signal'] in ['BUY', 'HOLD']
        
    def test_analyze_momentum_oversold(self):
        """Test momentum analysis with oversold condition."""
        # Create data that should result in oversold RSI
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        prices = [100 - i * 2 for i in range(50)]  # Strong downward trend
        
        data = pd.DataFrame({
            'close': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'volume': [1000] * 50
        }, index=dates)
        
        # Manually set low RSI to simulate oversold condition
        data['rsi'] = [25] * 50  # Oversold RSI
        
        momentum_analysis = self.analyzer._analyze_momentum(data)
        
        # Should indicate potential buy due to oversold condition
        assert momentum_analysis['scores']['rsi'] > 50
        
    def test_analyze_volume_high_volume(self):
        """Test volume analysis with high volume."""
        data = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [2000, 2100, 2200],  # High volume
            'volume_sma': [1000, 1000, 1000],  # Average volume
            'obv': [1000, 2000, 3000]  # Increasing OBV
        })
        
        volume_analysis = self.analyzer._analyze_volume(data)
        
        assert volume_analysis['scores']['volume_trend'] > 50
        assert volume_analysis['scores']['obv'] > 50
        
    def test_analyze_support_resistance(self):
        """Test support and resistance analysis."""
        # Create data with clear support and resistance levels
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Price that oscillates between support (95) and resistance (105)
        prices = []
        for i in range(100):
            if i % 20 < 10:
                prices.append(95 + (i % 10) * 1)  # Moving up from support
            else:
                prices.append(105 - (i % 10) * 1)  # Moving down from resistance
                
        data = pd.DataFrame({
            'close': prices,
            'high': [p + 2 for p in prices],
            'low': [p - 2 for p in prices],
            'volume': [1000] * 100
        }, index=dates)
        
        sr_analysis = self.analyzer._analyze_support_resistance(data)
        
        assert sr_analysis['score'] >= 0
        assert sr_analysis['current_price'] is not None
        assert 'support_levels' in sr_analysis
        assert 'resistance_levels' in sr_analysis
        
    def test_score_to_signal_mapping(self):
        """Test score to signal mapping."""
        assert self.analyzer._score_to_signal(70) == 'BUY'
        assert self.analyzer._score_to_signal(50) == 'HOLD'
        assert self.analyzer._score_to_signal(30) == 'SELL'
        
    def test_calculate_overall_score(self):
        """Test overall score calculation."""
        # Mock analysis results
        trend_analysis = {'weighted_score': 70}
        momentum_analysis = {'weighted_score': 60}
        volume_analysis = {'weighted_score': 50}
        support_resistance = {'score': 40}
        pattern_analysis = {'weighted_score': 30}
        
        overall_score = self.analyzer._calculate_overall_score(
            trend_analysis, momentum_analysis, volume_analysis,
            support_resistance, pattern_analysis
        )
        
        assert overall_score >= 0
        assert overall_score <= 100
        
        # Should be weighted average
        expected = (70 * 0.35 + 60 * 0.30 + 50 * 0.15 + 40 * 0.10 + 30 * 0.10)
        assert abs(overall_score - expected) < 0.1
        
    def test_generate_overall_signal_buy(self):
        """Test overall signal generation for buy."""
        mock_data = pd.DataFrame({
            'close': [100, 101, 102],
            'rsi': [50, 51, 52],
            'macd': [0.1, 0.2, 0.3],
            'volume': [1000, 1100, 1200]
        })
        
        signal = self.analyzer._generate_overall_signal(75, mock_data)
        
        assert signal.signal == 'BUY'
        assert signal.strength > 0.7
        assert signal.reason is not None
        assert signal.indicators is not None
        
    def test_generate_overall_signal_sell(self):
        """Test overall signal generation for sell."""
        mock_data = pd.DataFrame({
            'close': [100, 99, 98],
            'rsi': [30, 25, 20],
            'macd': [-0.1, -0.2, -0.3],
            'volume': [1000, 1100, 1200]
        })
        
        signal = self.analyzer._generate_overall_signal(25, mock_data)
        
        assert signal.signal == 'SELL'
        assert signal.strength < 0.3
        assert signal.reason is not None 