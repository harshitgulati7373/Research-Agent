"""
Integration tests for the complete stock analysis workflow.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, AsyncMock
import asyncio
from datetime import datetime

from src.agent.stock_agent import StockAnalysisAgent, StockRecommendation, AgentState
from src.data_sources.base import StockData
from src.data_sources.factory import DataSourceFactory
from src.analysis.fundamental import FundamentalAnalyzer
from src.analysis.technical import TechnicalAnalyzer


class TestStockAnalysisAgentIntegration:
    """Integration tests for the complete stock analysis workflow."""
    
    def setup_method(self):
        """Set up test environment."""
        self.agent = StockAnalysisAgent()
        
    def create_mock_stock_data(self):
        """Create comprehensive mock stock data."""
        # Create realistic price data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        price_data = pd.DataFrame({
            'open': [100 + i * 0.1 for i in range(100)],
            'high': [105 + i * 0.1 for i in range(100)],
            'low': [99 + i * 0.1 for i in range(100)],
            'close': [104 + i * 0.1 for i in range(100)],
            'volume': [1000 + i * 10 for i in range(100)]
        }, index=dates)
        
        # Create fundamental data
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
            'market_cap': 2000000000,
            'sector': 'Technology',
            'industry': 'Consumer Electronics'
        }
        
        # Create financial statements
        financial_statements = {
            'income_statement': pd.DataFrame({
                'Total Revenue': [100000, 110000, 120000],
                'Net Income': [20000, 22000, 24000],
                'Operating Income': [25000, 27500, 30000]
            }),
            'balance_sheet': pd.DataFrame({
                'Total Assets': [200000, 220000, 240000],
                'Total Debt': [50000, 55000, 60000],
                'Shareholders Equity': [150000, 165000, 180000]
            }),
            'cash_flow': pd.DataFrame({
                'Operating Cash Flow': [30000, 33000, 36000],
                'Free Cash Flow': [25000, 27500, 30000]
            })
        }
        
        return StockData(
            symbol="AAPL",
            price_data=price_data,
            fundamental_data=fundamental_data,
            financial_statements=financial_statements
        )
        
    @pytest.mark.asyncio
    async def test_complete_analysis_workflow_success(self):
        """Test the complete analysis workflow with successful data."""
        mock_stock_data = self.create_mock_stock_data()
        
        # Mock data factory to return our test data
        with patch.object(self.agent.data_factory, 'validate_symbol', return_value=True), \
             patch.object(self.agent.data_factory, 'get_stock_data', return_value=mock_stock_data):
            
            # Run the complete analysis
            recommendation = await self.agent.analyze_stock("AAPL")
            
            # Verify the recommendation
            assert isinstance(recommendation, StockRecommendation)
            assert recommendation.symbol == "AAPL"
            assert recommendation.recommendation in ['BUY', 'SELL', 'HOLD']
            assert 0.0 <= recommendation.confidence <= 1.0
            assert recommendation.fundamental_score >= 0
            assert recommendation.technical_score >= 0
            assert recommendation.combined_score >= 0
            assert recommendation.reasoning is not None
            assert recommendation.analysis_timestamp is not None
            
    @pytest.mark.asyncio
    async def test_complete_analysis_workflow_invalid_symbol(self):
        """Test the complete analysis workflow with invalid symbol."""
        # Mock data factory to return invalid symbol
        with patch.object(self.agent.data_factory, 'validate_symbol', return_value=False):
            
            # Run the analysis
            recommendation = await self.agent.analyze_stock("INVALID")
            
            # Should return a fallback recommendation
            assert isinstance(recommendation, StockRecommendation)
            assert recommendation.symbol == "INVALID"
            assert recommendation.recommendation == "HOLD"
            assert recommendation.confidence == 0.0
            assert "Analysis failed" in recommendation.reasoning
            
    @pytest.mark.asyncio
    async def test_complete_analysis_workflow_no_data(self):
        """Test the complete analysis workflow with no data available."""
        # Mock data factory to return empty data
        empty_stock_data = StockData(symbol="AAPL", price_data=pd.DataFrame())
        
        with patch.object(self.agent.data_factory, 'validate_symbol', return_value=True), \
             patch.object(self.agent.data_factory, 'get_stock_data', return_value=empty_stock_data):
            
            # Run the analysis
            recommendation = await self.agent.analyze_stock("AAPL")
            
            # Should return a fallback recommendation
            assert isinstance(recommendation, StockRecommendation)
            assert recommendation.symbol == "AAPL"
            assert recommendation.confidence == 0.0
            
    @pytest.mark.asyncio
    async def test_fetch_data_node_success(self):
        """Test the fetch_data node in isolation."""
        mock_stock_data = self.create_mock_stock_data()
        state = AgentState(symbol="AAPL")
        
        with patch.object(self.agent.data_factory, 'validate_symbol', return_value=True), \
             patch.object(self.agent.data_factory, 'get_stock_data', return_value=mock_stock_data):
            
            result = await self.agent._fetch_data(state)
            
            assert result["state"].stock_data is not None
            assert result["state"].stock_data.symbol == "AAPL"
            assert not result["state"].stock_data.price_data.empty
            assert len(result["state"].errors) == 0
            
    @pytest.mark.asyncio
    async def test_fetch_data_node_invalid_symbol(self):
        """Test the fetch_data node with invalid symbol."""
        state = AgentState(symbol="INVALID")
        
        with patch.object(self.agent.data_factory, 'validate_symbol', return_value=False):
            
            result = await self.agent._fetch_data(state)
            
            assert result["state"].stock_data is None
            assert len(result["state"].errors) > 0
            assert "Invalid or non-existent symbol" in result["state"].errors[0]
            
    @pytest.mark.asyncio
    async def test_fundamental_analysis_node_success(self):
        """Test the fundamental analysis node in isolation."""
        mock_stock_data = self.create_mock_stock_data()
        state = AgentState(symbol="AAPL", stock_data=mock_stock_data)
        
        result = await self.agent._run_fundamental_analysis(state)
        
        assert result["state"].fundamental_analysis is not None
        assert result["state"].fundamental_analysis.symbol == "AAPL"
        assert result["state"].fundamental_analysis.score >= 0
        assert result["state"].fundamental_analysis.overall_signal is not None
        
    @pytest.mark.asyncio
    async def test_fundamental_analysis_node_no_data(self):
        """Test the fundamental analysis node with no stock data."""
        state = AgentState(symbol="AAPL", stock_data=None)
        
        result = await self.agent._run_fundamental_analysis(state)
        
        assert result["state"].fundamental_analysis is None
        assert len(result["state"].errors) > 0
        assert "No stock data available" in result["state"].errors[0]
        
    @pytest.mark.asyncio
    async def test_technical_analysis_node_success(self):
        """Test the technical analysis node in isolation."""
        mock_stock_data = self.create_mock_stock_data()
        state = AgentState(symbol="AAPL", stock_data=mock_stock_data)
        
        result = await self.agent._run_technical_analysis(state)
        
        assert result["state"].technical_analysis is not None
        assert result["state"].technical_analysis.symbol == "AAPL"
        assert result["state"].technical_analysis.score >= 0
        assert result["state"].technical_analysis.overall_signal is not None
        
    @pytest.mark.asyncio
    async def test_technical_analysis_node_no_data(self):
        """Test the technical analysis node with no stock data."""
        state = AgentState(symbol="AAPL", stock_data=None)
        
        result = await self.agent._run_technical_analysis(state)
        
        assert result["state"].technical_analysis is None
        assert len(result["state"].errors) > 0
        assert "No stock data available" in result["state"].errors[0]
        
    @pytest.mark.asyncio
    async def test_combine_analysis_node(self):
        """Test the combine analysis node."""
        mock_stock_data = self.create_mock_stock_data()
        
        # Create mock analyses
        from src.analysis.fundamental import FundamentalAnalysis, FundamentalSignal
        from src.analysis.technical import TechnicalAnalysis, TechnicalSignal
        
        fundamental_signal = FundamentalSignal(
            signal="BUY",
            strength=0.8,
            reason="Strong fundamentals",
            metrics={"score": 75}
        )
        
        fundamental_analysis = FundamentalAnalysis(
            symbol="AAPL",
            analysis_date=datetime.now(),
            overall_signal=fundamental_signal,
            valuation_analysis={"weighted_score": 70},
            profitability_analysis={"weighted_score": 80},
            financial_health_analysis={"weighted_score": 75},
            growth_analysis={"weighted_score": 65},
            efficiency_analysis={"weighted_score": 70},
            market_position_analysis={"weighted_score": 75},
            risk_analysis={"weighted_score": 80},
            score=75
        )
        
        technical_signal = TechnicalSignal(
            signal="BUY",
            strength=0.7,
            reason="Positive momentum",
            indicators={"rsi": 60, "macd": 0.5}
        )
        
        technical_analysis = TechnicalAnalysis(
            symbol="AAPL",
            analysis_date=datetime.now(),
            overall_signal=technical_signal,
            trend_analysis={"weighted_score": 65},
            momentum_analysis={"weighted_score": 70},
            volume_analysis={"weighted_score": 60},
            support_resistance={"score": 55},
            pattern_analysis={"weighted_score": 60},
            score=65
        )
        
        state = AgentState(
            symbol="AAPL",
            stock_data=mock_stock_data,
            fundamental_analysis=fundamental_analysis,
            technical_analysis=technical_analysis
        )
        
        result = await self.agent._combine_analysis(state)
        
        # Check that messages were added
        assert len(result["state"].messages) > 0
        assert any("Combined score" in msg["content"] for msg in result["state"].messages)
        
    @pytest.mark.asyncio
    async def test_generate_recommendation_node(self):
        """Test the generate recommendation node."""
        mock_stock_data = self.create_mock_stock_data()
        
        # Create mock analyses (similar to above)
        from src.analysis.fundamental import FundamentalAnalysis, FundamentalSignal
        from src.analysis.technical import TechnicalAnalysis, TechnicalSignal
        
        fundamental_signal = FundamentalSignal(
            signal="BUY",
            strength=0.8,
            reason="Strong fundamentals",
            metrics={"score": 75}
        )
        
        fundamental_analysis = FundamentalAnalysis(
            symbol="AAPL",
            analysis_date=datetime.now(),
            overall_signal=fundamental_signal,
            valuation_analysis={"weighted_score": 70},
            profitability_analysis={"weighted_score": 80},
            financial_health_analysis={"weighted_score": 75},
            growth_analysis={"weighted_score": 65},
            efficiency_analysis={"weighted_score": 70},
            market_position_analysis={"weighted_score": 75},
            risk_analysis={"weighted_score": 80},
            score=75
        )
        
        technical_signal = TechnicalSignal(
            signal="BUY",
            strength=0.7,
            reason="Positive momentum",
            indicators={"rsi": 60, "macd": 0.5}
        )
        
        technical_analysis = TechnicalAnalysis(
            symbol="AAPL",
            analysis_date=datetime.now(),
            overall_signal=technical_signal,
            trend_analysis={"weighted_score": 65},
            momentum_analysis={"weighted_score": 70},
            volume_analysis={"weighted_score": 60},
            support_resistance={"score": 55},
            pattern_analysis={"weighted_score": 60},
            score=65
        )
        
        state = AgentState(
            symbol="AAPL",
            stock_data=mock_stock_data,
            fundamental_analysis=fundamental_analysis,
            technical_analysis=technical_analysis
        )
        
        # Mock the LLM response
        with patch.object(self.agent.llm, 'ainvoke') as mock_llm:
            mock_response = Mock()
            mock_response.content = "Based on the analysis, I recommend BUY with strong fundamentals and positive technical indicators."
            mock_llm.return_value = mock_response
            
            result = await self.agent._generate_recommendation(state)
            
            assert result["state"].recommendation is not None
            assert result["state"].recommendation.symbol == "AAPL"
            assert result["state"].recommendation.recommendation in ['BUY', 'SELL', 'HOLD']
            
    @pytest.mark.asyncio
    async def test_validate_recommendation_node(self):
        """Test the validate recommendation node."""
        recommendation = StockRecommendation(
            symbol="AAPL",
            recommendation="BUY",
            confidence=0.8,
            reasoning="Strong analysis",
            fundamental_score=75,
            technical_score=65,
            combined_score=71
        )
        
        state = AgentState(
            symbol="AAPL",
            recommendation=recommendation
        )
        
        result = await self.agent._validate_recommendation(state)
        
        assert result["state"].recommendation is not None
        assert result["state"].recommendation.recommendation in ['BUY', 'SELL', 'HOLD']
        assert 0.0 <= result["state"].recommendation.confidence <= 1.0
        
    @pytest.mark.asyncio
    async def test_prepare_analysis_summary(self):
        """Test the analysis summary preparation."""
        mock_stock_data = self.create_mock_stock_data()
        
        # Create a complete state with all analyses
        from src.analysis.fundamental import FundamentalAnalysis, FundamentalSignal
        from src.analysis.technical import TechnicalAnalysis, TechnicalSignal
        
        fundamental_signal = FundamentalSignal(
            signal="BUY",
            strength=0.8,
            reason="Strong fundamentals",
            metrics={"score": 75}
        )
        
        fundamental_analysis = FundamentalAnalysis(
            symbol="AAPL",
            analysis_date=datetime.now(),
            overall_signal=fundamental_signal,
            valuation_analysis={"weighted_score": 70},
            profitability_analysis={"weighted_score": 80},
            financial_health_analysis={"weighted_score": 75},
            growth_analysis={"weighted_score": 65},
            efficiency_analysis={"weighted_score": 70},
            market_position_analysis={"weighted_score": 75},
            risk_analysis={"weighted_score": 80},
            score=75
        )
        
        technical_signal = TechnicalSignal(
            signal="BUY",
            strength=0.7,
            reason="Positive momentum",
            indicators={"rsi": 60, "macd": 0.5}
        )
        
        technical_analysis = TechnicalAnalysis(
            symbol="AAPL",
            analysis_date=datetime.now(),
            overall_signal=technical_signal,
            trend_analysis={"weighted_score": 65},
            momentum_analysis={"weighted_score": 70},
            volume_analysis={"weighted_score": 60},
            support_resistance={"score": 55},
            pattern_analysis={"weighted_score": 60},
            score=65
        )
        
        state = AgentState(
            symbol="AAPL",
            stock_data=mock_stock_data,
            fundamental_analysis=fundamental_analysis,
            technical_analysis=technical_analysis
        )
        
        summary = self.agent._prepare_analysis_summary(state)
        
        # Check that summary contains key information
        assert "AAPL" in summary
        assert "FUNDAMENTAL ANALYSIS" in summary
        assert "TECHNICAL ANALYSIS" in summary
        assert "75.0/100" in summary  # Fundamental score
        assert "65.0/100" in summary  # Technical score
        assert "BUY" in summary
        
    @pytest.mark.asyncio
    async def test_parse_llm_response(self):
        """Test LLM response parsing."""
        mock_stock_data = self.create_mock_stock_data()
        
        # Create mock analyses
        from src.analysis.fundamental import FundamentalAnalysis, FundamentalSignal
        from src.analysis.technical import TechnicalAnalysis, TechnicalSignal
        
        fundamental_signal = FundamentalSignal(
            signal="BUY",
            strength=0.8,
            reason="Strong fundamentals",
            metrics={"score": 75}
        )
        
        fundamental_analysis = FundamentalAnalysis(
            symbol="AAPL",
            analysis_date=datetime.now(),
            overall_signal=fundamental_signal,
            valuation_analysis={"weighted_score": 70},
            profitability_analysis={"weighted_score": 80},
            financial_health_analysis={"weighted_score": 75},
            growth_analysis={"weighted_score": 65},
            efficiency_analysis={"weighted_score": 70},
            market_position_analysis={"weighted_score": 75},
            risk_analysis={"weighted_score": 80},
            score=75
        )
        
        technical_signal = TechnicalSignal(
            signal="BUY",
            strength=0.7,
            reason="Positive momentum",
            indicators={"rsi": 60, "macd": 0.5}
        )
        
        technical_analysis = TechnicalAnalysis(
            symbol="AAPL",
            analysis_date=datetime.now(),
            overall_signal=technical_signal,
            trend_analysis={"weighted_score": 65},
            momentum_analysis={"weighted_score": 70},
            volume_analysis={"weighted_score": 60},
            support_resistance={"score": 55},
            pattern_analysis={"weighted_score": 60},
            score=65
        )
        
        state = AgentState(
            symbol="AAPL",
            stock_data=mock_stock_data,
            fundamental_analysis=fundamental_analysis,
            technical_analysis=technical_analysis
        )
        
        # Test BUY response
        buy_response = "Based on the strong fundamentals and positive technical indicators, I recommend BUY for this stock."
        recommendation = self.agent._parse_llm_response(buy_response, state)
        
        assert recommendation.recommendation == "BUY"
        assert recommendation.symbol == "AAPL"
        assert recommendation.confidence > 0.0
        assert recommendation.fundamental_score == 75
        assert recommendation.technical_score == 65
        
        # Test SELL response
        sell_response = "Given the poor fundamentals and negative technical signals, I recommend SELL for this stock."
        recommendation = self.agent._parse_llm_response(sell_response, state)
        
        assert recommendation.recommendation == "SELL"
        
        # Test HOLD response
        hold_response = "The analysis shows mixed signals, so I recommend HOLD."
        recommendation = self.agent._parse_llm_response(hold_response, state)
        
        assert recommendation.recommendation == "HOLD"
        
    @pytest.mark.asyncio
    async def test_create_fallback_recommendation(self):
        """Test fallback recommendation creation."""
        mock_stock_data = self.create_mock_stock_data()
        
        # Create mock analyses
        from src.analysis.fundamental import FundamentalAnalysis, FundamentalSignal
        from src.analysis.technical import TechnicalAnalysis, TechnicalSignal
        
        fundamental_signal = FundamentalSignal(
            signal="BUY",
            strength=0.8,
            reason="Strong fundamentals",
            metrics={"score": 80}
        )
        
        fundamental_analysis = FundamentalAnalysis(
            symbol="AAPL",
            analysis_date=datetime.now(),
            overall_signal=fundamental_signal,
            valuation_analysis={"weighted_score": 80},
            profitability_analysis={"weighted_score": 85},
            financial_health_analysis={"weighted_score": 80},
            growth_analysis={"weighted_score": 75},
            efficiency_analysis={"weighted_score": 80},
            market_position_analysis={"weighted_score": 85},
            risk_analysis={"weighted_score": 90},
            score=80
        )
        
        technical_signal = TechnicalSignal(
            signal="BUY",
            strength=0.7,
            reason="Positive momentum",
            indicators={"rsi": 60, "macd": 0.5}
        )
        
        technical_analysis = TechnicalAnalysis(
            symbol="AAPL",
            analysis_date=datetime.now(),
            overall_signal=technical_signal,
            trend_analysis={"weighted_score": 70},
            momentum_analysis={"weighted_score": 75},
            volume_analysis={"weighted_score": 65},
            support_resistance={"score": 60},
            pattern_analysis={"weighted_score": 65},
            score=70
        )
        
        state = AgentState(
            symbol="AAPL",
            stock_data=mock_stock_data,
            fundamental_analysis=fundamental_analysis,
            technical_analysis=technical_analysis
        )
        
        recommendation = self.agent._create_fallback_recommendation(state)
        
        assert recommendation.recommendation == "BUY"  # High combined score should be BUY
        assert recommendation.symbol == "AAPL"
        assert recommendation.confidence > 0.0
        assert recommendation.reasoning is not None
        
    @pytest.mark.asyncio
    async def test_agent_close(self):
        """Test agent cleanup."""
        await self.agent.close()
        # Should not raise any exceptions 