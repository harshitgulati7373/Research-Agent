"""
Unit tests for data sources module.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import asyncio

from src.data_sources.base import BaseDataSource, StockData
from src.data_sources.yfinance_source import YFinanceDataSource
from src.data_sources.alpha_vantage_source import AlphaVantageDataSource
from src.data_sources.factory import DataSourceFactory, DataSourceConfig


class TestBaseDataSource:
    """Test base data source functionality."""
    
    def test_validate_symbol_valid(self):
        """Test symbol validation with valid symbols."""
        source = Mock(spec=BaseDataSource)
        source.validate_symbol = BaseDataSource.validate_symbol.__get__(source)
        
        assert source.validate_symbol("AAPL") == True
        assert source.validate_symbol("GOOGL") == True
        assert source.validate_symbol("MSFT") == True
        assert source.validate_symbol("BRK.A") == True
        assert source.validate_symbol("BRK-A") == True
        
    def test_validate_symbol_invalid(self):
        """Test symbol validation with invalid symbols."""
        source = Mock(spec=BaseDataSource)
        source.validate_symbol = BaseDataSource.validate_symbol.__get__(source)
        
        assert source.validate_symbol("") == False
        assert source.validate_symbol(None) == False
        assert source.validate_symbol(123) == False
        assert source.validate_symbol("AAPL@") == False
        assert source.validate_symbol("AAPL!") == False
        
    def test_stock_data_creation(self):
        """Test StockData object creation."""
        price_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [99, 100, 101],
            'close': [104, 105, 106],
            'volume': [1000, 1100, 1200]
        })
        
        stock_data = StockData(
            symbol="AAPL",
            price_data=price_data,
            fundamental_data={"pe_ratio": 25.5},
            key_metrics={"market_cap": 2000000000}
        )
        
        assert stock_data.symbol == "AAPL"
        assert len(stock_data.price_data) == 3
        assert stock_data.fundamental_data["pe_ratio"] == 25.5
        assert stock_data.key_metrics["market_cap"] == 2000000000


class TestYFinanceDataSource:
    """Test YFinance data source."""
    
    def setup_method(self):
        """Set up test environment."""
        self.source = YFinanceDataSource()
        
    @pytest.mark.asyncio
    async def test_get_price_data_success(self):
        """Test successful price data retrieval."""
        with patch('yfinance.Ticker') as mock_ticker:
            # Mock ticker instance
            mock_ticker_instance = Mock()
            mock_ticker.return_value = mock_ticker_instance
            
            # Mock price data
            mock_price_data = pd.DataFrame({
                'Open': [100, 101, 102],
                'High': [105, 106, 107],
                'Low': [99, 100, 101],
                'Close': [104, 105, 106],
                'Volume': [1000, 1100, 1200]
            })
            mock_ticker_instance.history.return_value = mock_price_data
            
            # Test the method
            result = await self.source.get_price_data("AAPL", "1y")
            
            assert not result.empty
            assert len(result) == 3
            assert 'close' in result.columns
            assert 'volume' in result.columns
            assert 'returns' in result.columns
            
    @pytest.mark.asyncio
    async def test_get_price_data_empty(self):
        """Test price data retrieval with empty result."""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker.return_value = mock_ticker_instance
            mock_ticker_instance.history.return_value = pd.DataFrame()
            
            result = await self.source.get_price_data("INVALID", "1y")
            
            assert result.empty
            
    @pytest.mark.asyncio
    async def test_get_fundamental_data_success(self):
        """Test successful fundamental data retrieval."""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker.return_value = mock_ticker_instance
            
            mock_info = {
                'marketCap': 2000000000,
                'trailingPE': 25.5,
                'forwardPE': 24.0,
                'priceToBook': 5.2,
                'sector': 'Technology',
                'industry': 'Consumer Electronics'
            }
            mock_ticker_instance.info = mock_info
            
            result = await self.source.get_fundamental_data("AAPL")
            
            assert result['market_cap'] == 2000000000
            assert result['pe_ratio'] == 25.5
            assert result['price_to_book'] == 5.2
            assert result['sector'] == 'Technology'
            assert 'last_updated' in result
            
    @pytest.mark.asyncio
    async def test_get_fundamental_data_empty(self):
        """Test fundamental data retrieval with empty result."""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker.return_value = mock_ticker_instance
            mock_ticker_instance.info = {}
            
            result = await self.source.get_fundamental_data("INVALID")
            
            assert result == {}
            
    @pytest.mark.asyncio
    async def test_get_financial_statements_success(self):
        """Test successful financial statements retrieval."""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker.return_value = mock_ticker_instance
            
            mock_income_stmt = pd.DataFrame({
                'Total Revenue': [100000, 110000, 120000],
                'Net Income': [20000, 22000, 24000]
            })
            mock_balance_sheet = pd.DataFrame({
                'Total Assets': [200000, 220000, 240000],
                'Total Debt': [50000, 55000, 60000]
            })
            mock_cash_flow = pd.DataFrame({
                'Operating Cash Flow': [30000, 33000, 36000]
            })
            
            mock_ticker_instance.financials = mock_income_stmt
            mock_ticker_instance.balance_sheet = mock_balance_sheet
            mock_ticker_instance.cashflow = mock_cash_flow
            
            result = await self.source.get_financial_statements("AAPL")
            
            assert 'income_statement' in result
            assert 'balance_sheet' in result
            assert 'cash_flow' in result
            assert len(result['income_statement']) == 3
            
    @pytest.mark.asyncio
    async def test_get_news_success(self):
        """Test successful news retrieval."""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker.return_value = mock_ticker_instance
            
            mock_news = [
                {
                    'title': 'Apple reports strong earnings',
                    'publisher': 'Reuters',
                    'link': 'https://example.com/news1',
                    'providerPublishTime': 1234567890,
                    'summary': 'Apple had a great quarter'
                },
                {
                    'title': 'Apple launches new product',
                    'publisher': 'Bloomberg',
                    'link': 'https://example.com/news2',
                    'providerPublishTime': 1234567900,
                    'summary': 'New iPhone announced'
                }
            ]
            mock_ticker_instance.news = mock_news
            
            result = await self.source.get_news("AAPL", limit=2)
            
            assert len(result) == 2
            assert result[0]['title'] == 'Apple reports strong earnings'
            assert result[0]['publisher'] == 'Reuters'
            assert result[1]['title'] == 'Apple launches new product'


class TestAlphaVantageDataSource:
    """Test Alpha Vantage data source."""
    
    def setup_method(self):
        """Set up test environment."""
        self.source = AlphaVantageDataSource("test_api_key")
        
    @pytest.mark.asyncio
    async def test_make_request_success(self):
        """Test successful API request."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {"test": "data"}
            mock_response.raise_for_status.return_value = None
            
            mock_client_instance = Mock()
            mock_client_instance.get.return_value = mock_response
            mock_client.return_value = mock_client_instance
            
            result = await self.source._make_request({"function": "TIME_SERIES_DAILY"})
            
            assert result == {"test": "data"}
            
    @pytest.mark.asyncio
    async def test_make_request_error(self):
        """Test API request with error."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = {"Error Message": "Invalid API key"}
            mock_response.raise_for_status.return_value = None
            
            mock_client_instance = Mock()
            mock_client_instance.get.return_value = mock_response
            mock_client.return_value = mock_client_instance
            
            with pytest.raises(ValueError, match="Alpha Vantage API Error"):
                await self.source._make_request({"function": "TIME_SERIES_DAILY"})
                
    @pytest.mark.asyncio
    async def test_get_price_data_success(self):
        """Test successful price data retrieval."""
        mock_data = {
            "Time Series (Daily)": {
                "2023-01-01": {
                    "1. open": "100.00",
                    "2. high": "105.00",
                    "3. low": "99.00",
                    "4. close": "104.00",
                    "5. adjusted close": "104.00",
                    "6. volume": "1000",
                    "7. dividend amount": "0.00",
                    "8. split coefficient": "1.00"
                },
                "2023-01-02": {
                    "1. open": "104.00",
                    "2. high": "106.00",
                    "3. low": "103.00",
                    "4. close": "105.00",
                    "5. adjusted close": "105.00",
                    "6. volume": "1100",
                    "7. dividend amount": "0.00",
                    "8. split coefficient": "1.00"
                }
            }
        }
        
        with patch.object(self.source, '_make_request', return_value=mock_data):
            result = await self.source.get_price_data("AAPL", "1y")
            
            assert not result.empty
            assert len(result) == 2
            assert 'close' in result.columns
            assert 'volume' in result.columns
            assert result['close'].iloc[0] == 104.0
            assert result['volume'].iloc[0] == 1000.0
            
    @pytest.mark.asyncio
    async def test_get_fundamental_data_success(self):
        """Test successful fundamental data retrieval."""
        mock_data = {
            "Symbol": "AAPL",
            "MarketCapitalization": "2000000000",
            "PERatio": "25.5",
            "PriceToBookRatio": "5.2",
            "Sector": "Technology",
            "Industry": "Consumer Electronics"
        }
        
        with patch.object(self.source, '_make_request', return_value=mock_data):
            result = await self.source.get_fundamental_data("AAPL")
            
            assert result['symbol'] == "AAPL"
            assert result['marketcapitalization'] == 2000000000.0
            assert result['peratio'] == 25.5
            assert result['sector'] == "Technology"
            assert 'last_updated' in result


class TestDataSourceFactory:
    """Test data source factory."""
    
    def setup_method(self):
        """Set up test environment."""
        self.config = DataSourceConfig()
        self.factory = DataSourceFactory(self.config)
        
    def test_factory_initialization(self):
        """Test factory initialization."""
        assert self.factory.config is not None
        assert len(self.factory.get_available_sources()) > 0
        assert "yfinance" in self.factory.get_available_sources()
        
    def test_get_source(self):
        """Test getting a specific source."""
        yf_source = self.factory.get_source("yfinance")
        assert yf_source is not None
        assert isinstance(yf_source, YFinanceDataSource)
        
    def test_get_primary_source(self):
        """Test getting primary source."""
        primary = self.factory.get_primary_source()
        assert primary is not None
        
    @pytest.mark.asyncio
    async def test_validate_symbol_valid(self):
        """Test symbol validation with valid symbol."""
        with patch.object(self.factory.get_primary_source(), 'get_price_data') as mock_get_price:
            mock_get_price.return_value = pd.DataFrame({'close': [100, 101, 102]})
            
            result = await self.factory.validate_symbol("AAPL")
            assert result == True
            
    @pytest.mark.asyncio
    async def test_validate_symbol_invalid(self):
        """Test symbol validation with invalid symbol."""
        with patch.object(self.factory.get_primary_source(), 'get_price_data') as mock_get_price:
            mock_get_price.return_value = pd.DataFrame()
            
            result = await self.factory.validate_symbol("INVALID")
            assert result == False
            
    @pytest.mark.asyncio
    async def test_get_stock_data_success(self):
        """Test successful stock data retrieval."""
        mock_price_data = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        })
        
        mock_fundamental_data = {
            'pe_ratio': 25.5,
            'market_cap': 2000000000
        }
        
        with patch.object(self.factory.get_primary_source(), 'get_complete_data') as mock_get_complete:
            mock_stock_data = StockData(
                symbol="AAPL",
                price_data=mock_price_data,
                fundamental_data=mock_fundamental_data
            )
            mock_get_complete.return_value = mock_stock_data
            
            result = await self.factory.get_stock_data("AAPL", "1y")
            
            assert result.symbol == "AAPL"
            assert not result.price_data.empty
            assert result.fundamental_data['pe_ratio'] == 25.5
            
    @pytest.mark.asyncio
    async def test_close(self):
        """Test factory cleanup."""
        await self.factory.close()
        # Should not raise any exceptions 