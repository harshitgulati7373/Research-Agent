"""
Data Source Factory for managing multiple financial data sources
"""

from typing import Dict, List, Optional, Union
import asyncio
import logging
from dataclasses import dataclass

from .base import BaseDataSource, StockData
from .yfinance_source import YFinanceDataSource
from .alpha_vantage_source import AlphaVantageDataSource
from config.settings import get_settings


logger = logging.getLogger(__name__)


@dataclass
class DataSourceConfig:
    """Configuration for data sources."""
    primary_source: str = "yfinance"
    fallback_sources: List[str] = None
    enable_alpha_vantage: bool = True
    enable_yfinance: bool = True
    
    def __post_init__(self):
        if self.fallback_sources is None:
            self.fallback_sources = ["alpha_vantage"]


class DataSourceFactory:
    """Factory for creating and managing data sources."""
    
    def __init__(self, config: Optional[DataSourceConfig] = None):
        self.config = config or DataSourceConfig()
        self.settings = get_settings()
        self._sources: Dict[str, BaseDataSource] = {}
        self._initialize_sources()
        
    def _initialize_sources(self):
        """Initialize available data sources."""
        
        # Always add YFinance (free)
        if self.config.enable_yfinance:
            self._sources["yfinance"] = YFinanceDataSource()
            
        # Add Alpha Vantage if API key is available
        if (self.config.enable_alpha_vantage and 
            self.settings.alpha_vantage_api_key):
            self._sources["alpha_vantage"] = AlphaVantageDataSource(
                self.settings.alpha_vantage_api_key
            )
        
        logger.info(f"Initialized data sources: {list(self._sources.keys())}")
        
    def get_source(self, source_name: str) -> Optional[BaseDataSource]:
        """Get a specific data source by name."""
        return self._sources.get(source_name)
        
    def get_primary_source(self) -> Optional[BaseDataSource]:
        """Get the primary data source."""
        return self._sources.get(self.config.primary_source)
        
    def get_available_sources(self) -> List[str]:
        """Get list of available data source names."""
        return list(self._sources.keys())
        
    async def get_stock_data(self, symbol: str, period: str = "1y") -> StockData:
        """
        Get comprehensive stock data using multiple sources.
        
        Args:
            symbol: Stock symbol
            period: Time period for price data
            
        Returns:
            StockData object with combined data from all sources
        """
        if not self._sources:
            raise ValueError("No data sources available")
            
        # Start with primary source
        primary_source = self.get_primary_source()
        if not primary_source:
            primary_source = next(iter(self._sources.values()))
            
        try:
            # Get data from primary source
            stock_data = await primary_source.get_complete_data(symbol, period)
            
            # Enhance with data from other sources
            await self._enhance_stock_data(stock_data, symbol, period)
            
            return stock_data
            
        except Exception as e:
            logger.error(f"Error getting stock data for {symbol}: {str(e)}")
            # Try fallback sources
            for fallback_name in self.config.fallback_sources:
                fallback_source = self._sources.get(fallback_name)
                if fallback_source:
                    try:
                        logger.info(f"Trying fallback source: {fallback_name}")
                        stock_data = await fallback_source.get_complete_data(symbol, period)
                        await self._enhance_stock_data(stock_data, symbol, period)
                        return stock_data
                    except Exception as fallback_error:
                        logger.error(f"Fallback source {fallback_name} failed: {str(fallback_error)}")
                        continue
                        
            # If all sources fail, return empty data
            return StockData(symbol=symbol, price_data=None)
            
    async def _enhance_stock_data(self, stock_data: StockData, symbol: str, period: str):
        """
        Enhance stock data with information from multiple sources.
        
        Args:
            stock_data: StockData object to enhance
            symbol: Stock symbol
            period: Time period
        """
        enhancement_tasks = []
        
        # Get fundamental data from Alpha Vantage if available
        if "alpha_vantage" in self._sources and stock_data.fundamental_data:
            enhancement_tasks.append(
                self._enhance_fundamental_data(stock_data, symbol)
            )
            
        # Get news from YFinance if available
        if "yfinance" in self._sources:
            enhancement_tasks.append(
                self._enhance_news_data(stock_data, symbol)
            )
            
        # Execute enhancements concurrently
        if enhancement_tasks:
            await asyncio.gather(*enhancement_tasks, return_exceptions=True)
            
    async def _enhance_fundamental_data(self, stock_data: StockData, symbol: str):
        """Enhance fundamental data with Alpha Vantage data."""
        try:
            av_source = self._sources.get("alpha_vantage")
            if av_source:
                av_fundamental = await av_source.get_fundamental_data(symbol)
                
                # Merge with existing fundamental data
                if stock_data.fundamental_data:
                    stock_data.fundamental_data.update(av_fundamental)
                else:
                    stock_data.fundamental_data = av_fundamental
                    
        except Exception as e:
            logger.error(f"Error enhancing fundamental data: {str(e)}")
            
    async def _enhance_news_data(self, stock_data: StockData, symbol: str):
        """Enhance with news data from YFinance."""
        try:
            yf_source = self._sources.get("yfinance")
            if yf_source and hasattr(yf_source, 'get_news'):
                news = await yf_source.get_news(symbol)
                stock_data.news = news
                
        except Exception as e:
            logger.error(f"Error enhancing news data: {str(e)}")
            
    async def validate_symbol(self, symbol: str) -> bool:
        """
        Validate a stock symbol using available sources.
        
        Args:
            symbol: Stock symbol to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not symbol:
            return False
            
        # Try primary source first
        primary_source = self.get_primary_source()
        if primary_source:
            try:
                # Try to get basic price data
                price_data = await primary_source.get_price_data(symbol, "1d")
                return not price_data.empty
            except Exception:
                pass
                
        # Try other sources
        for source_name, source in self._sources.items():
            if source_name != self.config.primary_source:
                try:
                    price_data = await source.get_price_data(symbol, "1d")
                    return not price_data.empty
                except Exception:
                    continue
                    
        return False
        
    async def close(self):
        """Close all data sources."""
        close_tasks = []
        for source in self._sources.values():
            if hasattr(source, 'close'):
                close_tasks.append(source.close())
                
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
            
    async def __aenter__(self):
        """Async context manager entry."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Global factory instance
_factory_instance = None


def get_data_source_factory() -> DataSourceFactory:
    """Get the global data source factory instance."""
    global _factory_instance
    if _factory_instance is None:
        _factory_instance = DataSourceFactory()
    return _factory_instance


async def get_stock_data(symbol: str, period: str = "1y") -> StockData:
    """
    Convenience function to get stock data using the global factory.
    
    Args:
        symbol: Stock symbol
        period: Time period
        
    Returns:
        StockData object
    """
    factory = get_data_source_factory()
    return await factory.get_stock_data(symbol, period) 