"""
Base Data Source Interface
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass


@dataclass
class StockData:
    """Data structure for stock information."""
    symbol: str
    price_data: pd.DataFrame
    fundamental_data: Optional[Dict[str, Any]] = None
    financial_statements: Optional[Dict[str, pd.DataFrame]] = None
    key_metrics: Optional[Dict[str, float]] = None
    analyst_ratings: Optional[Dict[str, Any]] = None
    news: Optional[List[Dict[str, Any]]] = None
    
    
class BaseDataSource(ABC):
    """Abstract base class for data sources."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.session = None
        
    @abstractmethod
    async def get_price_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """
        Get historical price data for a stock.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Time period (e.g., '1y', '6m', '3m')
            
        Returns:
            DataFrame with OHLCV data
        """
        pass
        
    @abstractmethod
    async def get_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get fundamental data for a stock.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary containing fundamental metrics
        """
        pass
        
    @abstractmethod
    async def get_financial_statements(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Get financial statements for a stock.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary containing income statement, balance sheet, cash flow
        """
        pass
        
    async def get_complete_data(self, symbol: str, period: str = "1y") -> StockData:
        """
        Get complete stock data combining all sources.
        
        Args:
            symbol: Stock symbol
            period: Time period for price data
            
        Returns:
            StockData object containing all available data
        """
        # Use asyncio.gather to fetch all data concurrently
        price_data, fundamental_data, financial_statements = await asyncio.gather(
            self.get_price_data(symbol, period),
            self.get_fundamental_data(symbol),
            self.get_financial_statements(symbol),
            return_exceptions=True
        )
        
        # Handle exceptions gracefully
        if isinstance(price_data, Exception):
            price_data = pd.DataFrame()
        if isinstance(fundamental_data, Exception):
            fundamental_data = {}
        if isinstance(financial_statements, Exception):
            financial_statements = {}
            
        return StockData(
            symbol=symbol,
            price_data=price_data,
            fundamental_data=fundamental_data,
            financial_statements=financial_statements
        )
        
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a stock symbol is valid.
        
        Args:
            symbol: Stock symbol to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not symbol or not isinstance(symbol, str):
            return False
        return symbol.replace(".", "").replace("-", "").isalnum()
        
    async def __aenter__(self):
        """Async context manager entry."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close() 