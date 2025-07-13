"""
Yahoo Finance Data Source Implementation
"""

import yfinance as yf
import pandas as pd
from typing import Dict, Any, Optional
import asyncio
from datetime import datetime, timedelta
import logging

from .base import BaseDataSource


logger = logging.getLogger(__name__)


class YFinanceDataSource(BaseDataSource):
    """Yahoo Finance data source implementation."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        self.session = None
        
    async def get_price_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """
        Get historical price data from Yahoo Finance.
        
        Args:
            symbol: Stock symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Run yfinance in executor to avoid blocking
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, yf.Ticker, symbol)
            
            # Get historical data
            hist = await loop.run_in_executor(
                None, 
                ticker.history, 
                period
            )
            
            if hist.empty:
                logger.warning(f"No price data found for symbol: {symbol}")
                return pd.DataFrame()
                
            # Standardize column names
            hist.columns = [col.lower().replace(' ', '_') for col in hist.columns]
            
            # Add technical indicators
            hist['returns'] = hist['close'].pct_change()
            hist['volatility'] = hist['returns'].rolling(window=20).std()
            hist['volume_sma'] = hist['volume'].rolling(window=20).mean()
            
            return hist
            
        except Exception as e:
            logger.error(f"Error fetching price data for {symbol}: {str(e)}")
            return pd.DataFrame()
            
    async def get_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get fundamental data from Yahoo Finance.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary containing fundamental metrics
        """
        try:
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, yf.Ticker, symbol)
            
            # Get ticker info
            info = await loop.run_in_executor(None, lambda: ticker.info)
            
            if not info:
                logger.warning(f"No fundamental data found for symbol: {symbol}")
                return {}
                
            # Extract key fundamental metrics
            fundamental_data = {
                'market_cap': info.get('marketCap'),
                'enterprise_value': info.get('enterpriseValue'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'price_to_book': info.get('priceToBook'),
                'price_to_sales': info.get('priceToSalesTrailing12Months'),
                'enterprise_to_revenue': info.get('enterpriseToRevenue'),
                'enterprise_to_ebitda': info.get('enterpriseToEbitda'),
                'profit_margin': info.get('profitMargins'),
                'operating_margin': info.get('operatingMargins'),
                'return_on_assets': info.get('returnOnAssets'),
                'return_on_equity': info.get('returnOnEquity'),
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'current_ratio': info.get('currentRatio'),
                'debt_to_equity': info.get('debtToEquity'),
                'gross_margin': info.get('grossMargins'),
                'beta': info.get('beta'),
                'dividend_yield': info.get('dividendYield'),
                'payout_ratio': info.get('payoutRatio'),
                'book_value': info.get('bookValue'),
                'price_to_book': info.get('priceToBook'),
                'earnings_per_share': info.get('trailingEps'),
                'revenue_per_share': info.get('revenuePerShare'),
                'shares_outstanding': info.get('sharesOutstanding'),
                'float_shares': info.get('floatShares'),
                'insider_ownership': info.get('heldPercentInsiders'),
                'institutional_ownership': info.get('heldPercentInstitutions'),
                'short_ratio': info.get('shortRatio'),
                'short_percent': info.get('shortPercentOfFloat'),
                'analyst_target_price': info.get('targetMeanPrice'),
                'analyst_recommendation': info.get('recommendationMean'),
                'analyst_count': info.get('numberOfAnalystOpinions'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'country': info.get('country'),
                'full_time_employees': info.get('fullTimeEmployees'),
                'business_summary': info.get('longBusinessSummary'),
                'website': info.get('website'),
                'last_updated': datetime.now().isoformat()
            }
            
            # Remove None values
            fundamental_data = {k: v for k, v in fundamental_data.items() if v is not None}
            
            return fundamental_data
            
        except Exception as e:
            logger.error(f"Error fetching fundamental data for {symbol}: {str(e)}")
            return {}
            
    async def get_financial_statements(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Get financial statements from Yahoo Finance.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary containing financial statements
        """
        try:
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, yf.Ticker, symbol)
            
            # Get financial statements concurrently
            income_stmt, balance_sheet, cash_flow = await asyncio.gather(
                loop.run_in_executor(None, lambda: ticker.financials),
                loop.run_in_executor(None, lambda: ticker.balance_sheet),
                loop.run_in_executor(None, lambda: ticker.cashflow),
                return_exceptions=True
            )
            
            statements = {}
            
            if not isinstance(income_stmt, Exception) and not income_stmt.empty:
                statements['income_statement'] = income_stmt
                
            if not isinstance(balance_sheet, Exception) and not balance_sheet.empty:
                statements['balance_sheet'] = balance_sheet
                
            if not isinstance(cash_flow, Exception) and not cash_flow.empty:
                statements['cash_flow'] = cash_flow
                
            return statements
            
        except Exception as e:
            logger.error(f"Error fetching financial statements for {symbol}: {str(e)}")
            return {}
            
    async def get_news(self, symbol: str, limit: int = 10) -> list:
        """
        Get recent news for a stock.
        
        Args:
            symbol: Stock symbol
            limit: Number of news items to return
            
        Returns:
            List of news items
        """
        try:
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, yf.Ticker, symbol)
            
            news = await loop.run_in_executor(None, lambda: ticker.news)
            
            if not news:
                return []
                
            # Format news data
            formatted_news = []
            for item in news[:limit]:
                formatted_news.append({
                    'title': item.get('title', ''),
                    'publisher': item.get('publisher', ''),
                    'link': item.get('link', ''),
                    'published_date': item.get('providerPublishTime', ''),
                    'summary': item.get('summary', '')
                })
                
            return formatted_news
            
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {str(e)}")
            return [] 