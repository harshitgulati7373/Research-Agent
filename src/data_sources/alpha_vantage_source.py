"""
Alpha Vantage Data Source Implementation
"""

import httpx
import pandas as pd
from typing import Dict, Any, Optional
import asyncio
import logging
from datetime import datetime
import time

from .base import BaseDataSource


logger = logging.getLogger(__name__)


class AlphaVantageDataSource(BaseDataSource):
    """Alpha Vantage data source implementation."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://www.alphavantage.co/query"
        self.session = None
        self.rate_limit_delay = 12  # seconds between requests (free tier: 5 requests/minute)
        self.last_request_time = 0
        
    async def _ensure_session(self):
        """Ensure we have a valid session."""
        if self.session is None or self.session.is_closed:
            try:
                # Close old session if it exists
                if self.session and not self.session.is_closed:
                    await self.session.aclose()
            except Exception as e:
                logger.warning(f"Error closing old session: {str(e)}")
                
            # Create new session
            self.session = httpx.AsyncClient(
                timeout=30.0,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )
        
    async def _make_request(self, params: Dict[str, str]) -> Dict[str, Any]:
        """Make rate-limited request to Alpha Vantage API."""
        # Rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
            
        params['apikey'] = self.api_key
        
        await self._ensure_session()
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Add timeout to prevent hanging
                response = await asyncio.wait_for(
                    self.session.get(self.base_url, params=params),
                    timeout=20.0  # 20 second timeout
                )
                response.raise_for_status()
                self.last_request_time = time.time()
                
                data = response.json()
                
                # Check for API errors
                if 'Error Message' in data:
                    raise ValueError(f"Alpha Vantage API Error: {data['Error Message']}")
                if 'Note' in data:
                    raise ValueError(f"Alpha Vantage API Note: {data['Note']}")
                    
                return data
                
            except asyncio.TimeoutError:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
            except httpx.HTTPError as e:
                logger.warning(f"HTTP error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                logger.error(f"Error in Alpha Vantage request: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
    async def get_price_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """
        Get historical price data from Alpha Vantage.
        
        Args:
            symbol: Stock symbol
            period: Time period (not directly used, gets daily data)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': symbol,
                'outputsize': 'full'
            }
            
            data = await self._make_request(params)
            
            if 'Time Series (Daily)' not in data:
                logger.warning(f"No price data found for symbol: {symbol}")
                return pd.DataFrame()
                
            # Convert to DataFrame
            time_series = data['Time Series (Daily)']
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Standardize column names
            df.columns = [
                'open', 'high', 'low', 'close', 'adjusted_close', 
                'volume', 'dividend_amount', 'split_coefficient'
            ]
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # Add technical indicators
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std()
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            
            # Filter by period if needed
            if period == '5y':
                df = df.last('1825D')  # 5 years = 5 * 365 = 1825 days
            elif period == '1y':
                df = df.last('365D')
            elif period == '6m':
                df = df.last('180D')
            elif period == '3m':
                df = df.last('90D')
            elif period == '1m':
                df = df.last('30D')
                
            return df
            
        except Exception as e:
            logger.error(f"Error fetching price data for {symbol}: {str(e)}")
            return pd.DataFrame()
            
    async def get_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get fundamental data from Alpha Vantage.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary containing fundamental metrics
        """
        try:
            params = {
                'function': 'OVERVIEW',
                'symbol': symbol
            }
            
            data = await self._make_request(params)
            
            if not data or 'Symbol' not in data:
                logger.warning(f"No fundamental data found for symbol: {symbol}")
                return {}
                
            # Convert numeric fields
            numeric_fields = [
                'MarketCapitalization', 'EBITDA', 'PERatio', 'PEGRatio', 
                'BookValue', 'DividendPerShare', 'DividendYield', 'EPS',
                'RevenuePerShareTTM', 'ProfitMargin', 'OperatingMarginTTM',
                'ReturnOnAssetsTTM', 'ReturnOnEquityTTM', 'RevenueTTM',
                'GrossProfitTTM', 'DilutedEPSTTM', 'QuarterlyEarningsGrowthYOY',
                'QuarterlyRevenueGrowthYOY', 'AnalystTargetPrice', 'TrailingPE',
                'ForwardPE', 'PriceToSalesRatioTTM', 'PriceToBookRatio',
                'EVToRevenue', 'EVToEBITDA', 'Beta', 'SharesOutstanding',
                'DividendDate', 'ExDividendDate'
            ]
            
            fundamental_data = {}
            for key, value in data.items():
                if key in numeric_fields:
                    try:
                        # Handle 'None' and '-' values
                        if value in ['None', '-', '']:
                            fundamental_data[key.lower()] = None
                        else:
                            fundamental_data[key.lower()] = float(value)
                    except (ValueError, TypeError):
                        fundamental_data[key.lower()] = None
                else:
                    fundamental_data[key.lower()] = value
                    
            # Add timestamp
            fundamental_data['last_updated'] = datetime.now().isoformat()
            
            return fundamental_data
            
        except Exception as e:
            logger.error(f"Error fetching fundamental data for {symbol}: {str(e)}")
            return {}
            
    async def get_financial_statements(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Get financial statements from Alpha Vantage.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary containing financial statements
        """
        try:
            # Get income statement, balance sheet, and cash flow concurrently
            income_params = {'function': 'INCOME_STATEMENT', 'symbol': symbol}
            balance_params = {'function': 'BALANCE_SHEET', 'symbol': symbol}
            cashflow_params = {'function': 'CASH_FLOW', 'symbol': symbol}
            
            # Make requests with delays to respect rate limits
            statements = {}
            
            # Income Statement
            try:
                income_data = await self._make_request(income_params)
                if 'annualReports' in income_data:
                    df = pd.DataFrame(income_data['annualReports'])
                    if not df.empty:
                        df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
                        df.set_index('fiscalDateEnding', inplace=True)
                        statements['income_statement'] = df
            except Exception as e:
                logger.error(f"Error fetching income statement: {str(e)}")
                
            # Balance Sheet
            try:
                balance_data = await self._make_request(balance_params)
                if 'annualReports' in balance_data:
                    df = pd.DataFrame(balance_data['annualReports'])
                    if not df.empty:
                        df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
                        df.set_index('fiscalDateEnding', inplace=True)
                        statements['balance_sheet'] = df
            except Exception as e:
                logger.error(f"Error fetching balance sheet: {str(e)}")
                
            # Cash Flow
            try:
                cashflow_data = await self._make_request(cashflow_params)
                if 'annualReports' in cashflow_data:
                    df = pd.DataFrame(cashflow_data['annualReports'])
                    if not df.empty:
                        df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
                        df.set_index('fiscalDateEnding', inplace=True)
                        statements['cash_flow'] = df
            except Exception as e:
                logger.error(f"Error fetching cash flow: {str(e)}")
                
            return statements
            
        except Exception as e:
            logger.error(f"Error fetching financial statements for {symbol}: {str(e)}")
            return {}
            
    async def get_earnings_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get earnings data from Alpha Vantage.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary containing earnings data
        """
        try:
            params = {
                'function': 'EARNINGS',
                'symbol': symbol
            }
            
            data = await self._make_request(params)
            
            if not data:
                return {}
                
            earnings_data = {}
            
            # Annual earnings
            if 'annualEarnings' in data:
                annual_df = pd.DataFrame(data['annualEarnings'])
                if not annual_df.empty:
                    annual_df['fiscalDateEnding'] = pd.to_datetime(annual_df['fiscalDateEnding'])
                    earnings_data['annual_earnings'] = annual_df
                    
            # Quarterly earnings
            if 'quarterlyEarnings' in data:
                quarterly_df = pd.DataFrame(data['quarterlyEarnings'])
                if not quarterly_df.empty:
                    quarterly_df['fiscalDateEnding'] = pd.to_datetime(quarterly_df['fiscalDateEnding'])
                    earnings_data['quarterly_earnings'] = quarterly_df
                    
            return earnings_data
            
        except Exception as e:
            logger.error(f"Error fetching earnings data for {symbol}: {str(e)}")
            return {}
            
    async def close(self):
        """Close the HTTP session."""
        if self.session:
            try:
                await self.session.aclose()
            except Exception as e:
                logger.warning(f"Error closing Alpha Vantage session: {str(e)}")
            finally:
                self.session = None
                
    async def __aenter__(self):
        """Async context manager entry."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close() 