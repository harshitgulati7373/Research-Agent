"""
Fundamental Analysis Module

This module provides comprehensive fundamental analysis capabilities including:
- Financial ratio calculations
- Growth analysis
- Valuation metrics
- Financial health assessment
- Investment recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging

from src.data_sources.base import StockData


logger = logging.getLogger(__name__)


@dataclass
class FundamentalSignal:
    """Signal from fundamental analysis."""
    signal: str  # 'BUY', 'SELL', 'HOLD'
    strength: float  # 0.0 to 1.0
    reason: str
    metrics: Dict[str, float]


@dataclass
class FundamentalAnalysis:
    """Complete fundamental analysis result."""
    symbol: str
    analysis_date: datetime
    overall_signal: FundamentalSignal
    valuation_analysis: Dict[str, Any]
    profitability_analysis: Dict[str, Any]
    financial_health_analysis: Dict[str, Any]
    growth_analysis: Dict[str, Any]
    efficiency_analysis: Dict[str, Any]
    market_position_analysis: Dict[str, Any]
    risk_analysis: Dict[str, Any]
    score: float  # Overall score 0-100


class FundamentalAnalyzer:
    """Comprehensive fundamental analysis engine."""
    
    def __init__(self):
        self.valuation_weights = {
            'pe_ratio': 0.25,
            'price_to_book': 0.20,
            'price_to_sales': 0.15,
            'peg_ratio': 0.20,
            'enterprise_to_ebitda': 0.20
        }
        
        self.profitability_weights = {
            'profit_margin': 0.30,
            'operating_margin': 0.25,
            'return_on_equity': 0.25,
            'return_on_assets': 0.20
        }
        
        self.financial_health_weights = {
            'current_ratio': 0.30,
            'debt_to_equity': 0.30,
            'quick_ratio': 0.20,
            'interest_coverage': 0.20
        }
        
        self.growth_weights = {
            'revenue_growth': 0.35,
            'earnings_growth': 0.35,
            'dividend_growth': 0.15,
            'book_value_growth': 0.15
        }
        
    async def analyze(self, stock_data: StockData) -> FundamentalAnalysis:
        """
        Perform comprehensive fundamental analysis.
        
        Args:
            stock_data: StockData object containing all relevant data
            
        Returns:
            FundamentalAnalysis object with complete analysis
        """
        if not stock_data.fundamental_data:
            raise ValueError(f"No fundamental data available for {stock_data.symbol}")
            
        # Perform individual analyses
        valuation_analysis = self._analyze_valuation(stock_data)
        profitability_analysis = self._analyze_profitability(stock_data)
        financial_health_analysis = self._analyze_financial_health(stock_data)
        growth_analysis = self._analyze_growth(stock_data)
        efficiency_analysis = self._analyze_efficiency(stock_data)
        market_position_analysis = self._analyze_market_position(stock_data)
        risk_analysis = self._analyze_risk(stock_data)
        
        # Calculate overall score and signal
        overall_score = self._calculate_overall_score(
            valuation_analysis, profitability_analysis, financial_health_analysis,
            growth_analysis, efficiency_analysis, market_position_analysis, risk_analysis
        )
        
        overall_signal = self._generate_overall_signal(overall_score, stock_data)
        
        return FundamentalAnalysis(
            symbol=stock_data.symbol,
            analysis_date=datetime.now(),
            overall_signal=overall_signal,
            valuation_analysis=valuation_analysis,
            profitability_analysis=profitability_analysis,
            financial_health_analysis=financial_health_analysis,
            growth_analysis=growth_analysis,
            efficiency_analysis=efficiency_analysis,
            market_position_analysis=market_position_analysis,
            risk_analysis=risk_analysis,
            score=overall_score
        )
        
    def _analyze_valuation(self, stock_data: StockData) -> Dict[str, Any]:
        """Analyze valuation metrics."""
        fundamental_data = stock_data.fundamental_data
        
        # Extract valuation metrics
        pe_ratio = fundamental_data.get('pe_ratio') or fundamental_data.get('trailingpe')
        price_to_book = fundamental_data.get('price_to_book') or fundamental_data.get('pricetobookratio')
        price_to_sales = fundamental_data.get('price_to_sales') or fundamental_data.get('pricetosalesratiottm')
        peg_ratio = fundamental_data.get('peg_ratio') or fundamental_data.get('pegratio')
        enterprise_to_ebitda = fundamental_data.get('enterprise_to_ebitda') or fundamental_data.get('evtoebitda')
        
        # Industry benchmarks (can be made configurable)
        industry_benchmarks = {
            'pe_ratio': 20.0,
            'price_to_book': 3.0,
            'price_to_sales': 2.5,
            'peg_ratio': 1.0,
            'enterprise_to_ebitda': 15.0
        }
        
        valuation_scores = {}
        
        # PE Ratio Analysis
        if pe_ratio and pe_ratio > 0:
            if pe_ratio < industry_benchmarks['pe_ratio'] * 0.7:
                valuation_scores['pe_ratio'] = 90  # Undervalued
            elif pe_ratio < industry_benchmarks['pe_ratio']:
                valuation_scores['pe_ratio'] = 70  # Fairly valued
            elif pe_ratio < industry_benchmarks['pe_ratio'] * 1.5:
                valuation_scores['pe_ratio'] = 50  # Slightly overvalued
            else:
                valuation_scores['pe_ratio'] = 20  # Overvalued
        else:
            valuation_scores['pe_ratio'] = 50  # Neutral for negative/missing PE
            
        # Price-to-Book Analysis
        if price_to_book and price_to_book > 0:
            if price_to_book < 1.0:
                valuation_scores['price_to_book'] = 90  # Trading below book value
            elif price_to_book < industry_benchmarks['price_to_book']:
                valuation_scores['price_to_book'] = 70  # Reasonable
            else:
                valuation_scores['price_to_book'] = 30  # Expensive
        else:
            valuation_scores['price_to_book'] = 50
            
        # Price-to-Sales Analysis
        if price_to_sales and price_to_sales > 0:
            if price_to_sales < 1.0:
                valuation_scores['price_to_sales'] = 80
            elif price_to_sales < industry_benchmarks['price_to_sales']:
                valuation_scores['price_to_sales'] = 60
            else:
                valuation_scores['price_to_sales'] = 30
        else:
            valuation_scores['price_to_sales'] = 50
            
        # PEG Ratio Analysis
        if peg_ratio and peg_ratio > 0:
            if peg_ratio < 1.0:
                valuation_scores['peg_ratio'] = 80  # Growth at reasonable price
            elif peg_ratio < 2.0:
                valuation_scores['peg_ratio'] = 60  # Acceptable
            else:
                valuation_scores['peg_ratio'] = 30  # Expensive growth
        else:
            valuation_scores['peg_ratio'] = 50
            
        # Enterprise Value to EBITDA Analysis
        if enterprise_to_ebitda and enterprise_to_ebitda > 0:
            if enterprise_to_ebitda < 10:
                valuation_scores['enterprise_to_ebitda'] = 80
            elif enterprise_to_ebitda < industry_benchmarks['enterprise_to_ebitda']:
                valuation_scores['enterprise_to_ebitda'] = 60
            else:
                valuation_scores['enterprise_to_ebitda'] = 30
        else:
            valuation_scores['enterprise_to_ebitda'] = 50
            
        # Calculate weighted valuation score
        weighted_score = sum(
            valuation_scores.get(metric, 50) * weight
            for metric, weight in self.valuation_weights.items()
        )
        
        return {
            'scores': valuation_scores,
            'weighted_score': weighted_score,
            'metrics': {
                'pe_ratio': pe_ratio,
                'price_to_book': price_to_book,
                'price_to_sales': price_to_sales,
                'peg_ratio': peg_ratio,
                'enterprise_to_ebitda': enterprise_to_ebitda
            },
            'benchmarks': industry_benchmarks,
            'signal': self._score_to_signal(weighted_score)
        }
        
    def _analyze_profitability(self, stock_data: StockData) -> Dict[str, Any]:
        """Analyze profitability metrics."""
        fundamental_data = stock_data.fundamental_data
        
        # Extract profitability metrics
        profit_margin = fundamental_data.get('profit_margin') or fundamental_data.get('profitmargin')
        operating_margin = fundamental_data.get('operating_margin') or fundamental_data.get('operatingmarginttm')
        return_on_equity = fundamental_data.get('return_on_equity') or fundamental_data.get('returnonequityttm')
        return_on_assets = fundamental_data.get('return_on_assets') or fundamental_data.get('returnonassetsttm')
        gross_margin = fundamental_data.get('gross_margin') or fundamental_data.get('grossmargin')
        
        # Convert percentages to decimals if needed
        metrics = [profit_margin, operating_margin, return_on_equity, return_on_assets, gross_margin]
        for i, metric in enumerate(metrics):
            if metric and metric > 1:
                metrics[i] = metric / 100
                
        profit_margin, operating_margin, return_on_equity, return_on_assets, gross_margin = metrics
        
        profitability_scores = {}
        
        # Profit Margin Analysis
        if profit_margin is not None:
            if profit_margin > 0.20:
                profitability_scores['profit_margin'] = 90
            elif profit_margin > 0.10:
                profitability_scores['profit_margin'] = 70
            elif profit_margin > 0.05:
                profitability_scores['profit_margin'] = 50
            else:
                profitability_scores['profit_margin'] = 20
        else:
            profitability_scores['profit_margin'] = 50
            
        # Operating Margin Analysis
        if operating_margin is not None:
            if operating_margin > 0.15:
                profitability_scores['operating_margin'] = 90
            elif operating_margin > 0.08:
                profitability_scores['operating_margin'] = 70
            elif operating_margin > 0.03:
                profitability_scores['operating_margin'] = 50
            else:
                profitability_scores['operating_margin'] = 20
        else:
            profitability_scores['operating_margin'] = 50
            
        # Return on Equity Analysis
        if return_on_equity is not None:
            if return_on_equity > 0.15:
                profitability_scores['return_on_equity'] = 90
            elif return_on_equity > 0.10:
                profitability_scores['return_on_equity'] = 70
            elif return_on_equity > 0.05:
                profitability_scores['return_on_equity'] = 50
            else:
                profitability_scores['return_on_equity'] = 20
        else:
            profitability_scores['return_on_equity'] = 50
            
        # Return on Assets Analysis
        if return_on_assets is not None:
            if return_on_assets > 0.10:
                profitability_scores['return_on_assets'] = 90
            elif return_on_assets > 0.05:
                profitability_scores['return_on_assets'] = 70
            elif return_on_assets > 0.02:
                profitability_scores['return_on_assets'] = 50
            else:
                profitability_scores['return_on_assets'] = 20
        else:
            profitability_scores['return_on_assets'] = 50
            
        # Calculate weighted profitability score
        weighted_score = sum(
            profitability_scores.get(metric, 50) * weight
            for metric, weight in self.profitability_weights.items()
        )
        
        return {
            'scores': profitability_scores,
            'weighted_score': weighted_score,
            'metrics': {
                'profit_margin': profit_margin,
                'operating_margin': operating_margin,
                'return_on_equity': return_on_equity,
                'return_on_assets': return_on_assets,
                'gross_margin': gross_margin
            },
            'signal': self._score_to_signal(weighted_score)
        }
        
    def _analyze_financial_health(self, stock_data: StockData) -> Dict[str, Any]:
        """Analyze financial health metrics."""
        fundamental_data = stock_data.fundamental_data
        
        # Extract financial health metrics
        current_ratio = fundamental_data.get('current_ratio') or fundamental_data.get('currentratio')
        debt_to_equity = fundamental_data.get('debt_to_equity') or fundamental_data.get('debttoequity')
        quick_ratio = fundamental_data.get('quick_ratio') or fundamental_data.get('quickratio')
        
        # Calculate interest coverage if possible
        interest_coverage = None
        if stock_data.financial_statements and 'income_statement' in stock_data.financial_statements:
            income_stmt = stock_data.financial_statements['income_statement']
            if not income_stmt.empty:
                try:
                    # Try to calculate interest coverage ratio
                    operating_income = income_stmt.get('operatingIncome', income_stmt.get('Operating Income'))
                    interest_expense = income_stmt.get('interestExpense', income_stmt.get('Interest Expense'))
                    
                    if operating_income is not None and interest_expense is not None:
                        latest_operating_income = operating_income.iloc[0] if hasattr(operating_income, 'iloc') else operating_income
                        latest_interest_expense = interest_expense.iloc[0] if hasattr(interest_expense, 'iloc') else interest_expense
                        
                        if latest_interest_expense and latest_interest_expense != 0:
                            interest_coverage = latest_operating_income / abs(latest_interest_expense)
                except Exception as e:
                    logger.warning(f"Could not calculate interest coverage: {str(e)}")
        
        financial_health_scores = {}
        
        # Current Ratio Analysis
        if current_ratio is not None:
            if current_ratio > 2.0:
                financial_health_scores['current_ratio'] = 90
            elif current_ratio > 1.5:
                financial_health_scores['current_ratio'] = 80
            elif current_ratio > 1.0:
                financial_health_scores['current_ratio'] = 60
            else:
                financial_health_scores['current_ratio'] = 20
        else:
            financial_health_scores['current_ratio'] = 50
            
        # Debt-to-Equity Analysis
        if debt_to_equity is not None:
            if debt_to_equity < 0.3:
                financial_health_scores['debt_to_equity'] = 90
            elif debt_to_equity < 0.6:
                financial_health_scores['debt_to_equity'] = 70
            elif debt_to_equity < 1.0:
                financial_health_scores['debt_to_equity'] = 50
            else:
                financial_health_scores['debt_to_equity'] = 20
        else:
            financial_health_scores['debt_to_equity'] = 50
            
        # Quick Ratio Analysis
        if quick_ratio is not None:
            if quick_ratio > 1.0:
                financial_health_scores['quick_ratio'] = 90
            elif quick_ratio > 0.7:
                financial_health_scores['quick_ratio'] = 70
            elif quick_ratio > 0.5:
                financial_health_scores['quick_ratio'] = 50
            else:
                financial_health_scores['quick_ratio'] = 20
        else:
            financial_health_scores['quick_ratio'] = 50
            
        # Interest Coverage Analysis
        if interest_coverage is not None:
            if interest_coverage > 5.0:
                financial_health_scores['interest_coverage'] = 90
            elif interest_coverage > 2.5:
                financial_health_scores['interest_coverage'] = 70
            elif interest_coverage > 1.5:
                financial_health_scores['interest_coverage'] = 50
            else:
                financial_health_scores['interest_coverage'] = 20
        else:
            financial_health_scores['interest_coverage'] = 50
            
        # Calculate weighted financial health score
        weighted_score = sum(
            financial_health_scores.get(metric, 50) * weight
            for metric, weight in self.financial_health_weights.items()
        )
        
        return {
            'scores': financial_health_scores,
            'weighted_score': weighted_score,
            'metrics': {
                'current_ratio': current_ratio,
                'debt_to_equity': debt_to_equity,
                'quick_ratio': quick_ratio,
                'interest_coverage': interest_coverage
            },
            'signal': self._score_to_signal(weighted_score)
        }
        
    def _analyze_growth(self, stock_data: StockData) -> Dict[str, Any]:
        """Analyze growth metrics."""
        fundamental_data = stock_data.fundamental_data
        
        # Extract growth metrics
        revenue_growth = fundamental_data.get('revenue_growth') or fundamental_data.get('quarterlyrevenuegrowthyoy')
        earnings_growth = fundamental_data.get('earnings_growth') or fundamental_data.get('quarterlyearningsgrowthyoy')
        
        # Calculate additional growth metrics from financial statements
        dividend_growth = None
        book_value_growth = None
        
        if stock_data.financial_statements:
            # Calculate book value growth if balance sheet is available
            if 'balance_sheet' in stock_data.financial_statements:
                balance_sheet = stock_data.financial_statements['balance_sheet']
                if not balance_sheet.empty and balance_sheet.shape[0] >= 2:
                    try:
                        # Get shareholders' equity for current and previous year
                        equity_col = None
                        for col in balance_sheet.columns:
                            # Convert column to string if it's not already
                            col_str = str(col).lower() if hasattr(col, 'lower') else str(col).lower()
                            if 'equity' in col_str or 'book' in col_str:
                                equity_col = col
                                break
                        
                        if equity_col:
                            current_equity = balance_sheet[equity_col].iloc[0]
                            previous_equity = balance_sheet[equity_col].iloc[1]
                            
                            if current_equity and previous_equity and previous_equity != 0:
                                book_value_growth = (current_equity - previous_equity) / previous_equity
                    except Exception as e:
                        logger.warning(f"Could not calculate book value growth: {str(e)}")
        
        # Convert percentages to decimals if needed
        if revenue_growth and revenue_growth > 1:
            revenue_growth = revenue_growth / 100
        if earnings_growth and earnings_growth > 1:
            earnings_growth = earnings_growth / 100
            
        growth_scores = {}
        
        # Revenue Growth Analysis
        if revenue_growth is not None:
            if revenue_growth > 0.20:
                growth_scores['revenue_growth'] = 90
            elif revenue_growth > 0.10:
                growth_scores['revenue_growth'] = 70
            elif revenue_growth > 0.05:
                growth_scores['revenue_growth'] = 60
            elif revenue_growth > 0:
                growth_scores['revenue_growth'] = 50
            else:
                growth_scores['revenue_growth'] = 20
        else:
            growth_scores['revenue_growth'] = 50
            
        # Earnings Growth Analysis
        if earnings_growth is not None:
            if earnings_growth > 0.15:
                growth_scores['earnings_growth'] = 90
            elif earnings_growth > 0.08:
                growth_scores['earnings_growth'] = 70
            elif earnings_growth > 0.03:
                growth_scores['earnings_growth'] = 60
            elif earnings_growth > 0:
                growth_scores['earnings_growth'] = 50
            else:
                growth_scores['earnings_growth'] = 20
        else:
            growth_scores['earnings_growth'] = 50
            
        # Dividend Growth Analysis
        if dividend_growth is not None:
            if dividend_growth > 0.10:
                growth_scores['dividend_growth'] = 90
            elif dividend_growth > 0.05:
                growth_scores['dividend_growth'] = 70
            elif dividend_growth > 0:
                growth_scores['dividend_growth'] = 60
            else:
                growth_scores['dividend_growth'] = 40
        else:
            growth_scores['dividend_growth'] = 50
            
        # Book Value Growth Analysis
        if book_value_growth is not None:
            if book_value_growth > 0.15:
                growth_scores['book_value_growth'] = 90
            elif book_value_growth > 0.08:
                growth_scores['book_value_growth'] = 70
            elif book_value_growth > 0.03:
                growth_scores['book_value_growth'] = 60
            elif book_value_growth > 0:
                growth_scores['book_value_growth'] = 50
            else:
                growth_scores['book_value_growth'] = 20
        else:
            growth_scores['book_value_growth'] = 50
            
        # Calculate weighted growth score
        weighted_score = sum(
            growth_scores.get(metric, 50) * weight
            for metric, weight in self.growth_weights.items()
        )
        
        return {
            'scores': growth_scores,
            'weighted_score': weighted_score,
            'metrics': {
                'revenue_growth': revenue_growth,
                'earnings_growth': earnings_growth,
                'dividend_growth': dividend_growth,
                'book_value_growth': book_value_growth
            },
            'signal': self._score_to_signal(weighted_score)
        }
        
    def _analyze_efficiency(self, stock_data: StockData) -> Dict[str, Any]:
        """Analyze operational efficiency metrics."""
        fundamental_data = stock_data.fundamental_data
        
        # Extract efficiency metrics
        asset_turnover = None
        inventory_turnover = None
        receivables_turnover = None
        
        # Calculate efficiency ratios from financial statements
        if stock_data.financial_statements:
            if ('income_statement' in stock_data.financial_statements and 
                'balance_sheet' in stock_data.financial_statements):
                
                income_stmt = stock_data.financial_statements['income_statement']
                balance_sheet = stock_data.financial_statements['balance_sheet']
                
                if not income_stmt.empty and not balance_sheet.empty:
                    try:
                        # Asset Turnover = Revenue / Average Total Assets
                        revenue = income_stmt.get('totalRevenue', income_stmt.get('Total Revenue'))
                        total_assets = balance_sheet.get('totalAssets', balance_sheet.get('Total Assets'))
                        
                        if revenue is not None and total_assets is not None:
                            latest_revenue = revenue.iloc[0] if hasattr(revenue, 'iloc') else revenue
                            latest_assets = total_assets.iloc[0] if hasattr(total_assets, 'iloc') else total_assets
                            
                            if latest_assets and latest_assets != 0:
                                asset_turnover = latest_revenue / latest_assets
                                
                    except Exception as e:
                        logger.warning(f"Could not calculate efficiency ratios: {str(e)}")
        
        efficiency_scores = {}
        
        # Asset Turnover Analysis
        if asset_turnover is not None:
            if asset_turnover > 1.0:
                efficiency_scores['asset_turnover'] = 90
            elif asset_turnover > 0.7:
                efficiency_scores['asset_turnover'] = 70
            elif asset_turnover > 0.5:
                efficiency_scores['asset_turnover'] = 50
            else:
                efficiency_scores['asset_turnover'] = 30
        else:
            efficiency_scores['asset_turnover'] = 50
            
        # Overall efficiency score
        efficiency_score = np.mean(list(efficiency_scores.values()))
        
        return {
            'scores': efficiency_scores,
            'weighted_score': efficiency_score,
            'metrics': {
                'asset_turnover': asset_turnover,
                'inventory_turnover': inventory_turnover,
                'receivables_turnover': receivables_turnover
            },
            'signal': self._score_to_signal(efficiency_score)
        }
        
    def _analyze_market_position(self, stock_data: StockData) -> Dict[str, Any]:
        """Analyze market position and competitive metrics."""
        fundamental_data = stock_data.fundamental_data
        
        # Extract market position metrics
        market_cap = fundamental_data.get('market_cap') or fundamental_data.get('marketcap')
        beta = fundamental_data.get('beta')
        analyst_recommendation = fundamental_data.get('analyst_recommendation') or fundamental_data.get('recommendationmean')
        analyst_count = fundamental_data.get('analyst_count') or fundamental_data.get('numberofanalystopinions')
        
        market_position_scores = {}
        
        # Market Cap Analysis
        if market_cap is not None:
            if market_cap > 50_000_000_000:  # Large cap
                market_position_scores['market_cap'] = 70
            elif market_cap > 10_000_000_000:  # Mid cap
                market_position_scores['market_cap'] = 60
            elif market_cap > 2_000_000_000:  # Small cap
                market_position_scores['market_cap'] = 50
            else:  # Micro cap
                market_position_scores['market_cap'] = 40
        else:
            market_position_scores['market_cap'] = 50
            
        # Beta Analysis
        if beta is not None:
            if 0.8 <= beta <= 1.2:
                market_position_scores['beta'] = 80  # Moderate volatility
            elif 0.5 <= beta <= 1.5:
                market_position_scores['beta'] = 60  # Acceptable volatility
            else:
                market_position_scores['beta'] = 40  # High volatility
        else:
            market_position_scores['beta'] = 50
            
        # Analyst Recommendation Analysis
        if analyst_recommendation is not None:
            if analyst_recommendation <= 2.0:  # Strong Buy to Buy
                market_position_scores['analyst_recommendation'] = 80
            elif analyst_recommendation <= 3.0:  # Hold
                market_position_scores['analyst_recommendation'] = 60
            else:  # Sell
                market_position_scores['analyst_recommendation'] = 30
        else:
            market_position_scores['analyst_recommendation'] = 50
            
        # Analyst Coverage Analysis
        if analyst_count is not None:
            if analyst_count >= 10:
                market_position_scores['analyst_coverage'] = 80
            elif analyst_count >= 5:
                market_position_scores['analyst_coverage'] = 60
            elif analyst_count >= 2:
                market_position_scores['analyst_coverage'] = 40
            else:
                market_position_scores['analyst_coverage'] = 20
        else:
            market_position_scores['analyst_coverage'] = 50
            
        # Overall market position score
        market_position_score = np.mean(list(market_position_scores.values()))
        
        return {
            'scores': market_position_scores,
            'weighted_score': market_position_score,
            'metrics': {
                'market_cap': market_cap,
                'beta': beta,
                'analyst_recommendation': analyst_recommendation,
                'analyst_count': analyst_count
            },
            'signal': self._score_to_signal(market_position_score)
        }
        
    def _analyze_risk(self, stock_data: StockData) -> Dict[str, Any]:
        """Analyze risk factors."""
        fundamental_data = stock_data.fundamental_data
        
        # Extract risk metrics
        beta = fundamental_data.get('beta')
        debt_to_equity = fundamental_data.get('debt_to_equity') or fundamental_data.get('debttoequity')
        short_ratio = fundamental_data.get('short_ratio') or fundamental_data.get('shortratio')
        short_percent = fundamental_data.get('short_percent') or fundamental_data.get('shortpercentoffloat')
        
        risk_scores = {}
        
        # Beta Risk Analysis
        if beta is not None:
            if beta < 0.5:
                risk_scores['beta_risk'] = 90  # Low risk
            elif beta < 1.0:
                risk_scores['beta_risk'] = 70  # Moderate risk
            elif beta < 1.5:
                risk_scores['beta_risk'] = 50  # High risk
            else:
                risk_scores['beta_risk'] = 30  # Very high risk
        else:
            risk_scores['beta_risk'] = 50
            
        # Debt Risk Analysis
        if debt_to_equity is not None:
            if debt_to_equity < 0.3:
                risk_scores['debt_risk'] = 90  # Low debt risk
            elif debt_to_equity < 0.6:
                risk_scores['debt_risk'] = 70  # Moderate debt risk
            elif debt_to_equity < 1.0:
                risk_scores['debt_risk'] = 50  # High debt risk
            else:
                risk_scores['debt_risk'] = 20  # Very high debt risk
        else:
            risk_scores['debt_risk'] = 50
            
        # Short Interest Risk Analysis
        if short_percent is not None:
            if short_percent < 0.05:
                risk_scores['short_risk'] = 80  # Low short interest
            elif short_percent < 0.10:
                risk_scores['short_risk'] = 60  # Moderate short interest
            elif short_percent < 0.20:
                risk_scores['short_risk'] = 40  # High short interest
            else:
                risk_scores['short_risk'] = 20  # Very high short interest
        else:
            risk_scores['short_risk'] = 50
            
        # Overall risk score (inverse of risk)
        risk_score = np.mean(list(risk_scores.values()))
        
        return {
            'scores': risk_scores,
            'weighted_score': risk_score,
            'metrics': {
                'beta': beta,
                'debt_to_equity': debt_to_equity,
                'short_ratio': short_ratio,
                'short_percent': short_percent
            },
            'signal': self._score_to_signal(risk_score)
        }
        
    def _calculate_overall_score(self, valuation_analysis: Dict[str, Any], 
                               profitability_analysis: Dict[str, Any],
                               financial_health_analysis: Dict[str, Any],
                               growth_analysis: Dict[str, Any],
                               efficiency_analysis: Dict[str, Any],
                               market_position_analysis: Dict[str, Any],
                               risk_analysis: Dict[str, Any]) -> float:
        """Calculate overall fundamental score."""
        
        # Weights for different analysis categories
        category_weights = {
            'valuation': 0.25,
            'profitability': 0.20,
            'financial_health': 0.15,
            'growth': 0.20,
            'efficiency': 0.05,
            'market_position': 0.10,
            'risk': 0.05
        }
        
        overall_score = (
            valuation_analysis['weighted_score'] * category_weights['valuation'] +
            profitability_analysis['weighted_score'] * category_weights['profitability'] +
            financial_health_analysis['weighted_score'] * category_weights['financial_health'] +
            growth_analysis['weighted_score'] * category_weights['growth'] +
            efficiency_analysis['weighted_score'] * category_weights['efficiency'] +
            market_position_analysis['weighted_score'] * category_weights['market_position'] +
            risk_analysis['weighted_score'] * category_weights['risk']
        )
        
        return overall_score
        
    def _generate_overall_signal(self, score: float, stock_data: StockData) -> FundamentalSignal:
        """Generate overall investment signal based on score."""
        
        if score >= 75:
            signal = 'BUY'
            strength = min(1.0, score / 100)
            reason = "Strong fundamentals with good value, profitability, and growth prospects"
        elif score >= 60:
            signal = 'BUY'
            strength = min(0.8, score / 100)
            reason = "Good fundamentals with decent value and growth potential"
        elif score >= 45:
            signal = 'HOLD'
            strength = min(0.6, score / 100)
            reason = "Mixed fundamentals with some concerns"
        elif score >= 30:
            signal = 'SELL'
            strength = min(0.4, score / 100)
            reason = "Poor fundamentals with significant risks"
        else:
            signal = 'SELL'
            strength = min(0.2, score / 100)
            reason = "Very poor fundamentals with high risk"
            
        return FundamentalSignal(
            signal=signal,
            strength=strength,
            reason=reason,
            metrics={'overall_score': score}
        )
        
    def _score_to_signal(self, score: float) -> str:
        """Convert numeric score to signal."""
        if score >= 70:
            return 'BUY'
        elif score >= 50:
            return 'HOLD'
        else:
            return 'SELL' 