"""
Research Agent using LangGraph

This agent combines fundamental and technical analysis to provide comprehensive 
stock investment recommendations. It uses LangGraph for structured decision-making
and LangChain for LLM interactions.
"""

import asyncio
import os
from typing import Dict, Any, List, Optional, Tuple, Annotated
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver

# LangSmith integration
from langsmith import Client as LangSmithClient
from langchain.globals import set_debug, set_verbose

from src.data_sources.factory import get_data_source_factory
from src.analysis.fundamental import FundamentalAnalyzer, FundamentalAnalysis
from src.analysis.technical import TechnicalAnalyzer, TechnicalAnalysis
from src.data_sources.base import StockData
from config.settings import get_settings
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class StockRecommendation:
    """Final stock recommendation."""
    symbol: str
    recommendation: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 to 1.0
    price_target: Optional[float] = None
    reasoning: str = ""
    fundamental_score: float = 0.0
    technical_score: float = 0.0
    combined_score: float = 0.0
    fundamental_signal: str = ""
    technical_signal: str = ""
    risk_level: str = "MEDIUM"  # LOW, MEDIUM, HIGH
    time_horizon: str = "MEDIUM_TERM"  # SHORT_TERM, MEDIUM_TERM, LONG_TERM
    key_factors: List[str] = None
    analysis_timestamp: datetime = None
    enriched_price_data: Optional[pd.DataFrame] = None  # Price data with technical indicators for charts
    # NEW: MOAT analysis fields
    moat_score: float = 0.0  # Economic moat score 0-100
    moat_strength: str = "No Moat"  # Wide Moat, Narrow Moat, or No Moat
    competitive_advantages: List[str] = None  # List of identified competitive advantages
    
    def __post_init__(self):
        if self.key_factors is None:
            self.key_factors = []
        if self.analysis_timestamp is None:
            self.analysis_timestamp = datetime.now()
        if self.competitive_advantages is None:
            self.competitive_advantages = []


@dataclass
class AgentState:
    """State for the stock analysis agent."""
    symbol: str
    stock_data: Optional[StockData] = None
    fundamental_analysis: Optional[FundamentalAnalysis] = None
    technical_analysis: Optional[TechnicalAnalysis] = None
    recommendation: Optional[StockRecommendation] = None
    errors: List[str] = None
    messages: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.messages is None:
            self.messages = []


# Global flag to track LangSmith initialization
_LANGSMITH_INITIALIZED = False

class StockAnalysisAgent:
    """
    Stock analysis agent using LangGraph for structured decision-making.
    
    This agent orchestrates the complete stock analysis workflow:
    1. Data fetching from multiple sources
    2. Fundamental analysis
    3. Technical analysis  
    4. LLM-powered recommendation generation
    5. Validation and final recommendation
    """
    
    def __init__(self, checkpoint_saver: Optional[BaseCheckpointSaver] = None):
        """Initialize the stock analysis agent."""
        self.settings = get_settings()
        
        # Initialize data sources
        self.data_factory = get_data_source_factory()
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        
        # Initialize checkpoint saver for persistence
        self.checkpoint_saver = checkpoint_saver or MemorySaver()
        
        # Setup LangSmith tracing
        self._setup_langsmith()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.settings.model_name,
            temperature=self.settings.model_temperature,
            api_key=self.settings.openai_api_key
        )
        
        # Build the analysis graph
        self.graph = self._build_graph()
        logger.info("✅ Stock Analysis Agent initialized successfully")

    def _setup_langsmith(self):
        """Setup LangSmith tracing if configured."""
        global _LANGSMITH_INITIALIZED
        
        try:
            if self.settings.langsmith_api_key and not _LANGSMITH_INITIALIZED:
                # Set environment variables for LangSmith
                os.environ["LANGSMITH_API_KEY"] = self.settings.langsmith_api_key
                os.environ["LANGSMITH_PROJECT"] = self.settings.langsmith_project
                os.environ["LANGSMITH_TRACING"] = "true"
                
                # Clear any existing endpoint to prevent conflicts
                if "LANGSMITH_ENDPOINT" in os.environ:
                    del os.environ["LANGSMITH_ENDPOINT"]
                
                # Initialize LangSmith client with error handling
                try:
                    self.langsmith_client = LangSmithClient()
                    _LANGSMITH_INITIALIZED = True
                    logger.info(f"✅ LangSmith tracing enabled for project: {self.settings.langsmith_project}")
                except Exception as client_error:
                    logger.warning(f"⚠️ LangSmith client initialization failed: {client_error}")
                    self.langsmith_client = None
                    # Disable tracing if client fails
                    os.environ["LANGSMITH_TRACING"] = "false"
                
                # Enable debug mode for development (only if client initialized successfully)
                if self.langsmith_client and self.settings.log_level == "DEBUG":
                    set_debug(True)
                    set_verbose(True)
            elif _LANGSMITH_INITIALIZED:
                logger.info("🔄 LangSmith already initialized globally, reusing configuration")
                self.langsmith_client = LangSmithClient() if self.settings.langsmith_api_key else None
            else:
                logger.info("📊 LangSmith not configured - tracing disabled")
                self.langsmith_client = None
                # Ensure tracing is disabled
                os.environ["LANGSMITH_TRACING"] = "false"
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to setup LangSmith: {e}")
            self.langsmith_client = None
            # Ensure tracing is disabled on any error
            os.environ["LANGSMITH_TRACING"] = "false"
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph analysis workflow."""
        
        # Define the workflow with state type
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("fetch_data", self._fetch_data)
        workflow.add_node("fundamental_analysis", self._run_fundamental_analysis)
        workflow.add_node("technical_analysis", self._run_technical_analysis)
        workflow.add_node("combine_analysis", self._combine_analysis)
        workflow.add_node("generate_recommendation", self._generate_recommendation)
        workflow.add_node("validate_recommendation", self._validate_recommendation)
        
        # Add edges - Make execution sequential to avoid concurrent state updates
        workflow.add_edge("fetch_data", "fundamental_analysis")
        workflow.add_edge("fundamental_analysis", "technical_analysis")
        workflow.add_edge("technical_analysis", "combine_analysis")
        workflow.add_edge("combine_analysis", "generate_recommendation")
        workflow.add_edge("generate_recommendation", "validate_recommendation")
        workflow.add_edge("validate_recommendation", END)
        
        # Set entry point
        workflow.set_entry_point("fetch_data")
        
        return workflow.compile()  # Disabled checkpointer for now
        
    async def analyze_stock(self, symbol: str, 
                          config: Optional[Dict[str, Any]] = None) -> StockRecommendation:
        """
        Analyze a stock and provide recommendation.
        
        Args:
            symbol: Stock symbol to analyze
            config: Optional configuration for the analysis
            
        Returns:
            StockRecommendation object with complete analysis
        """
        try:
            # Validate symbol
            if not symbol or not isinstance(symbol, str):
                raise ValueError("Invalid stock symbol")
                
            symbol = symbol.upper().strip()
            
            # Initialize state
            initial_state = AgentState(symbol=symbol)
            
            # Run the analysis graph
            final_state = await self.graph.ainvoke(initial_state)
            
            if final_state and final_state.get('recommendation'):
                return final_state['recommendation']
            else:
                raise ValueError("Failed to generate recommendation")
                
        except Exception as e:
            logger.error(f"Error analyzing stock {symbol}: {str(e)}")
            
            # Return a default recommendation with error
            return StockRecommendation(
                symbol=symbol,
                recommendation="HOLD",
                confidence=0.0,
                reasoning=f"Analysis failed: {str(e)}",
                risk_level="HIGH"
            )
            
    async def _fetch_data(self, state: AgentState) -> AgentState:
        """Fetch stock data from multiple sources."""
        try:
            logger.info(f"Fetching data for {state.symbol}")
            
            # Validate symbol first
            is_valid = await self.data_factory.validate_symbol(state.symbol)
            if not is_valid:
                state.errors.append(f"Invalid or non-existent symbol: {state.symbol}")
                return state
            
            # Fetch comprehensive stock data
            stock_data = await self.data_factory.get_stock_data(state.symbol)
            
            if (not stock_data or 
                stock_data.price_data is None or 
                stock_data.price_data.empty):
                state.errors.append(f"No price data available for {state.symbol}")
                return state
            
            state.stock_data = stock_data
            state.messages.append({
                "type": "info",
                "content": f"Successfully fetched data for {state.symbol}",
                "timestamp": datetime.now()
            })
            
            logger.info(f"Successfully fetched data for {state.symbol}")
            
        except Exception as e:
            error_msg = f"Error fetching data for {state.symbol}: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
            
        return state
        
    async def _run_fundamental_analysis(self, state: AgentState) -> AgentState:
        """Run fundamental analysis."""
        try:
            if not state.stock_data:
                state.errors.append("No stock data available for fundamental analysis")
                return state
                
            logger.info(f"Running fundamental analysis for {state.symbol}")
            
            # Run fundamental analysis
            fundamental_analysis = await self.fundamental_analyzer.analyze(state.stock_data)
            state.fundamental_analysis = fundamental_analysis
            
            state.messages.append({
                "type": "analysis",
                "content": f"Fundamental analysis complete. Score: {fundamental_analysis.score:.1f}",
                "timestamp": datetime.now()
            })
            
            logger.info(f"Fundamental analysis complete for {state.symbol}")
            
        except Exception as e:
            error_msg = f"Error in fundamental analysis for {state.symbol}: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
            
        return state
        
    async def _run_technical_analysis(self, state: AgentState) -> AgentState:
        """Run technical analysis."""
        try:
            if not state.stock_data:
                state.errors.append("No stock data available for technical analysis")
                return state
                
            logger.info(f"Running technical analysis for {state.symbol}")
            
            # Run technical analysis (this will also enrich the price data with indicators)
            technical_analysis = await self.technical_analyzer.analyze(state.stock_data)
            state.technical_analysis = technical_analysis
            
            state.messages.append({
                "type": "analysis",
                "content": f"Technical analysis complete. Score: {technical_analysis.score:.1f}",
                "timestamp": datetime.now()
            })
            
            logger.info(f"Technical analysis complete for {state.symbol}. Price data now includes technical indicators.")
            
        except Exception as e:
            error_msg = f"Error in technical analysis for {state.symbol}: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
            
        return state
        
    async def _combine_analysis(self, state: AgentState) -> AgentState:
        """Combine fundamental and technical analysis."""
        try:
            logger.info(f"Combining analyses for {state.symbol}")
            
            # Calculate combined score
            fundamental_score = state.fundamental_analysis.score if state.fundamental_analysis else 50.0
            technical_score = state.technical_analysis.score if state.technical_analysis else 50.0
            
            # Weighted combination (can be made configurable)
            fundamental_weight = 0.6  # Favor fundamental analysis
            technical_weight = 0.4
            
            combined_score = (
                fundamental_score * fundamental_weight +
                technical_score * technical_weight
            )
            
            state.messages.append({
                "type": "combination",
                "content": f"Combined score: {combined_score:.1f} (F: {fundamental_score:.1f}, T: {technical_score:.1f})",
                "timestamp": datetime.now()
            })
            
            logger.info(f"Combined analysis complete for {state.symbol}")
            
        except Exception as e:
            error_msg = f"Error combining analyses for {state.symbol}: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
            
        return state
        
    async def _generate_recommendation(self, state: AgentState) -> AgentState:
        """Generate investment recommendation using LLM."""
        try:
            logger.info(f"Generating recommendation for {state.symbol}")
            
            # Prepare analysis summary
            analysis_summary = self._prepare_analysis_summary(state)
            
            # Create prompt for LLM
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a professional stock analyst. Based on the provided 
                fundamental and technical analysis, provide a clear investment recommendation.
                
                Consider:
                - Investment time horizon (short, medium, long-term)
                - Risk level (low, medium, high)
                - Key factors driving the recommendation
                - Confidence level in the recommendation
                
                Be specific and actionable in your recommendations."""),
                ("human", "Analyze this stock and provide a recommendation:\n\n{analysis_summary}")
            ])
            
            # Try to get LLM response
            try:
                response = await self.llm.ainvoke(
                    prompt.format_messages(analysis_summary=analysis_summary)
                )
                
                # Parse the response and create recommendation
                recommendation = self._parse_llm_response(response.content, state)
                state.recommendation = recommendation
                
                state.messages.append({
                    "type": "recommendation",
                    "content": f"Recommendation: {recommendation.recommendation} (Confidence: {recommendation.confidence:.1f})",
                    "timestamp": datetime.now()
                })
                
                logger.info(f"Generated recommendation for {state.symbol}: {recommendation.recommendation}")
                
            except Exception as llm_error:
                logger.warning(f"LLM failed for {state.symbol}, using fallback: {str(llm_error)}")
                state.recommendation = self._create_fallback_recommendation(state)
                state.messages.append({
                    "type": "recommendation",
                    "content": f"Fallback recommendation: {state.recommendation.recommendation} (LLM failed)",
                    "timestamp": datetime.now()
                })
                
        except Exception as e:
            error_msg = f"Error generating recommendation for {state.symbol}: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
            
            # Create fallback recommendation
            state.recommendation = self._create_fallback_recommendation(state)
            
        return state
        
    async def _validate_recommendation(self, state: AgentState) -> AgentState:
        """Validate and finalize the recommendation."""
        try:
            logger.info(f"Validating recommendation for {state.symbol}")
            
            if not state.recommendation:
                state.recommendation = self._create_fallback_recommendation(state)
                
            # Ensure recommendation is valid
            if state.recommendation.recommendation not in ['BUY', 'SELL', 'HOLD']:
                state.recommendation.recommendation = 'HOLD'
                
            # Ensure confidence is within bounds
            state.recommendation.confidence = max(0.0, min(1.0, state.recommendation.confidence))
            
            # Add final validation message
            state.messages.append({
                "type": "validation",
                "content": f"Final recommendation validated: {state.recommendation.recommendation}",
                "timestamp": datetime.now()
            })
            
            logger.info(f"Recommendation validated for {state.symbol}")
            
        except Exception as e:
            error_msg = f"Error validating recommendation for {state.symbol}: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
            
        return state
        
    def _prepare_analysis_summary(self, state: AgentState) -> str:
        """Prepare a comprehensive analysis summary for the LLM."""
        
        summary_parts = [f"Stock Analysis for {state.symbol}"]
        
        # Stock data summary
        if state.stock_data:
            current_price = state.stock_data.price_data['close'].iloc[-1]
            summary_parts.append(f"Current Price: ${current_price:.2f}")
            
            if state.stock_data.fundamental_data:
                market_cap = state.stock_data.fundamental_data.get('market_cap')
                if market_cap:
                    summary_parts.append(f"Market Cap: ${market_cap:,.0f}")
                    
                sector = state.stock_data.fundamental_data.get('sector')
                if sector:
                    summary_parts.append(f"Sector: {sector}")
        
        # Fundamental analysis summary
        if state.fundamental_analysis:
            fa = state.fundamental_analysis
            summary_parts.append(f"\nFUNDAMENTAL ANALYSIS (Score: {fa.score:.1f}/100)")
            summary_parts.append(f"Signal: {fa.overall_signal.signal} (Strength: {fa.overall_signal.strength:.2f})")
            summary_parts.append(f"Reasoning: {fa.overall_signal.reason}")
            
            # Key metrics
            if fa.valuation_analysis:
                summary_parts.append(f"Valuation Score: {fa.valuation_analysis['weighted_score']:.1f}")
            if fa.profitability_analysis:
                summary_parts.append(f"Profitability Score: {fa.profitability_analysis['weighted_score']:.1f}")
            if fa.financial_health_analysis:
                summary_parts.append(f"Financial Health Score: {fa.financial_health_analysis['weighted_score']:.1f}")
            if fa.growth_analysis:
                summary_parts.append(f"Growth Score: {fa.growth_analysis['weighted_score']:.1f}")
        
        # Technical analysis summary
        if state.technical_analysis:
            ta = state.technical_analysis
            summary_parts.append(f"\nTECHNICAL ANALYSIS (Score: {ta.score:.1f}/100)")
            summary_parts.append(f"Signal: {ta.overall_signal.signal} (Strength: {ta.overall_signal.strength:.2f})")
            summary_parts.append(f"Reasoning: {ta.overall_signal.reason}")
            
            # Key indicators
            if ta.trend_analysis:
                summary_parts.append(f"Trend Score: {ta.trend_analysis['weighted_score']:.1f}")
            if ta.momentum_analysis:
                summary_parts.append(f"Momentum Score: {ta.momentum_analysis['weighted_score']:.1f}")
                
            # Current indicators
            if ta.overall_signal.indicators:
                rsi = ta.overall_signal.indicators.get('rsi')
                if rsi:
                    summary_parts.append(f"RSI: {rsi:.1f}")
                    
                macd = ta.overall_signal.indicators.get('macd')
                if macd:
                    summary_parts.append(f"MACD: {macd:.4f}")
        
        # Error summary
        if state.errors:
            summary_parts.append(f"\nERRORS/WARNINGS:")
            for error in state.errors:
                summary_parts.append(f"- {error}")
        
        return "\n".join(summary_parts)
        
    def _parse_llm_response(self, response: str, state: AgentState) -> StockRecommendation:
        """Parse LLM response and create structured recommendation."""
        
        # Extract recommendation signal
        recommendation = "HOLD"  # Default
        if "BUY" in response.upper():
            recommendation = "BUY"
        elif "SELL" in response.upper():
            recommendation = "SELL"
            
        # Calculate confidence based on analysis scores
        fundamental_score = state.fundamental_analysis.score if state.fundamental_analysis else 50.0
        technical_score = state.technical_analysis.score if state.technical_analysis else 50.0
        combined_score = (fundamental_score * 0.6) + (technical_score * 0.4)
        
        # Map score to confidence
        if combined_score >= 75:
            confidence = 0.9
        elif combined_score >= 60:
            confidence = 0.7
        elif combined_score >= 40:
            confidence = 0.5
        else:
            confidence = 0.3
            
        # Extract key factors
        key_factors = []
        if state.fundamental_analysis:
            if state.fundamental_analysis.overall_signal.signal == "BUY":
                key_factors.append("Strong fundamental metrics")
            elif state.fundamental_analysis.overall_signal.signal == "SELL":
                key_factors.append("Weak fundamental metrics")
                
        if state.technical_analysis:
            if state.technical_analysis.overall_signal.signal == "BUY":
                key_factors.append("Positive technical indicators")
            elif state.technical_analysis.overall_signal.signal == "SELL":
                key_factors.append("Negative technical indicators")
                
        # Determine risk level
        risk_level = "MEDIUM"
        if combined_score >= 70:
            risk_level = "LOW"
        elif combined_score <= 30:
            risk_level = "HIGH"
            
        # Extract MOAT data from fundamental analysis
        moat_score = 0.0
        moat_strength = "No Moat"
        competitive_advantages = []
        
        if state.fundamental_analysis and hasattr(state.fundamental_analysis, 'moat_analysis'):
            moat_analysis = state.fundamental_analysis.moat_analysis
            moat_score = moat_analysis.get('weighted_score', 0.0)
            moat_strength = moat_analysis.get('moat_strength', 'No Moat')
            
            # Identify competitive advantages based on scores
            scores = moat_analysis.get('scores', {})
            for advantage, score in scores.items():
                if score > 70:  # Strong advantage
                    if advantage == 'brand_strength':
                        competitive_advantages.append("Strong Brand Power")
                    elif advantage == 'cost_advantage':
                        competitive_advantages.append("Cost Leadership")
                    elif advantage == 'network_effects':
                        competitive_advantages.append("Network Effects")
                    elif advantage == 'switching_costs':
                        competitive_advantages.append("High Switching Costs")
                    elif advantage == 'regulatory_moat':
                        competitive_advantages.append("Regulatory Protection")
                    elif advantage == 'scale_economies':
                        competitive_advantages.append("Economies of Scale")
        
        # Create final recommendation
        recommendation = StockRecommendation(
            symbol=state.symbol,
            recommendation=recommendation,
            confidence=confidence,
            price_target=None, # LLM doesn't provide price target
            reasoning=response,
            fundamental_score=state.fundamental_analysis.score if state.fundamental_analysis else 0.0,
            technical_score=state.technical_analysis.score if state.technical_analysis else 0.0,
            combined_score=combined_score,
            fundamental_signal=state.fundamental_analysis.overall_signal.signal if state.fundamental_analysis else "HOLD",
            technical_signal=state.technical_analysis.overall_signal.signal if state.technical_analysis else "HOLD",
            risk_level=risk_level,
            time_horizon="MEDIUM_TERM", # Default
            key_factors=key_factors,
            enriched_price_data=state.stock_data.price_data if state.stock_data else None,  # Include enriched data
            # NEW: Include MOAT analysis data
            moat_score=moat_score,
            moat_strength=moat_strength,
            competitive_advantages=competitive_advantages
        )
        
        return recommendation
        
    def _create_fallback_recommendation(self, state: AgentState) -> StockRecommendation:
        """Create a fallback recommendation when LLM fails."""
        current_price = None
        
        if state.stock_data and state.stock_data.price_data is not None and not state.stock_data.price_data.empty:
            current_price = state.stock_data.price_data['close'].iloc[-1]
        
        # Conservative approach when analysis fails
        return StockRecommendation(
            symbol=state.symbol,
            recommendation="HOLD",
            confidence=0.0,
            reasoning="Analysis failed - insufficient data for recommendation. Consider manual review.",
            fundamental_score=state.fundamental_analysis.score if state.fundamental_analysis else 0.0,
            technical_score=state.technical_analysis.score if state.technical_analysis else 0.0,
            combined_score=0.0,
            fundamental_signal=state.fundamental_analysis.overall_signal.signal if state.fundamental_analysis else "HOLD",
            technical_signal=state.technical_analysis.overall_signal.signal if state.technical_analysis else "HOLD",
            risk_level="HIGH",
            key_factors=["Analysis failed", "Insufficient data", "Manual review recommended"],
            enriched_price_data=state.stock_data.price_data if state.stock_data else None,  # Include enriched data
            # NEW: Default MOAT values for fallback
            moat_score=0.0,
            moat_strength="No Moat",
            competitive_advantages=[]
        )
        
    async def get_analysis_history(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get historical analysis for a symbol."""
        try:
            # This would query the checkpoint database for historical analyses
            # For now, return empty list
            return []
        except Exception as e:
            logger.error(f"Error getting analysis history for {symbol}: {str(e)}")
            return []
            
    async def close(self):
        """Clean up agent resources."""
        global _LANGSMITH_INITIALIZED
        
        if self.data_factory:
            await self.data_factory.close()
        
        # Reset LangSmith global state to allow fresh initialization
        if _LANGSMITH_INITIALIZED:
            _LANGSMITH_INITIALIZED = False
            if "LANGSMITH_TRACING" in os.environ:
                os.environ["LANGSMITH_TRACING"] = "false"
            logger.info("🧹 LangSmith state cleaned up")
            
        logger.info("Agent closed successfully")


# Global agent instance
_agent_instance = None


def get_stock_agent() -> StockAnalysisAgent:
    """Get the global stock analysis agent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = StockAnalysisAgent()
    return _agent_instance


async def analyze_stock(symbol: str) -> StockRecommendation:
    """
    Convenience function to analyze a stock.
    
    Args:
        symbol: Stock symbol to analyze
        
    Returns:
        StockRecommendation object
    """
    agent = get_stock_agent()
    return await agent.analyze_stock(symbol) 