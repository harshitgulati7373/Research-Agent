"""
Research Agent - Streamlit Application

This application provides a user-friendly interface for stock analysis using
both fundamental and technical analysis to generate investment recommendations.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Research Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    
    .buy-signal {
        background: #d4edda;
        border-left-color: #28a745;
    }
    
    .sell-signal {
        background: #f8d7da;
        border-left-color: #dc3545;
    }
    
    .hold-signal {
        background: #fff3cd;
        border-left-color: #ffc107;
    }
    
    .analysis-section {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .score-bar {
        height: 20px;
        border-radius: 10px;
        background: linear-gradient(90deg, #dc3545 0%, #ffc107 50%, #28a745 100%);
        position: relative;
        margin: 0.5rem 0;
    }
    
    .score-indicator {
        position: absolute;
        top: -5px;
        width: 4px;
        height: 30px;
        background: #000;
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Import the agent (with proper error handling)
try:
    from src.agent.stock_agent import get_stock_agent, StockRecommendation
    from src.data_sources.factory import get_data_source_factory
    from config.settings import get_settings
    
    # Initialize settings
    settings = get_settings()
    
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()

# Initialize session state
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}
if "current_symbol" not in st.session_state:
    st.session_state.current_symbol = ""
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []


def create_score_visualization(score: float, title: str) -> str:
    """Create a visual score indicator."""
    color = "#dc3545" if score < 40 else "#ffc107" if score < 60 else "#28a745"
    return f"""
    <div class="metric-card">
        <h4>{title}</h4>
        <div class="score-bar">
            <div class="score-indicator" style="left: {score}%;"></div>
        </div>
        <p><strong>{score:.1f}/100</strong></p>
    </div>
    """


def create_recommendation_card(recommendation: StockRecommendation) -> str:
    """Create a recommendation card with styling."""
    signal_class = f"{recommendation.recommendation.lower()}-signal"
    
    confidence_color = "#28a745" if recommendation.confidence > 0.7 else "#ffc107" if recommendation.confidence > 0.4 else "#dc3545"
    
    return f"""
    <div class="metric-card {signal_class}">
        <h2>📊 {recommendation.recommendation}</h2>
        <p><strong>Confidence:</strong> 
            <span style="color: {confidence_color};">
                {recommendation.confidence:.1%}
            </span>
        </p>
        <p><strong>Risk Level:</strong> {recommendation.risk_level}</p>
        <p><strong>Time Horizon:</strong> {recommendation.time_horizon}</p>
        <hr>
        <p><strong>Key Reasoning:</strong></p>
        <p>{recommendation.reasoning}</p>
    </div>
    """


def create_price_chart(price_data: pd.DataFrame, symbol: str) -> go.Figure:
    """Create an interactive price chart with technical indicators."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{symbol} Price and Moving Averages', 'Volume'),
        row_width=[0.7, 0.3]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=price_data.index,
            open=price_data['open'],
            high=price_data['high'],
            low=price_data['low'],
            close=price_data['close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Moving averages
    if 'sma_20' in price_data.columns:
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data['sma_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
    
    if 'sma_50' in price_data.columns:
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data['sma_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
    
    if 'sma_200' in price_data.columns:
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data['sma_200'],
                mode='lines',
                name='SMA 200',
                line=dict(color='red', width=1)
            ),
            row=1, col=1
        )
    
    # Volume
    colors = ['red' if row['close'] < row['open'] else 'green' for idx, row in price_data.iterrows()]
    fig.add_trace(
        go.Bar(
            x=price_data.index,
            y=price_data['volume'],
            name='Volume',
            marker_color=colors
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f'{symbol} Stock Analysis',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_white',
        height=600,
        showlegend=True
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    
    return fig


def create_technical_indicators_chart(price_data: pd.DataFrame, symbol: str) -> go.Figure:
    """Create technical indicators chart."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('RSI', 'MACD', 'Bollinger Bands', 'Volume'),
        vertical_spacing=0.08,
        horizontal_spacing=0.08
    )
    
    # RSI
    if 'rsi' in price_data.columns:
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data['rsi'],
                mode='lines',
                name='RSI',
                line=dict(color='purple')
            ),
            row=1, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
    
    # MACD
    if 'macd' in price_data.columns and 'macd_signal' in price_data.columns:
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data['macd'],
                mode='lines',
                name='MACD',
                line=dict(color='blue')
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data['macd_signal'],
                mode='lines',
                name='Signal',
                line=dict(color='red')
            ),
            row=1, col=2
        )
    
    # Bollinger Bands
    if all(col in price_data.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data['bb_upper'],
                mode='lines',
                name='BB Upper',
                line=dict(color='red', dash='dash')
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data['bb_middle'],
                mode='lines',
                name='BB Middle',
                line=dict(color='blue')
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data['bb_lower'],
                mode='lines',
                name='BB Lower',
                line=dict(color='green', dash='dash')
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='black')
            ),
            row=2, col=1
        )
    
    # Volume
    fig.add_trace(
        go.Bar(
            x=price_data.index,
            y=price_data['volume'],
            name='Volume',
            marker_color='lightblue'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title=f'{symbol} Technical Indicators',
        template='plotly_white',
        height=600,
        showlegend=False
    )
    
    return fig


async def run_analysis(symbol: str) -> Optional[StockRecommendation]:
    """Run stock analysis asynchronously."""
    try:
        agent = get_stock_agent()
        
        with st.spinner(f"Analyzing {symbol}..."):
            recommendation = await agent.analyze_stock(symbol)
            
        return recommendation
        
    except Exception as e:
        st.error(f"Error analyzing {symbol}: {str(e)}")
        return None


def display_fundamental_analysis(recommendation: StockRecommendation):
    """Display fundamental analysis results."""
    st.subheader("📊 Fundamental Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            create_score_visualization(
                recommendation.fundamental_score, 
                "Overall Fundamental Score"
            ), 
            unsafe_allow_html=True
        )
    
    with col2:
        st.metric(
            label="Fundamental Signal",
            value=recommendation.fundamental_signal,
            delta=f"Score: {recommendation.fundamental_score:.1f}/100"
        )
    
    with col3:
        st.metric(
            label="Combined Score",
            value=f"{recommendation.combined_score:.1f}/100",
            delta=f"Weight: 60% Fundamental"
        )


def display_technical_analysis(recommendation: StockRecommendation):
    """Display technical analysis results."""
    st.subheader("📈 Technical Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            create_score_visualization(
                recommendation.technical_score, 
                "Overall Technical Score"
            ), 
            unsafe_allow_html=True
        )
    
    with col2:
        st.metric(
            label="Technical Signal",
            value=recommendation.technical_signal,
            delta=f"Score: {recommendation.technical_score:.1f}/100"
        )
    
    with col3:
        st.metric(
            label="Combined Score",
            value=f"{recommendation.combined_score:.1f}/100",
            delta=f"Weight: 40% Technical"
        )


def display_key_metrics(recommendation: StockRecommendation):
    """Display key metrics and factors."""
    st.subheader("🔍 Key Factors")
    
    if recommendation.key_factors:
        for factor in recommendation.key_factors:
            st.write(f"• {factor}")
    else:
        st.write("No specific key factors identified.")
    
    # Display analysis timestamp
    st.caption(f"Analysis completed at: {recommendation.analysis_timestamp}")


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Research Agent</h1>', unsafe_allow_html=True)
    st.markdown("**Powered by AI-driven Fundamental & Technical Analysis**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Stock symbol input
        symbol = st.text_input(
            "Enter Stock Symbol",
            value="AAPL",
            placeholder="e.g., AAPL, GOOGL, MSFT"
        ).upper()
        
        # Analysis period
        period = st.selectbox(
            "Analysis Period",
            ["1y", "6mo", "3mo", "1mo"],
            index=0
        )
        
        # Analysis button
        analyze_button = st.button("🔍 Analyze Stock", type="primary")
        
        st.divider()
        
        # Settings
        st.header("⚙️ Settings")
        show_charts = st.checkbox("Show Charts", value=True)
        show_detailed_analysis = st.checkbox("Show Detailed Analysis", value=True)
        
        st.divider()
        
        # Information
        st.header("ℹ️ Information")
        st.write("""
        This application provides comprehensive stock analysis using:
        - **Fundamental Analysis**: Financial metrics, ratios, and company health
        - **Technical Analysis**: Price patterns, indicators, and trends
        - **AI-Powered Recommendations**: LLM-driven investment advice
        """)
    
    # Main content area
    if analyze_button and symbol:
        st.session_state.current_symbol = symbol
        
        # Run analysis
        try:
            recommendation = asyncio.run(run_analysis(symbol))
            
            if recommendation:
                st.session_state.analysis_results[symbol] = recommendation
                
                # Display recommendation
                st.markdown(
                    create_recommendation_card(recommendation), 
                    unsafe_allow_html=True
                )
                
                # Analysis results
                col1, col2 = st.columns(2)
                
                with col1:
                    display_fundamental_analysis(recommendation)
                
                with col2:
                    display_technical_analysis(recommendation)
                
                # Key metrics
                display_key_metrics(recommendation)
                
                # Charts (if enabled)
                if show_charts:
                    st.header("📊 Charts & Visualizations")
                    
                    # Get stock data for charts
                    try:
                        factory = get_data_source_factory()
                        stock_data = asyncio.run(factory.get_stock_data(symbol, period))
                        
                        if stock_data and not stock_data.price_data.empty:
                            # Price chart
                            price_fig = create_price_chart(stock_data.price_data, symbol)
                            st.plotly_chart(price_fig, use_container_width=True)
                            
                            # Technical indicators chart
                            tech_fig = create_technical_indicators_chart(stock_data.price_data, symbol)
                            st.plotly_chart(tech_fig, use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"Error creating charts: {str(e)}")
                
                # Detailed analysis (if enabled)
                if show_detailed_analysis:
                    st.header("🔍 Detailed Analysis")
                    
                    with st.expander("View Full Analysis Details"):
                        st.write("**Recommendation Details:**")
                        st.json({
                            "symbol": recommendation.symbol,
                            "recommendation": recommendation.recommendation,
                            "confidence": recommendation.confidence,
                            "fundamental_score": recommendation.fundamental_score,
                            "technical_score": recommendation.technical_score,
                            "combined_score": recommendation.combined_score,
                            "risk_level": recommendation.risk_level,
                            "time_horizon": recommendation.time_horizon,
                            "key_factors": recommendation.key_factors
                        })
                
                # Add to history
                st.session_state.analysis_history.append({
                    "symbol": symbol,
                    "recommendation": recommendation.recommendation,
                    "confidence": recommendation.confidence,
                    "timestamp": recommendation.analysis_timestamp
                })
                
            else:
                st.error("Failed to analyze the stock. Please check the symbol and try again.")
                
        except Exception as e:
            st.error(f"Error running analysis: {str(e)}")
    
    # Analysis history
    if st.session_state.analysis_history:
        st.header("📋 Analysis History")
        
        history_df = pd.DataFrame(st.session_state.analysis_history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        history_df = history_df.sort_values('timestamp', ascending=False)
        
        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: gray;">
        <p>Research Agent v1.0 | Powered by LangChain & OpenAI</p>
        <p>⚠️ This is not financial advice. Always consult with a financial advisor before making investment decisions.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main() 