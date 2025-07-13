# Research Agent Application Flow

## Overview

The Research Agent is a sophisticated stock analysis application that combines fundamental and technical analysis to provide comprehensive investment recommendations. Built with Streamlit for the frontend and LangGraph for the AI agent orchestration, it integrates multiple financial data sources and uses advanced AI reasoning to generate investment insights.

### Key Components
- **Frontend**: Streamlit web interface for user interaction
- **Agent Core**: LangGraph-based stock analysis agent
- **Data Sources**: YFinance, Alpha Vantage, Finnhub integrations
- **Analysis Engines**: Fundamental and technical analysis modules
- **AI Integration**: OpenAI GPT models for reasoning and recommendations

## User Journey

### 1. Application Launch
```
User → Streamlit Interface → Initialize Session State → Load Configuration
```

### 2. Stock Symbol Input
```
User enters symbol → Validation → Symbol normalization → Store in session state
```

### 3. Analysis Request
```
User clicks "Analyze" → Trigger async analysis → Show loading state → Display results
```

### 4. Results Display
```
Analysis complete → Display recommendation → Show charts → Update history
```

### 5. Historical Review
```
User views previous analyses → Filter by symbol → Compare recommendations
```

## System Architecture

### High-Level Components

```mermaid
graph TD
    A[User Interface<br/>Streamlit] --> B[Agent Controller<br/>stock_agent.py]
    B --> C[Data Sources<br/>Factory Pattern]
    B --> D[Analysis Engines<br/>Fundamental & Technical]
    B --> E[AI Reasoning<br/>OpenAI GPT]
    
    C --> F[YFinance<br/>Price Data]
    C --> G[Alpha Vantage<br/>Fundamentals]
    C --> H[Finnhub<br/>Company Data]
    
    D --> I[Technical Indicators<br/>SMA, RSI, MACD]
    D --> J[Fundamental Ratios<br/>PE, PEG, ROE]
    
    E --> K[Recommendation<br/>BUY/SELL/HOLD]
```

### Technology Stack
- **Frontend**: Streamlit 1.29.0+
- **AI Framework**: LangChain 0.1.0+, LangGraph 0.0.20+
- **Data Analysis**: Pandas 2.2.0+, NumPy 1.26.2+
- **Visualization**: Plotly 5.18.0+
- **Financial Data**: yfinance, alpha-vantage, finnhub-python
- **Technical Analysis**: ta library

## Data Flow

### 1. Data Collection Flow

```mermaid
sequenceDiagram
    participant U as User
    participant S as Streamlit
    participant A as Agent
    participant F as Data Factory
    participant Y as YFinance
    participant AV as Alpha Vantage
    participant FH as Finnhub
    
    U->>S: Enter stock symbol
    S->>A: analyze_stock(symbol)
    A->>F: get_stock_data(symbol)
    
    par Parallel Data Fetching
        F->>Y: fetch_price_data()
        F->>AV: fetch_fundamentals()
        F->>FH: fetch_company_profile()
    end
    
    Y-->>F: Price history, volume
    AV-->>F: Financial ratios, earnings
    FH-->>F: Company info, metrics
    
    F->>A: StockData object
    A->>A: Process and validate data
```

### 2. Data Processing Pipeline

```python
# Data flow through the system
StockData {
    symbol: str
    price_data: pandas.DataFrame      # OHLCV data
    fundamentals: Dict[str, Any]      # Financial metrics
    company_info: Dict[str, Any]      # Company details
    technical_indicators: Dict        # Calculated indicators
}
```

### 3. Data Validation Flow

```mermaid
graph TD
    A[Raw Data] --> B{Data Validation}
    B -->|Valid| C[Process Data]
    B -->|Invalid| D[Error Handling]
    C --> E[Store in StockData]
    D --> F[Fallback Data]
    F --> E
    E --> G[Continue Analysis]
```

## Agent Processing

### LangGraph Agent Workflow

The agent uses a structured decision-making process implemented with LangGraph:

```mermaid
graph TD
    A[START] --> B[fetch_data]
    B --> C[fundamental_analysis]
    C --> D[technical_analysis]
    D --> E[combine_analysis]
    E --> F[generate_recommendation]
    F --> G[validate_recommendation]
    G --> H[END]
    
    B -->|Error| I[Error State]
    C -->|Error| I
    D -->|Error| I
    I --> J[Fallback Recommendation]
    J --> H
```

### 1. Data Fetching Node (`fetch_data`)

```python
async def _fetch_data(self, state: AgentState) -> AgentState:
    """
    Flow:
    1. Validate symbol format
    2. Check if symbol exists in market
    3. Fetch data from multiple sources concurrently
    4. Validate data quality
    5. Store in state or handle errors
    """
```

### 2. Fundamental Analysis Node (`fundamental_analysis`)

```python
async def _run_fundamental_analysis(self, state: AgentState) -> AgentState:
    """
    Flow:
    1. Extract fundamental data from stock_data
    2. Calculate key ratios (P/E, P/B, ROE, etc.)
    3. Compare with industry benchmarks
    4. Generate fundamental score (0-100)
    5. Create fundamental signal (BUY/SELL/HOLD)
    """
```

### 3. Technical Analysis Node (`technical_analysis`)

```python
async def _run_technical_analysis(self, state: AgentState) -> AgentState:
    """
    Flow:
    1. Calculate technical indicators (SMA, RSI, MACD)
    2. Identify chart patterns
    3. Analyze trend direction and strength
    4. Generate technical score (0-100)
    5. Create technical signal (BUY/SELL/HOLD)
    """
```

### 4. Analysis Combination Node (`combine_analysis`)

```python
async def _combine_analysis(self, state: AgentState) -> AgentState:
    """
    Flow:
    1. Weight fundamental vs technical scores
    2. Consider market conditions
    3. Apply risk adjustments
    4. Calculate combined score
    5. Prepare summary for LLM
    """
```

### 5. Recommendation Generation Node (`generate_recommendation`)

```python
async def _generate_recommendation(self, state: AgentState) -> AgentState:
    """
    Flow:
    1. Prepare analysis summary for LLM
    2. Call OpenAI with structured prompt
    3. Parse LLM response
    4. Extract recommendation, confidence, reasoning
    5. Set risk level and time horizon
    """
```

### 6. Validation Node (`validate_recommendation`)

```python
async def _validate_recommendation(self, state: AgentState) -> AgentState:
    """
    Flow:
    1. Validate recommendation format
    2. Check confidence bounds (0-1)
    3. Ensure reasoning is present
    4. Verify risk/time horizon alignment
    5. Create final recommendation object
    """
```

## Error Handling

### Error Flow Hierarchy

```mermaid
graph TD
    A[Error Occurs] --> B{Error Type}
    B -->|Data Error| C[Data Fallback]
    B -->|API Error| D[API Fallback]
    B -->|Analysis Error| E[Analysis Fallback]
    B -->|LLM Error| F[LLM Fallback]
    
    C --> G[Partial Analysis]
    D --> H[Cached Data]
    E --> I[Simple Scoring]
    F --> J[Rule-based Rec]
    
    G --> K[HOLD Recommendation]
    H --> K
    I --> K
    J --> K
```

### Error Handling Levels

#### 1. Data Source Errors
```python
# Handle API failures gracefully
try:
    data = await api_source.fetch_data(symbol)
except (ConnectionError, TimeoutError) as e:
    logger.warning(f"API failure: {e}")
    data = await fallback_source.fetch_data(symbol)
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    return create_error_state(symbol, str(e))
```

#### 2. Analysis Errors
```python
# Handle analysis failures
try:
    score = calculate_fundamental_score(data)
except (ValueError, KeyError) as e:
    logger.warning(f"Analysis error: {e}")
    score = 50.0  # Neutral score
except Exception as e:
    logger.error(f"Critical analysis error: {e}")
    score = 0.0  # Conservative score
```

#### 3. LLM Errors
```python
# Handle LLM failures
try:
    recommendation = await llm.ainvoke(prompt)
except Exception as e:
    logger.error(f"LLM error: {e}")
    recommendation = create_fallback_recommendation(
        symbol, fundamental_score, technical_score
    )
```

### Error Recovery Strategies

1. **Graceful Degradation**: Provide partial results when possible
2. **Fallback Sources**: Switch to alternative data sources
3. **Cached Data**: Use recent cached data when APIs fail
4. **Conservative Defaults**: Default to HOLD recommendations on errors
5. **User Notification**: Inform users about data limitations

## API Integration

### Data Source Integration Flow

```mermaid
sequenceDiagram
    participant A as Agent
    participant F as Factory
    participant Y as YFinance
    participant AV as AlphaVantage
    participant FH as Finnhub
    
    A->>F: get_stock_data(symbol)
    F->>F: determine_best_source(symbol)
    
    par Primary Data Collection
        F->>Y: get_price_data(symbol)
        F->>AV: get_fundamentals(symbol)
        F->>FH: get_company_profile(symbol)
    end
    
    alt Success Path
        Y-->>F: OHLCV data
        AV-->>F: Financial metrics
        FH-->>F: Company info
        F->>F: merge_data_sources()
        F-->>A: Complete StockData
    else Failure Path
        Y-->>F: Error/Timeout
        F->>F: try_fallback_source()
        F-->>A: Partial StockData
    end
```

### API Rate Limiting

```python
# Implement rate limiting for API calls
from asyncio_throttle import Throttler

class RateLimitedAPIClient:
    def __init__(self, calls_per_minute: int = 60):
        self.throttler = Throttler(rate_limit=calls_per_minute, period=60)
    
    async def make_request(self, url: str, params: dict):
        async with self.throttler:
            return await self.session.get(url, params=params)
```

### API Error Handling

```python
async def fetch_with_retry(self, url: str, max_retries: int = 3):
    """Fetch data with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            return await self.make_request(url)
        except (ConnectionError, TimeoutError) as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

## Performance Considerations

### Caching Strategy

```python
# Multi-level caching
@lru_cache(maxsize=100)
def get_stock_data_cached(symbol: str, date: str) -> StockData:
    """Cache stock data by symbol and date."""
    pass

# Session-level caching
if symbol not in st.session_state.cache:
    st.session_state.cache[symbol] = fetch_data(symbol)
```

### Async Operations

```python
# Concurrent data fetching
async def fetch_all_data(symbol: str) -> StockData:
    tasks = [
        fetch_price_data(symbol),
        fetch_fundamentals(symbol),
        fetch_company_info(symbol)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return merge_results(results)
```

### Memory Management

```python
# Limit session state size
MAX_CACHE_SIZE = 50
if len(st.session_state.cache) > MAX_CACHE_SIZE:
    # Remove oldest entries
    oldest_key = min(st.session_state.cache.keys())
    del st.session_state.cache[oldest_key]
```

## Testing Strategy

### Test Flow Coverage

```python
# Test complete flow
@pytest.mark.asyncio
async def test_complete_analysis_flow():
    """Test the complete analysis flow end-to-end."""
    agent = get_stock_agent()
    
    # Test data flow
    result = await agent.analyze_stock("AAPL")
    
    # Verify flow completion
    assert result.symbol == "AAPL"
    assert result.recommendation in ["BUY", "SELL", "HOLD"]
    assert 0.0 <= result.confidence <= 1.0
    assert result.reasoning is not None
```

### Error Flow Testing

```python
# Test error handling
@pytest.mark.asyncio
async def test_error_flow():
    """Test error handling in the flow."""
    agent = get_stock_agent()
    
    # Test invalid symbol
    result = await agent.analyze_stock("INVALID")
    assert result.recommendation == "HOLD"
    assert result.confidence == 0.0
```

## Monitoring and Logging

### Flow Monitoring

```python
# Add flow monitoring
import time
import logging

logger = logging.getLogger(__name__)

async def _fetch_data(self, state: AgentState) -> AgentState:
    """Fetch data with monitoring."""
    start_time = time.time()
    
    try:
        # Fetch data
        duration = time.time() - start_time
        logger.info(f"Data fetch completed in {duration:.2f}s")
        
        # Log metrics
        self.metrics.record_timing("data_fetch", duration)
        self.metrics.increment("data_fetch_success")
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Data fetch failed after {duration:.2f}s: {e}")
        self.metrics.increment("data_fetch_error")
        
    return state
```

### Performance Tracking

```python
# Track key performance indicators
class PerformanceTracker:
    def __init__(self):
        self.metrics = {
            'analysis_time': [],
            'data_fetch_time': [],
            'llm_response_time': [],
            'success_rate': 0.0
        }
    
    def record_analysis(self, duration: float, success: bool):
        self.metrics['analysis_time'].append(duration)
        if success:
            self.metrics['success_rate'] += 1
```

## Deployment Considerations

### Environment Configuration

```python
# Environment-specific configuration
class Settings:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.finnhub_api_key = os.getenv("FINNHUB_API_KEY")
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        
        # Performance settings
        self.max_concurrent_requests = int(os.getenv("MAX_CONCURRENT_REQUESTS", 5))
        self.cache_ttl = int(os.getenv("CACHE_TTL", 300))  # 5 minutes
```

### Health Checks

```python
# Application health check
async def health_check() -> Dict[str, Any]:
    """Check application health."""
    health = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {}
    }
    
    # Check data sources
    try:
        await yfinance_source.validate_connection()
        health["services"]["yfinance"] = "healthy"
    except Exception as e:
        health["services"]["yfinance"] = f"error: {e}"
        health["status"] = "degraded"
    
    return health
```

## Future Enhancements

### Planned Flow Improvements

1. **Real-time Updates**: WebSocket integration for live data
2. **Advanced Caching**: Redis-based distributed caching
3. **Batch Processing**: Analyze multiple stocks simultaneously
4. **ML Integration**: Custom ML models for prediction
5. **Portfolio Analysis**: Multi-stock portfolio optimization

### Scalability Considerations

```python
# Horizontal scaling preparation
from langgraph.checkpoint.postgres import PostgresCheckpointSaver

# Use PostgreSQL for production checkpointing
checkpoint_saver = PostgresCheckpointSaver(
    connection_string="postgresql://user:pass@localhost/db"
)

agent = StockAnalysisAgent(checkpoint_saver=checkpoint_saver)
```

---

This documentation should be updated whenever the application flow changes. Keep it current to maintain development efficiency and prevent errors. 