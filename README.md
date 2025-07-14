# Research Agent 📈

A comprehensive AI-powered research agent that combines fundamental and technical analysis to provide intelligent investment recommendations. Built with LangChain, LangGraph, and Streamlit.

## Features

### 🔍 Comprehensive Analysis
- **Fundamental Analysis**: Financial ratios, growth metrics, profitability, and financial health
- **Technical Analysis**: Price patterns, momentum indicators, volume analysis, and trend identification
- **AI-Powered Recommendations**: LLM-driven investment advice combining both analysis types

### 📊 Data Sources
- **Yahoo Finance**: Free real-time and historical stock data
- **Alpha Vantage**: Enhanced fundamental data and financial statements
- **Multiple fallback sources**: Ensuring data reliability and availability

### 🎯 Key Capabilities
- Real-time stock symbol validation
- Multi-timeframe analysis (1 month to 5 years)
- Interactive visualizations with Plotly
- Historical analysis tracking
- Professional-grade scoring system
- Risk assessment and time horizon recommendations

## Architecture

### Core Components

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Data Sources      │    │   Analysis Engine   │    │   Agent Framework   │
│                     │    │                     │    │                     │
│ • Yahoo Finance     │───▶│ • Fundamental       │───▶│ • LangGraph Workflow│
│ • Alpha Vantage     │    │ • Technical         │    │ • LangChain LLM     │
│ • Data Factory      │    │ • Scoring System    │    │ • State Management  │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
                                                                  │
                                                                  ▼
                                                       ┌─────────────────────┐
                                                       │   Streamlit UI      │
                                                       │                     │
                                                       │ • Interactive Forms │
                                                       │ • Real-time Charts  │
                                                       │ • Analysis Results  │
                                                       └─────────────────────┘
```

### Analysis Framework

1. **Data Collection**: Multi-source data aggregation with validation
2. **Fundamental Analysis**: 
   - Valuation metrics (P/E, P/B, P/S, PEG)
   - Profitability analysis (margins, ROE, ROA)
   - Financial health (liquidity, leverage)
   - Growth analysis (revenue, earnings)
3. **Technical Analysis**:
   - Trend indicators (SMA, EMA, MACD)
   - Momentum oscillators (RSI, Stochastic)
   - Volume analysis (OBV, volume trends)
   - Support/resistance levels
4. **AI Integration**: LLM-powered recommendation synthesis
5. **Scoring & Recommendation**: Weighted scoring system with confidence levels

## Installation

### Prerequisites
- Python 3.11+
- OpenAI API key
- Optional: Alpha Vantage API key for enhanced data

### Quick Start

1. **Clone the repository**
   ```bash
git clone https://github.com/yourusername/research-agent.git
cd research-agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

### Docker Setup

```bash
# Development
docker-compose up

# Production
docker-compose -f docker-compose.yml up --build
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (for enhanced data)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here
FINNHUB_API_KEY=your_finnhub_api_key_here

# LangSmith (optional, for tracing)
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=research-agent
LANGSMITH_OTEL_ENABLED=1

# Model Configuration
MODEL_NAME=gpt-4-turbo-preview
MODEL_TEMPERATURE=0.0

# Analysis Configuration
DEFAULT_ANALYSIS_PERIOD=1y
MAX_CONCURRENT_REQUESTS=10
```

## Usage

### Web Interface

1. **Launch the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Enter a stock symbol** (e.g., AAPL, GOOGL, MSFT)

3. **Select analysis period** (1m, 3m, 6m, 1y, 5y)

4. **Click "Analyze Stock"** to get comprehensive analysis

### Programmatic Usage

```python
import asyncio
from src.agent.stock_agent import get_stock_agent

async def analyze_stock_example():
    # Initialize the agent
    agent = get_stock_agent()
    
    # Analyze a stock
    recommendation = await agent.analyze_stock("AAPL")
    
    # Print results
    print(f"Recommendation: {recommendation.recommendation}")
    print(f"Confidence: {recommendation.confidence:.2f}")
    print(f"Reasoning: {recommendation.reasoning}")
    print(f"Fundamental Score: {recommendation.fundamental_score:.1f}")
    print(f"Technical Score: {recommendation.technical_score:.1f}")
    
    # Clean up
    await agent.close()

# Run the example
asyncio.run(analyze_stock_example())
```

### Data Source Usage

```python
from src.data_sources.factory import get_data_source_factory

async def get_stock_data_example():
    factory = get_data_source_factory()
    
    # Get comprehensive stock data
    stock_data = await factory.get_stock_data("AAPL", "1y")
    
    print(f"Symbol: {stock_data.symbol}")
    print(f"Price data shape: {stock_data.price_data.shape}")
    print(f"Latest price: ${stock_data.price_data['close'].iloc[-1]:.2f}")
    print(f"Market cap: ${stock_data.fundamental_data.get('market_cap', 'N/A')}")
    
    await factory.close()

asyncio.run(get_stock_data_example())
```

## API Reference

### Core Classes

#### `StockAnalysisAgent`
Main agent class for stock analysis.

```python
class StockAnalysisAgent:
    async def analyze_stock(self, symbol: str) -> StockRecommendation:
        """Analyze a stock and return recommendation."""
        
    async def get_analysis_history(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Get historical analysis for a symbol."""
```

#### `StockRecommendation`
Data class containing analysis results.

```python
@dataclass
class StockRecommendation:
    symbol: str
    recommendation: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float    # 0.0 to 1.0
    reasoning: str
    fundamental_score: float
    technical_score: float
    combined_score: float
    risk_level: str      # 'LOW', 'MEDIUM', 'HIGH'
    time_horizon: str    # 'SHORT_TERM', 'MEDIUM_TERM', 'LONG_TERM'
    key_factors: List[str]
```

### Analysis Modules

#### Fundamental Analysis
```python
from src.analysis.fundamental import FundamentalAnalyzer

analyzer = FundamentalAnalyzer()
result = await analyzer.analyze(stock_data)
```

#### Technical Analysis
```python
from src.analysis.technical import TechnicalAnalyzer

analyzer = TechnicalAnalyzer()
result = await analyzer.analyze(stock_data)
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test types
pytest tests/unit/        # Unit tests only
pytest tests/integration/ # Integration tests only
pytest -m "not slow"      # Skip slow tests
```

### Test Coverage

The project maintains >85% test coverage with:
- Unit tests for all core modules
- Integration tests for complete workflows
- Mock tests for external API calls
- Async test support

## Development

### Project Structure

```
research-agent/
├── app.py                 # Streamlit application
├── requirements.txt       # Dependencies
├── docker-compose.yml     # Container orchestration
├── pytest.ini           # Test configuration
├── config/
│   └── settings.py       # Application settings
├── src/
│   ├── data_sources/     # Data retrieval modules
│   │   ├── base.py
│   │   ├── yfinance_source.py
│   │   ├── alpha_vantage_source.py
│   │   └── factory.py
│   ├── analysis/         # Analysis engines
│   │   ├── fundamental.py
│   │   └── technical.py
│   └── agent/            # LangGraph agent
│       └── stock_agent.py
└── tests/
    ├── unit/             # Unit tests
    └── integration/      # Integration tests
```

### Adding New Data Sources

1. **Create a new data source class**:
   ```python
   from src.data_sources.base import BaseDataSource
   
   class MyDataSource(BaseDataSource):
       async def get_price_data(self, symbol: str, period: str) -> pd.DataFrame:
           # Implementation
           pass
   ```

2. **Register in the factory**:
   ```python
   # In src/data_sources/factory.py
   if self.config.enable_my_source:
       self._sources["my_source"] = MyDataSource(api_key)
   ```

### Adding New Analysis Techniques

1. **Extend the analyzers**:
   ```python
   # In src/analysis/technical.py
   def _analyze_new_indicator(self, data: pd.DataFrame) -> Dict[str, Any]:
       # Your analysis logic
       return {"score": score, "signal": signal}
   ```

2. **Update the scoring system**:
   ```python
   # Add to weights configuration
   self.new_weights = {
       'new_indicator': 0.15,
       # ... other weights
   }
   ```

## Performance Optimization

### Async Operations
- All data fetching is async with concurrent requests
- Rate limiting for API calls
- Connection pooling for HTTP clients

### Caching
- Data source results caching
- Analysis result caching
- Streamlit session state management

### Memory Management
- Efficient pandas operations
- Cleanup of large datasets
- Connection cleanup on exit

## Security

### API Key Management
- Environment variable storage
- No hardcoded secrets
- Secure default configurations

### Input Validation
- Symbol validation
- Parameter sanitization
- Error handling for malformed input

### Rate Limiting
- Respect API rate limits
- Exponential backoff for retries
- Concurrent request limiting

## Contributing

### Development Setup

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/new-analysis-method
   ```
3. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run tests**
   ```bash
   pytest
   ```
5. **Submit a pull request**

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Add docstrings to all functions
- Write comprehensive tests
- Keep functions focused and small

### Commit Messages

Use conventional commits:
```
feat: add new technical indicator
fix: resolve data source timeout issue
docs: update API documentation
test: add integration tests for agent
```

## Deployment

### Production Deployment

1. **Build the Docker image**
   ```bash
   docker build -t research-agent --target prod .
   ```

2. **Deploy with docker-compose**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

3. **Health checks**
   ```bash
   curl http://localhost:8501/_stcore/health
   ```

### Environment-specific Configurations

- **Development**: Full logging, debug mode
- **Staging**: Production-like with test data
- **Production**: Optimized, secure, monitored

## Monitoring

### Metrics
- Request latency
- Analysis success rates
- API call volumes
- Error rates

### Logging
- Structured logging with context
- Error tracking and alerts
- Performance monitoring

### Observability
- OpenTelemetry integration
- LangSmith tracing
- Prometheus metrics

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

### Documentation
- [API Reference](docs/api.md)
- [Architecture Guide](docs/architecture.md)
- [Deployment Guide](docs/deployment.md)

### Community
- [GitHub Issues](https://github.com/yourusername/research-agent/issues)
- [Discussions](https://github.com/yourusername/research-agent/discussions)

## Disclaimer

⚠️ **Important**: This application is for educational and informational purposes only. The stock recommendations provided are based on algorithmic analysis and should not be considered as financial advice. Always consult with a qualified financial advisor before making investment decisions. The authors are not responsible for any financial losses incurred from using this application.

## Acknowledgments

- [LangChain](https://langchain.com/) for the AI framework
- [Streamlit](https://streamlit.io/) for the web interface
- [Yahoo Finance](https://finance.yahoo.com/) for stock data
- [Alpha Vantage](https://www.alphavantage.co/) for financial data
- [Plotly](https://plotly.com/) for interactive visualizations 