"""
Application Settings and Configuration
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    
    # Financial Data API Keys
    alpha_vantage_api_key: Optional[str] = Field(None, env="ALPHA_VANTAGE_API_KEY")
    finnhub_api_key: Optional[str] = Field(None, env="FINNHUB_API_KEY")
    
    # LangSmith Configuration
    langsmith_api_key: Optional[str] = Field(None, env="LANGSMITH_API_KEY")
    langsmith_project: str = Field("research-agent", env="LANGSMITH_PROJECT")
    langsmith_otel_enabled: bool = Field(True, env="LANGSMITH_OTEL_ENABLED")
    
    # Database Configuration
    database_url: str = Field("sqlite:///./stock_analysis.db", env="DATABASE_URL")
    
    # Streamlit Configuration
    streamlit_server_port: int = Field(8501, env="STREAMLIT_SERVER_PORT")
    streamlit_server_address: str = Field("0.0.0.0", env="STREAMLIT_SERVER_ADDRESS")
    
    # Security
    secret_key: str = Field("development-secret-key", env="SECRET_KEY")
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    # Analysis Configuration
    default_analysis_period: str = Field("1y", env="DEFAULT_ANALYSIS_PERIOD")
    max_concurrent_requests: int = Field(10, env="MAX_CONCURRENT_REQUESTS")
    
    # Model Configuration
    model_name: str = Field("gpt-4-turbo-preview", env="MODEL_NAME")
    model_temperature: float = Field(0.0, env="MODEL_TEMPERATURE")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False
    }


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the application settings."""
    return settings 