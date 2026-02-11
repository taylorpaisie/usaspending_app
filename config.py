"""
Configuration module for USAspending Dash App.

Loads configuration from environment variables with sensible defaults.
"""

import os
from functools import lru_cache



def _load_dotenv(path: str = ".env") -> None:
    """Load environment variables from a local .env file when present."""
    env_path = os.path.join(os.getcwd(), path)
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


# Load environment variables from .env file if it exists
_load_dotenv()


@lru_cache(maxsize=1)
def get_config() -> dict:
    """
    Load and cache configuration from environment variables.
    
    Returns:
        dict: Configuration settings with keys for API, cache, server, and UI settings.
    """
    return {
        # API Configuration
        "api_base_url": os.getenv("USASPENDING_API_BASE", "https://api.usaspending.gov/api/v2"),
        "api_timeout": int(os.getenv("API_TIMEOUT", "30")),
        "api_retries": int(os.getenv("API_RETRIES", "3")),
        
        # Cache Configuration
        "cache_max_size": int(os.getenv("CACHE_MAX_SIZE", "256")),
        "cache_ttl_minutes": int(os.getenv("CACHE_TTL_MINUTES", "10")),
        
        # Server Configuration
        "host": os.getenv("HOST", "0.0.0.0"),
        "port": int(os.getenv("PORT", "8050")),
        "debug": os.getenv("DEBUG", "True").lower() == "true",
        
        # UI Configuration
        "default_top_n": int(os.getenv("DEFAULT_TOP_N", "20")),
        "max_awards_table_rows": int(os.getenv("MAX_AWARDS_TABLE_ROWS", "50")),
        "date_range_days": int(os.getenv("DATE_RANGE_DAYS", "365")),
        "auto_refresh_seconds": int(os.getenv("AUTO_REFRESH_SECONDS", "60")),
        
        # Feature Flags
        "enable_logging": os.getenv("ENABLE_LOGGING", "True").lower() == "true",
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
    }


# Convenience accessor for quick access
CONFIG = get_config()
