"""
Reddit Community Data Scraping Package

This package provides tools for collecting Reddit posts from multiple subreddits
for community analysis research.
"""

import os
from typing import List, Dict, Any

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, continue with system env vars
    pass

from .scraper import (
    BaseScraper,
    RedditScraper,
    CommunityDataCollector,
    Post
)

__version__ = "1.0.0"

# Default configuration
DEFAULT_CONFIG = {
    'posts_per_community': 1000,
    'output_dir': 'results/scraped_data',
    'rate_limits': {
        'reddit': 1.0,  # seconds between requests
        'bluesky': 0.5,
        'mastodon': 1.0
    }
}

# Subreddit definitions
# Asian entertainment and media communities for social analysis
SUBREDDITS: List[str] = [
    'indiantellytalk',
    'bollywood',
    'kpop',
    'kdramas',
    'cdrama',
    'cpop',
    'jpop',
    'anime',
    'PPOPcommunity',
    'AsianDrama',
    'AsianCinema'
]

# API credentials (set via environment variables)
API_CREDENTIALS = {
    'reddit': {
        'client_id': os.getenv('REDDIT_CLIENT_ID'),
        'client_secret': os.getenv('REDDIT_CLIENT_SECRET')
    }
}

def validate_credentials() -> bool:
    """
    Validate that required Reddit API credentials are available.

    Returns:
        True if Reddit credentials are available, False otherwise
    """
    reddit_creds = API_CREDENTIALS['reddit']
    return bool(reddit_creds['client_id'] and reddit_creds['client_secret'])

def get_available_subreddits() -> List[str]:
    """
    Get subreddits that can be scraped based on available credentials.

    Returns:
        List of subreddit names that can be accessed
    """
    if validate_credentials():
        return SUBREDDITS
    else:
        print("Warning: Reddit credentials not available. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables.")
        return []

def create_data_collection_config(custom_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create a complete data collection configuration.

    Args:
        custom_config: Custom configuration overrides

    Returns:
        Complete configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    if custom_config:
        config.update(custom_config)

    config['subreddits'] = get_available_subreddits()
    config['credentials'] = API_CREDENTIALS

    return config

__all__ = [
    "BaseScraper",
    "RedditScraper",
    "CommunityDataCollector",
    "Post",
    "DEFAULT_CONFIG",
    "SUBREDDITS",
    "API_CREDENTIALS",
    "validate_credentials",
    "get_available_subreddits",
    "create_data_collection_config"
]
