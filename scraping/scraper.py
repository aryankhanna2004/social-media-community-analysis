"""
Reddit Community Data Scraper

This module provides functionality to collect posts from Reddit subreddits
for community analysis research.

Author: Social Media Community Analysis Project
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Post:
    """Data class representing a Reddit post."""
    subreddit: str
    post_id: str
    title: str
    content: str
    author: str
    timestamp: datetime
    url: str
    score: int = 0
    num_comments: int = 0
    upvote_ratio: float = 0.0
    is_original_content: bool = False



class BaseScraper:
    """Base class for social media scrapers."""

    def __init__(self, rate_limit: float = 1.0):
        """
        Initialize the base scraper.

        Args:
            rate_limit: Minimum seconds between requests
        """
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.session = None

    async def _rate_limit_wait(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            await asyncio.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    async def _make_request(self, url: str, **kwargs) -> Optional[Dict]:
        """Make an HTTP request with rate limiting."""
        await self._rate_limit_wait()

        try:
            async with self.session.get(url, **kwargs) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"Request failed with status {response.status}: {url}")
                    return None
        except Exception as e:
            logger.error(f"Request error for {url}: {e}")
            return None

    def save_posts(self, posts: List[Post], filename: str):
        """Save posts to a JSON file."""
        data = [asdict(post) for post in posts]

        # Convert datetime objects to ISO format strings
        for item in data:
            if isinstance(item['timestamp'], datetime):
                item['timestamp'] = item['timestamp'].isoformat()

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(posts)} posts to {filename}")

    async def __aenter__(self):
        """Async context manager entry."""
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()


class RedditScraper(BaseScraper):
    """Scraper for Reddit subreddits."""

    def __init__(self, client_id: str = None, client_secret: str = None, user_agent: str = "SocialMediaAnalysis/1.0"):
        """
        Initialize Reddit scraper.

        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: User agent string for API requests
        """
        super().__init__(rate_limit=1.0)
        self.client_id = client_id or os.getenv('REDDIT_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('REDDIT_CLIENT_SECRET')
        self.user_agent = user_agent
        self.access_token = None
        self.token_expires = 0

    async def _get_access_token(self):
        """Get Reddit API access token."""
        if time.time() < self.token_expires - 60:  # Refresh 1 minute before expiry
            return

        auth_data = {
            'grant_type': 'client_credentials'
        }

        try:
            async with self.session.post(
                'https://www.reddit.com/api/v1/access_token',
                data=auth_data,
                auth=aiohttp.BasicAuth(self.client_id, self.client_secret),
                headers={'User-Agent': self.user_agent}
            ) as response:
                if response.status == 200:
                    token_data = await response.json()
                    self.access_token = token_data['access_token']
                    self.token_expires = time.time() + token_data['expires_in']
                    logger.info("Successfully obtained Reddit access token")
                else:
                    logger.error(f"Failed to get Reddit access token: {response.status}")
        except Exception as e:
            logger.error(f"Error getting Reddit access token: {e}")

    async def scrape_subreddit(self, subreddit_name: str, limit: int = 1000,
                             time_filter: str = 'month') -> List[Post]:
        """
        Scrape posts from a subreddit.

        Args:
            subreddit_name: Name of the subreddit (without r/)
            limit: Maximum number of posts to collect
            time_filter: Time filter ('hour', 'day', 'week', 'month', 'year', 'all')

        Returns:
            List of Post objects
        """
        await self._get_access_token()

        if not self.access_token:
            logger.error("No access token available for Reddit API")
            return []

        posts = []
        after = None

        while len(posts) < limit:
            params = {
                'limit': min(100, limit - len(posts)),
                't': time_filter
            }

            if after:
                params['after'] = after

            headers = {
                'Authorization': f'bearer {self.access_token}',
                'User-Agent': self.user_agent
            }

            url = f'https://oauth.reddit.com/r/{subreddit_name}/hot'
            data = await self._make_request(url, params=params, headers=headers)

            if not data or 'data' not in data:
                break

            for post_data in data['data']['children']:
                post = post_data['data']

                # Skip stickied posts and announcements
                if post.get('stickied', False):
                    continue

                post_obj = Post(
                    subreddit=subreddit_name,
                    post_id=post['id'],
                    title=post.get('title', ''),
                    content=post.get('selftext', ''),
                    author=post.get('author', '[deleted]'),
                    timestamp=datetime.fromtimestamp(post['created_utc']),
                    url=f"https://reddit.com{post['permalink']}",
                    score=post.get('score', 0),
                    num_comments=post.get('num_comments', 0),
                    upvote_ratio=post.get('upvote_ratio', 0),
                    is_original_content=post.get('is_original_content', False)
                )

                posts.append(post_obj)

            after = data['data'].get('after')
            if not after:
                break

            # Avoid rate limits
            await asyncio.sleep(1)

        logger.info(f"Collected {len(posts)} posts from r/{subreddit_name}")
        return posts[:limit]


class CommunityDataCollector:
    """Main class for collecting data from Reddit subreddits."""

    def __init__(self, output_dir: str = "data"):
        """
        Initialize the data collector.

        Args:
            output_dir: Directory to save collected data
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Initialize scraper
        self.reddit_scraper = RedditScraper()

    async def collect_subreddit_data(self, subreddits: List[str],
                                   posts_per_subreddit: int = 1000) -> Dict[str, List[Post]]:
        """
        Collect data from multiple subreddits.

        Args:
            subreddits: List of subreddit names (without r/)
            posts_per_subreddit: Target number of posts per subreddit

        Returns:
            Dictionary mapping subreddit names to lists of posts
        """
        all_data = {}

        async with self.reddit_scraper:
            for subreddit in subreddits:
                logger.info(f"Collecting data from r/{subreddit}")

                try:
                    posts = await self.reddit_scraper.scrape_subreddit(
                        subreddit, limit=posts_per_subreddit
                    )

                    if posts:
                        all_data[subreddit] = posts

                        # Save individual subreddit data
                        filename = os.path.join(self.output_dir, f"r_{subreddit}.json")
                        self.reddit_scraper.save_posts(posts, filename)

                except Exception as e:
                    logger.error(f"Error collecting data from r/{subreddit}: {e}")

        return all_data

    def save_combined_data(self, all_data: Dict[str, List[Post]], filename: str = "combined_data.json"):
        """Save all collected data to a single file."""
        combined_posts = []
        for posts in all_data.values():
            combined_posts.extend(posts)

        filepath = os.path.join(self.output_dir, filename)
        self.reddit_scraper.save_posts(combined_posts, filepath)

        logger.info(f"Saved {len(combined_posts)} total posts to {filepath}")

    def generate_data_summary(self, all_data: Dict[str, List[Post]]) -> Dict[str, Any]:
        """Generate a summary of collected data."""
        summary = {
            'total_subreddits': len(all_data),
            'total_posts': sum(len(posts) for posts in all_data.values()),
            'subreddit_stats': {}
        }

        for subreddit_name, posts in all_data.items():
            if posts:
                timestamps = [p.timestamp for p in posts]
                summary['subreddit_stats'][subreddit_name] = {
                    'total_posts': len(posts),
                    'date_range': f"{min(timestamps)} - {max(timestamps)}",
                    'avg_score': sum(p.score for p in posts) / len(posts),
                    'avg_comments': sum(p.num_comments for p in posts) / len(posts),
                    'avg_upvote_ratio': sum(p.upvote_ratio for p in posts) / len(posts)
                }

        return summary


async def main():
    """Example usage of the data collector."""

    # Import subreddits from configuration
    from . import SUBREDDITS as subreddits

    # Initialize collector
    collector = CommunityDataCollector(output_dir="data")

    # Collect data
    logger.info("Starting data collection...")
    all_data = await collector.collect_subreddit_data(subreddits, posts_per_subreddit=500)

    # Save combined data
    collector.save_combined_data(all_data)

    # Generate and save summary
    logger.info("Generating data summary...")
    summary = collector.generate_data_summary(all_data)

    # Save summary as JSON
    with open(os.path.join(collector.output_dir, "data_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Data collection completed!")
    logger.info(f"Collected data from {len(all_data)} subreddits")
    logger.info(f"Total posts collected: {sum(len(posts) for posts in all_data.values())}")


if __name__ == "__main__":
    # Run the data collection
    asyncio.run(main())
