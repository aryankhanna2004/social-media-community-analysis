#!/usr/bin/env python3
"""
Social Media Data Collection Script

This script runs the data collection process for the social media community analysis project.

Usage:
    python run_scraper.py [--communities COUNT] [--posts COUNT] [--output DIR]

Options:
    --communities COUNT    Number of communities to collect from (default: all available)
    --posts COUNT         Number of posts per community (default: 500)
    --output DIR          Output directory (default: data)
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Add the scraping module to the path
sys.path.insert(0, str(Path(__file__).parent / "scraping"))

from scraping import CommunityDataCollector
from scraping import get_available_subreddits, validate_credentials

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main data collection function."""
    parser = argparse.ArgumentParser(description="Collect social media data for community analysis")
    parser.add_argument(
        "--communities",
        type=int,
        help="Number of communities to collect from (default: all available)"
    )
    parser.add_argument(
        "--posts",
        type=int,
        default=500,
        help="Number of posts per community (default: 500)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/scraped_data",
        help="Output directory (default: results/scraped_data)"
    )

    args = parser.parse_args()

    # Validate credentials
    logger.info("Validating Reddit API credentials...")
    credentials_valid = validate_credentials()

    if not credentials_valid:
        logger.error("Reddit API credentials not available. Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables.")
        return

    # Get available subreddits
    subreddits = get_available_subreddits()
    logger.info(f"Found {len(subreddits)} available subreddits")

    if not subreddits:
        logger.error("No subreddits available for scraping. Check your API credentials.")
        return

    # Limit subreddits if specified
    if args.communities:
        subreddits = subreddits[:args.communities]
        logger.info(f"Limited to {len(subreddits)} subreddits")

    # Initialize collector
    logger.info(f"Initializing data collector with output directory: {args.output}")
    collector = CommunityDataCollector(output_dir=args.output)

    # Collect data
    logger.info(f"Starting data collection ({args.posts} posts per subreddit)...")
    try:
        all_data = await collector.collect_subreddit_data(
            subreddits,
            posts_per_subreddit=args.posts
        )

        # Save combined data
        logger.info("Saving combined data...")
        collector.save_combined_data(all_data)

        # Generate and save summary
        logger.info("Generating data summary...")
        summary = collector.generate_data_summary(all_data)
        summary_path = Path(args.output) / "data_summary.json"
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Print results
        total_posts = sum(len(posts) for posts in all_data.values())
        logger.info("Data collection completed successfully!")
        logger.info(f"Subreddits processed: {len(all_data)}")
        logger.info(f"Total posts collected: {total_posts}")
        logger.info(f"Average posts per subreddit: {total_posts / len(all_data):.1f}")
        logger.info(f"Data saved to: {args.output}")
        logger.info(f"Summary saved to: {summary_path}")

        # Show subreddit breakdown
        print("\nSubreddit Breakdown:")
        for subreddit_name, posts in all_data.items():
            print(f"  r/{subreddit_name}: {len(posts)} posts")

    except KeyboardInterrupt:
        logger.info("Data collection interrupted by user")
    except Exception as e:
        logger.error(f"Error during data collection: {e}")
        raise


def show_help():
    """Show usage information."""
    print(__doc__)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        show_help()
    else:
        asyncio.run(main())
