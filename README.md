# Social Media Community Analysis

A comprehensive analysis project examining homogeneity and heterogeneity in social media communities using advanced machine learning techniques.

## Project Documentation

📄 **[Detailed Project Requirements & Guidelines](https://docs.google.com/document/d/1dnXiTp3WDfJuSFIQXRD8FMca91je2ij8LA49rWh5uj8/edit?usp=sharing)**

## Overview

This project analyzes social media communities to understand their topic distributions, similarities, and clustering patterns. It employs BERTopic modeling, cosine similarity calculations, and K-means clustering to extract meaningful insights from community data.

## Features

- **Topic Modeling**: Uses BERTopic library for automatic topic extraction from social media posts
- **Community Analysis**: Computes cosine similarity between communities based on topic distributions
- **Clustering**: Applies K-means clustering to identify community groups
- **Visualization**: Creates interactive visualizations including word clouds and 2D topic maps
- **Similarity Matrix**: Builds asymmetric matrices showing shared topic fractions

## Methodology

1. **Data Collection**: Crawl posts from 10+ social media sub-communities
2. **Topic Extraction**: Apply BERTopic modeling to identify topics
3. **Community Representation**: Create one-hot topic vectors for each community
4. **Similarity Analysis**: Compute cosine similarity and shared topic fractions
5. **Clustering**: Use K-means to detect community clusters
6. **Visualization**: Generate word clouds and 2D visualizations

## Project Structure

```
social-media-community-analysis/
├── README.md              # Project documentation
├── GOALV1.md             # Detailed project requirements
├── LICENSE               # Project license
├── requirements.txt      # Python dependencies
├── run_scraper.py        # Main script to run data collection
├── env-example.txt       # Environment variables template
├── .env                  # Reddit API credentials (auto-loaded)
├── scraping/             # Reddit data collection module
│   ├── __init__.py      # Package initialization and configuration
│   └── scraper.py       # Main scraping functionality
├── test_env_simple.py   # Test credential loading
└── data/                # Output directory (created during scraping)
    ├── combined_data.json    # All collected posts
    ├── data_summary.json     # Collection statistics
    └── r_[subreddit].json    # Individual subreddit data
```

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up Reddit API credentials (see Data Collection section)
4. Run data collection: `python run_scraper.py`
5. Follow the detailed requirements in `GOALV1.md`

## Data Collection

This project collects data from Reddit subreddits using the official Reddit API. To use the scraper:

### Reddit API Setup

✅ **COMPLETED**: Your Reddit app is created and `.env` file is configured!

The scraper automatically loads credentials from the `.env` file in your project root.

**Credentials Setup**:  
- Create a `.env` file in your project root using the template provided in `env-example.txt` (or `.env.example`).  
- Fill in your own Reddit API `client_id` and `client_secret` as described in the template.

### Running the Scraper

```bash
# Collect data from all configured subreddits
python run_scraper.py

# Collect from first 5 subreddits only
python run_scraper.py --communities 5

# Collect 200 posts per subreddit
python run_scraper.py --posts 200

# Save to custom directory
python run_scraper.py --output my_data
```

### Default Subreddits

The scraper is configured to collect from 13 data science and AI-related subreddits:
- MachineLearning, datascience, Python, statistics, ArtificialIntelligence
- deeplearning, DataEngineering, rstats, computervision, NLP
- bigdata, analytics, machinelearningmemes

### Output Format

Data is saved as JSON files with the following structure:

```json
{
  "subreddit": "MachineLearning",
  "post_id": "abc123",
  "title": "Example Post Title",
  "content": "Post content here...",
  "author": "username",
  "timestamp": "2024-01-15T10:30:00",
  "url": "https://reddit.com/r/MachineLearning/comments/abc123/",
  "score": 150,
  "num_comments": 25,
  "upvote_ratio": 0.85,
  "is_original_content": false
}
```

## Next Steps

Now that your Reddit API is set up, you can:

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test Credentials
```bash
# Test that .env credentials are loaded
python test_env_simple.py
```

### 3. Run Data Collection
```bash
# Collect from all 13 subreddits (500 posts each)
python run_scraper.py

# Or collect from first 3 subreddits only
python run_scraper.py --communities 3

# Or collect 200 posts per subreddit
python run_scraper.py --posts 200
```

### 4. Check Results
After running, you'll find collected data in the `data/` folder:
- `combined_data.json` - All posts in one file
- `data_summary.json` - Collection statistics
- `r_[subreddit].json` - Individual subreddit data

### 5. Proceed to Analysis
Once you have the data, move to the next phase:
- Apply BERTopic for topic modeling
- Compute similarity matrices
- Generate visualizations

## Requirements

- Python 3.8+
- BERTopic library
- Scikit-learn
- Pandas, NumPy
- Visualization libraries (Matplotlib, Seaborn)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
