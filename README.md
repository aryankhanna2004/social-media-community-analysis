# Social Media Community Analysis

A comprehensive analysis project examining homogeneity and heterogeneity in social media communities using advanced machine learning techniques.

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
├── README.md           # Project documentation
├── GOALV1.md          # Detailed project requirements
├── LICENSE            # Project license
└── requirements.txt   # Python dependencies (to be added)
```

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Follow the detailed requirements in `GOALV1.md`

## Requirements

- Python 3.8+
- BERTopic library
- Scikit-learn
- Pandas, NumPy
- Visualization libraries (Matplotlib, Seaborn)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
