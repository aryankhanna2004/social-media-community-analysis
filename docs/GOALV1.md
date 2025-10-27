# Analyzing Community-Level Homogeneity and Heterogeneity in Social Media

## 1. Data Collection
   1. Select at least 10 social media sub-communities (e.g., from Reddit's subreddits, Bluesky's feeds, or Mastodon's servers)
   2. Crawl at least 200 trending/popular posts (1k to 5k posts are recommended for better experiment results) from each community

## 2. Interest/Topic Extraction
   1. Apply topic modeling on all collected social media posts using the bertopic Python library (https://bertopic.readthedocs.io/en/latest/)
   2. After topic modeling, each post will have an assigned topic. You can build a community-level one-hot (1XN) dimension vector topic representation (0 = not an interested topic; 1 = an interested topic) for each community. You need to set a threshold (e.g., <5% or <10 posts) to filter out rarely interested topics. For example, if community-1 only has very few posts on topic-1, you should mark this cell to "0"

|            | Topic-1 | Topic-2 | … | Topic_n |
|------------|---------|---------|---|---------|
| Community-1| 0       | 1       |   | 0       |
| Community-2| 1       | 1       |   | 1       |
| …          |         |         |   |         |
| Community-k| 1       | 0       |   | 0       |

   3. Compute the cosine similarity for each community pair
   4. Apply K-means with different "K" to detect community-level clusters
   5. Build an asymmetric matrix to indicate the fraction of shared topics
      1. Given a community pair, please calculate the num_of_shared_topics / num_of_distinct_topics

For example, <C1, C2> shares 2 topics and covers 4 topics in total. The fraction of shared topics will be 2/4 = 0.5

|            | Community-1 | Community-2 | … | Community-k |
|------------|-------------|-------------|---|-------------|
| Community-1| 1           | 0.5         |   |             |
| Community-2| 0.5         | 1           |   | 1           |
| …          |             |             |   |             |
| Community-k|             |             |   | 1           |

## 3. Result Visualization
   1. Create Word Clouds and give proper names for each topic
   2. Create 2D visualizations for:
      1. The most heterogeneous community (one that covers the most topics)
      2. The most homogeneous community (one that covers the fewest topics)
      3. Each dot in the 2D map represents one post; use the same color for posts that belong to the same topic

## 4. Discussion and Report
   1. Compare and discuss results in section 2
   2. Follow the same guidance in the course project description (section 3). In the final submission, you need to include: (1) source codes; (2) datasets; (3) README.md; (4) requirements.txt; and (5) a PDF report with 7 sections

Grading will be based on the level of completeness and the overall effort you put in.
