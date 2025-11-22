# Social Media Community Analysis

This project analyzes different Reddit communities (subreddits) to see how similar they are based on the topics people discuss. It uses machine learning (BERTopic) to extract topics and then clusters communities together to find patterns.

## Folder Structure

📄 **[Detailed Project Requirements & Guidelines](https://docs.google.com/document/d/1dnXiTp3WDfJuSFIQXRD8FMca91je2ij8LA49rWh5uj8/edit?usp=sharing)**

*   **`scraping/`**: Contains the code to download posts from Reddit.
*   **`topic_modelling/`**: Has the scripts that use AI to figure out what topics are in the posts.
*   **`one_hot_encoding/`**: Converts the complex topic data into simple 1s and 0s so we can do math on it (like finding similarities).
*   **`2D visualization/`**: Scripts to make graphs and word clouds so we can see the results.
*   **`results/`**: This is where all the data goes.
    *   `scraped_data/`: The raw JSON files from Reddit.
    *   `topic_modelling_output/`: The topics found by the AI.
    *   `one_hot_encoding/`: The matrices and clustering results.
    *   `visualization/`: The generated charts.
*   **`images/`**: Saved word cloud images.
*   **`docs/`**: Extra documentation about the project goals.

## How to Run

Follow these steps in order to run the whole project:

### 1. Setup
First, install the required libraries:
```bash
pip install -r requirements.txt
```

You also need to create a `.env` file in the main folder with your Reddit API keys:
```
REDDIT_CLIENT_ID=your_id_here
REDDIT_CLIENT_SECRET=your_secret_here
```

### 2. Get Data
Run the scraper to download posts from the subreddits.
```bash
python run_scraper.py
```
*You can add `--posts 200` to download fewer posts if you want it to be faster.*

### 3. Find Topics
Run the topic modeling script. This uses BERTopic to read all the posts and find common themes.
```bash
python topic_modelling/topic_modeling.py
```
*This might take a few minutes depending on how much data you collected.*

### 4. Process Topics
Run this script to turn the topics into a simple "One-Hot" format (basically marking which topics are present in which subreddit).
```bash
python one_hot_encoding/one_hot.py
```

### 5. Analyze
Now you can run the analysis to see which communities are similar and group them using K-Means clustering.
```bash
python one_hot_encoding/cosine_kmeans.py
```

### 6. Visualize
Finally, generate the visualizations.

To make the 2D maps:
```bash
python "2D visualization/tsne_visualization.py"
```

To make the word clouds:
```bash
python "2D visualization/wordclouds.py"
```
