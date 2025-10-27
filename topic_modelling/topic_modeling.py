#!/usr/bin/env python3
"""
Topic Modeling for Social Media Community Analysis

This script applies BERTopic modeling to analyze topics in collected social media posts
from various subreddits. It processes the combined data and generates topic clusters,
visualizations, and analysis results.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SocialMediaTopicModeler:
    """Class for performing topic modeling on social media posts using BERTopic."""

    def __init__(self, data_path: str = "../results/scraped_data/combined_data.json"):
        """
        Initialize the topic modeler.

        Args:
            data_path: Path to the combined data JSON file
        """
        self.data_path = Path(data_path)
        self.posts_df = None
        self.topic_model = None
        self.topics = None
        self.probs = None

        # Create output directory
        self.output_dir = Path(__file__).parent.parent / "results" / "topic_modelling_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized topic modeler with data path: {self.data_path}")

    def load_data(self) -> pd.DataFrame:
        """
        Load and preprocess the social media posts data.

        Returns:
            DataFrame containing processed posts
        """
        logger.info("Loading data from JSON file...")

        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                posts = json.load(f)
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.data_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file: {e}")
            sys.exit(1)

        logger.info(f"Loaded {len(posts)} posts from JSON file")

        # Convert to DataFrame
        df = pd.DataFrame(posts)

        # Combine title and content for text analysis
        df['full_text'] = df.apply(self._combine_text, axis=1)

        # Filter out posts with no text content
        initial_count = len(df)
        df = df[df['full_text'].str.len() > 10].copy()  # Remove very short posts
        filtered_count = len(df)

        logger.info(f"Filtered out {initial_count - filtered_count} posts with insufficient text")
        logger.info(f"Remaining posts: {filtered_count}")

        # Add text length for analysis
        df['text_length'] = df['full_text'].str.len()

        self.posts_df = df
        return df

    def _combine_text(self, row: pd.Series) -> str:
        """
        Combine title and content into a single text field.

        Args:
            row: DataFrame row containing post data

        Returns:
            Combined text from title and content
        """
        title = str(row.get('title', '')).strip()
        content = str(row.get('content', '')).strip()

        if content and title:
            return f"{title}. {content}"
        elif title:
            return title
        elif content:
            return content
        else:
            return ""

    def preprocess_text(self) -> List[str]:
        """
        Extract and preprocess text for topic modeling.

        Returns:
            List of processed text documents
        """
        if self.posts_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        texts = self.posts_df['full_text'].tolist()

        logger.info(f"Extracted {len(texts)} text documents for topic modeling")

        # Basic text cleaning (BERTopic handles most preprocessing internally)
        cleaned_texts = []
        for text in texts:
            # Remove excessive whitespace
            text = ' '.join(text.split())
            cleaned_texts.append(text)

        return cleaned_texts

    def create_topic_model(self, min_topic_size: int = 50, nr_topics: Optional[int] = None) -> BERTopic:
        """
        Create and configure the BERTopic model.

        Args:
            min_topic_size: Minimum size of topics
            nr_topics: Number of topics to extract (None for automatic)

        Returns:
            Configured BERTopic model
        """
        logger.info("Creating BERTopic model...")

        # Configure embedding model
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Configure vectorizer
        vectorizer_model = CountVectorizer(
            stop_words="english",
            min_df=5,
            max_df=0.8,
            ngram_range=(1, 2)
        )

        # Configure clustering model
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )

        # Configure representation model
        representation_model = KeyBERTInspired()

        # Create BERTopic model
        self.topic_model = BERTopic(
            embedding_model=embedding_model,
            vectorizer_model=vectorizer_model,
            hdbscan_model=hdbscan_model,
            representation_model=representation_model,
            ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
            nr_topics=nr_topics,
            verbose=True
        )

        logger.info("BERTopic model created successfully")
        return self.topic_model

    def fit_model(self, texts: List[str]) -> tuple:
        """
        Fit the topic model to the text data.

        Args:
            texts: List of text documents

        Returns:
            Tuple of (topics, probabilities)
        """
        logger.info("Fitting topic model to data...")

        self.topics, self.probs = self.topic_model.fit_transform(texts)

        logger.info(f"Topic modeling completed. Found {len(set(self.topics))} topics")
        return self.topics, self.probs

    def analyze_topics(self) -> Dict[str, Any]:
        """
        Analyze the discovered topics and their properties.

        Returns:
            Dictionary containing topic analysis results
        """
        if self.topic_model is None or self.topics is None:
            raise ValueError("Model not fitted. Call fit_model() first.")

        logger.info("Analyzing discovered topics...")

        # Get topic information
        topic_info = self.topic_model.get_topic_info()

        # Get topic representations
        topic_representations = {}
        for topic_id in topic_info['Topic']:
            if topic_id != -1:  # Skip outlier topic
                topic_representations[topic_id] = self.topic_model.get_topic(topic_id)

        # Analyze topic distribution
        topic_counts = pd.Series(self.topics).value_counts().sort_index()
        topic_counts = topic_counts[topic_counts.index != -1]  # Remove outliers

        # Analyze subreddit distribution per topic
        topic_subreddit_dist = {}
        for topic_id in topic_counts.index:
            topic_posts = self.posts_df.iloc[np.where(np.array(self.topics) == topic_id)[0]]
            subreddit_counts = topic_posts['subreddit'].value_counts()
            topic_subreddit_dist[topic_id] = subreddit_counts.to_dict()

        analysis_results = {
            'topic_info': topic_info,
            'topic_representations': topic_representations,
            'topic_counts': topic_counts.to_dict(),
            'topic_subreddit_distribution': topic_subreddit_dist,
            'total_posts': len(self.posts_df),
            'outlier_posts': (np.array(self.topics) == -1).sum(),
            'num_topics': len(topic_counts)
        }

        logger.info(f"Analysis completed. Found {len(topic_counts)} meaningful topics")
        return analysis_results


    def save_results(self, analysis_results: Dict[str, Any]):
        """Save topic modeling results to files."""
        logger.info("Saving results to files...")

        # Save topic information
        topic_info_path = self.output_dir / "topic_info.csv"
        analysis_results['topic_info'].to_csv(topic_info_path, index=False)
        logger.info(f"Saved topic info to {topic_info_path}")

        # Save detailed analysis
        analysis_path = self.output_dir / "topic_analysis.json"
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        logger.info(f"Saved analysis results to {analysis_path}")

        # Save posts with topic assignments
        posts_with_topics = self.posts_df.copy()
        posts_with_topics['topic'] = self.topics

        # Handle topic probabilities - can be 1D or 2D array
        if self.probs is not None:
            if self.probs.ndim == 1:
                # 1D array - use values directly
                posts_with_topics['topic_probability'] = self.probs
            elif self.probs.ndim == 2:
                # 2D array - take max probability for each document
                posts_with_topics['topic_probability'] = self.probs.max(axis=1)
            else:
                posts_with_topics['topic_probability'] = None
        else:
            posts_with_topics['topic_probability'] = None

        posts_path = self.output_dir / "posts_with_topics.csv"
        posts_with_topics.to_csv(posts_path, index=False)
        logger.info(f"Saved posts with topics to {posts_path}")

        # Save topic representations - convert numpy types to Python types
        representations_path = self.output_dir / "topic_representations.json"
        
        # Convert topic representations to JSON-serializable format
        serializable_representations = {}
        for topic_id, words_scores in analysis_results['topic_representations'].items():
            # Convert each (word, score) tuple to use Python float instead of numpy float32
            serializable_representations[str(topic_id)] = [
                (word, float(score)) for word, score in words_scores
            ]
        
        with open(representations_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_representations, f, indent=2)
        logger.info(f"Saved topic representations to {representations_path}")

    def run_complete_analysis(self):
        """Run the complete topic modeling pipeline."""
        logger.info("Starting complete topic modeling analysis...")

        # Load and preprocess data
        self.load_data()
        texts = self.preprocess_text()

        # Create and fit model
        self.create_topic_model(min_topic_size=30, nr_topics=None)
        self.fit_model(texts)

        # Analyze results
        analysis_results = self.analyze_topics()

        # Save results
        self.save_results(analysis_results)

        logger.info("Topic modeling analysis completed successfully!")
        logger.info(f"Results saved to: {self.output_dir}")

        # Print summary
        self._print_summary(analysis_results)

    def _print_summary(self, analysis_results: Dict[str, Any]):
        """Print a summary of the topic modeling results."""
        print("\n" + "="*60)
        print("TOPIC MODELING ANALYSIS SUMMARY")
        print("="*60)

        total_posts = analysis_results['total_posts']
        outlier_posts = analysis_results['outlier_posts']
        num_topics = analysis_results['num_topics']

        print(f"Total posts analyzed: {total_posts}")
        print(f"Outlier posts (not assigned to topics): {outlier_posts}")
        print(f"Posts assigned to topics: {total_posts - outlier_posts}")
        print(f"Number of topics discovered: {num_topics}")

        print(f"\nTop 10 Topics by Size:")
        topic_counts = analysis_results['topic_counts']
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)

        for i, (topic_id, count) in enumerate(sorted_topics[:10]):
            topic_name = analysis_results['topic_info'][analysis_results['topic_info']['Topic'] == topic_id]['Name'].iloc[0]
            percentage = (count / (total_posts - outlier_posts)) * 100
            print(f"  {i+1}. {topic_name}: {count} posts ({percentage:.1f}%)")

        print(f"\nResults saved to: {self.output_dir}")
        print("="*60)


def main():
    """Main function to run the topic modeling analysis."""
    # Check if data file exists
    data_path = Path(__file__).parent.parent / "results" / "scraped_data" / "combined_data.json"
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.error("Please ensure the combined_data.json file exists in the results/scraped_data directory")
        sys.exit(1)

    # Initialize and run analysis
    modeler = SocialMediaTopicModeler(str(data_path))
    modeler.run_complete_analysis()


if __name__ == "__main__":
    main()
