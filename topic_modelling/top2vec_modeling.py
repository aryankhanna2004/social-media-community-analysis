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

import pandas as pd
from tqdm import tqdm

# Top2Vec for social media topic modeling
try:
    from top2vec import Top2Vec
    TOP2VEC_AVAILABLE = True
except ImportError:
    TOP2VEC_AVAILABLE = False
    print("Top2Vec not installed. Install with: pip install top2vec")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SocialMediaTopicModeler:
    """Class for performing topic modeling on social media posts using Top2Vec."""

    def __init__(self, data_path: str = "../results/scraped_data/combined_data.json"):
        """
        Initialize the Top2Vec topic modeler.

        Args:
            data_path: Path to the combined data JSON file
        """
        self.data_path = Path(data_path)
        self.posts_df = None
        self.top2vec_model = None

        # Create output directory
        self.output_dir = Path(__file__).parent.parent / "results" / "topic_modelling_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized Top2Vec topic modeler with data path: {self.data_path}")

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

    def create_top2vec_model(self) -> Optional[Top2Vec]:
        """
        Create a Top2Vec model for better social media topic discovery.

        Returns:
            Top2Vec model or None if not available
        """
        if not TOP2VEC_AVAILABLE:
            logger.warning("Top2Vec not available. Install with: pip install top2vec")
            return None

        logger.info("Creating Top2Vec model for social media topic modeling...")
        return Top2Vec

    def fit_top2vec_model(self, texts: List[str]) -> Optional[Dict[str, Any]]:
        """
        Fit Top2Vec model to the text data.

        Args:
            texts: List of text documents

        Returns:
            Dictionary with Top2Vec results or None if failed
        """
        if not TOP2VEC_AVAILABLE:
            return None

        try:
            logger.info("Fitting Top2Vec model to data...")

            # Create Top2Vec model
            self.top2vec_model = Top2Vec(
                documents=texts,
                speed="learn",  # Learn embeddings (slower but better)
                workers=4 if len(texts) > 1000 else 1
            )

            # Get topic information
            num_topics = self.top2vec_model.get_num_topics()
            topic_words, word_scores, topic_nums = self.top2vec_model.get_topics(num_topics)

            logger.info(f"Top2Vec found {num_topics} topics")

            return {
                'model': self.top2vec_model,
                'num_topics': num_topics,
                'topic_words': topic_words,
                'word_scores': word_scores,
                'topic_nums': topic_nums
            }

        except Exception as e:
            logger.error(f"Top2Vec fitting failed: {e}")
            return None

    def analyze_top2vec_results(self, top2vec_results: Dict[str, Any], texts: List[str]) -> Dict[str, Any]:
        """
        Analyze Top2Vec results and create analysis summary.

        Args:
            top2vec_results: Results from fit_top2vec_model
            texts: Original text documents

        Returns:
            Dictionary with analysis results
        """
        logger.info("Analyzing Top2Vec results...")

        model = top2vec_results['model']
        num_topics = top2vec_results['num_topics']

        # Get document-topic assignments using Top2Vec methods
        doc_topics, doc_scores, _, _ = model.get_documents_topics(list(range(len(texts))))
        topic_words, word_scores, topic_nums = model.get_topics(num_topics)

        # Get topic sizes
        topic_sizes = {}
        for i in range(num_topics):
            topic_sizes[i] = sum(1 for topic in doc_topics if topic == i)

        # Top2Vec uses cosine similarity scores (doc_scores), not probabilities
        # Scores range from -1 to 1, with higher values indicating better fit
        # Documents with low similarity scores are potential outliers
        outlier_threshold = 0.5  # Documents with similarity < 0.5 are considered outliers
        outlier_posts = sum(1 for score in doc_scores if score < outlier_threshold)
        assigned_posts = len(texts) - outlier_posts

        # Get topic information
        topic_info = []
        for topic_id in range(num_topics):
            topic_words_list = topic_words[topic_id][:10] if topic_id < len(topic_words) else []
            topic_info.append({
                'Topic': topic_id,
                'Count': topic_sizes[topic_id],
                'Name': f"{topic_id}_{'_'.join(topic_words_list[:4])}",
                'Representation': topic_words_list
            })

        # Create topic info DataFrame
        topic_info_df = pd.DataFrame(topic_info)

        analysis_results = {
            'topic_info': topic_info_df,
            'topic_representations': {i: topic_words[i] for i in range(num_topics)},
            'topic_counts': topic_sizes,
            'total_posts': len(texts),
            'outlier_posts': outlier_posts,
            'assigned_posts': assigned_posts,
            'num_topics': num_topics,
            'doc_topics': doc_topics,
            'doc_scores': doc_scores,
            'model': model
        }

        logger.info(f"Analysis completed. Found {num_topics} topics with {assigned_posts} assigned posts")
        return analysis_results

    def save_top2vec_results(self, analysis_results: Dict[str, Any]):
        """Save Top2Vec analysis results to files."""
        logger.info("Saving Top2Vec results to files...")

        # Save topic information
        topic_info_path = self.output_dir / "topic_info.csv"
        analysis_results['topic_info'].to_csv(topic_info_path, index=False)
        logger.info(f"Saved topic info to {topic_info_path}")

        # Save detailed analysis
        analysis_path = self.output_dir / "topic_analysis.json"
        # Remove non-serializable items
        serializable_results = {k: v for k, v in analysis_results.items()
                               if k not in ['model', 'doc_topics', 'doc_scores']}
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        logger.info(f"Saved analysis results to {analysis_path}")

        # Save posts with topic assignments
        posts_with_topics = self.posts_df.copy()
        posts_with_topics['topic'] = analysis_results['doc_topics']
        posts_with_topics['topic_similarity_score'] = analysis_results['doc_scores']

        posts_path = self.output_dir / "posts_with_topics.csv"
        posts_with_topics.to_csv(posts_path, index=False)
        logger.info(f"Saved posts with topics to {posts_path}")

        # Save topic representations
        representations_path = self.output_dir / "topic_representations.json"
        serializable_representations = {}
        for topic_id, words in analysis_results['topic_representations'].items():
            # Convert numpy array to list if needed
            if hasattr(words, 'tolist'):
                serializable_representations[str(topic_id)] = words.tolist()
            else:
                serializable_representations[str(topic_id)] = list(words)

        with open(representations_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_representations, f, indent=2)
        logger.info(f"Saved topic representations to {representations_path}")

    def _print_top2vec_summary(self, analysis_results: Dict[str, Any]):
        """Print a summary of the Top2Vec results."""
        print("\n" + "="*60)
        print("TOP2VEC TOPIC MODELING ANALYSIS SUMMARY")
        print("="*60)

        total_posts = analysis_results['total_posts']
        outlier_posts = analysis_results['outlier_posts']
        assigned_posts = analysis_results['assigned_posts']
        num_topics = analysis_results['num_topics']

        print(f"Total posts analyzed: {total_posts}")
        print(f"Outlier posts (low confidence): {outlier_posts}")
        print(f"Posts assigned to topics: {assigned_posts}")
        print(f"Number of topics discovered: {num_topics}")

        print(f"\nTop 10 Topics by Size:")
        topic_counts = analysis_results['topic_counts']
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)

        for i, (topic_id, count) in enumerate(sorted_topics[:10]):
            topic_name = analysis_results['topic_info'][analysis_results['topic_info']['Topic'] == topic_id]['Name'].iloc[0]
            percentage = (count / assigned_posts) * 100 if assigned_posts > 0 else 0
            print(f"  {i+1}. {topic_name}: {count} posts ({percentage:.1f}%)")

        print(f"\nResults saved to: {self.output_dir}")
        print("="*60)

    def run_complete_analysis(self):
        """Run the complete topic modeling pipeline using Top2Vec only."""
        logger.info("Starting complete topic modeling analysis with Top2Vec...")

        # Load and preprocess data
        self.load_data()
        texts = self.preprocess_text()

        # Run Top2Vec analysis
        logger.info("Running Top2Vec analysis...")
        top2vec_results = self.fit_top2vec_model(texts)

        if not top2vec_results:
            logger.error("Top2Vec analysis failed!")
            return

        # Analyze Top2Vec results
        analysis_results = self.analyze_top2vec_results(top2vec_results, texts)

        # Save results
        self.save_top2vec_results(analysis_results)

        logger.info("Topic modeling analysis completed successfully!")
        logger.info(f"Results saved to: {self.output_dir}")

        # Print summary
        self._print_top2vec_summary(analysis_results)

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
