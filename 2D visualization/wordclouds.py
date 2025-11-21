import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import ast
import pandas as pd

# ------------------------------
# Helper to parse keyword strings
# ------------------------------
def parse_keywords(s):
    try:
        s = s.replace("'", '"')      # normalize quotes
        s = s.replace("  ", ", ")    # help literal_eval
        return ast.literal_eval(s)
    except:
        print("Could not parse keywords properly. Raw value:", s)
        return s.strip("[]").replace("'", "").split()

# ------------------------------------------
# Function to generate word cloud for one subreddit
# ------------------------------------------
def make_subreddit_wordcloud(subreddit_name, df_posts, df_topics, out_dir="images"):
    print("\n==============================")
    print(f"Generating Wordcloud for r/{subreddit_name}")
    print("==============================")

    # Create output directory if missing
    os.makedirs(out_dir, exist_ok=True)

    # Filter posts
    df_sub = df_posts[df_posts["subreddit"] == subreddit_name]
    print(f"Posts found: {len(df_sub)}")
    if df_sub.empty:
        print(f"No posts found for subreddit: {subreddit_name}")
        return

    # Identify topic columns
    topic_columns = [c for c in df_posts.columns if c.startswith("topic_")]
    print(f" Number of topic columns: {len(topic_columns)}")

    # Count active topics (sum of True values)
    topic_counts = df_sub[topic_columns].sum().to_dict()
    active_topics = {k: v for k, v in topic_counts.items() if v > 0}
    print(f"Active topics (nonzero counts): {active_topics}")

    if len(active_topics) == 0:
        print("No active topics for this subreddit — skipping wordcloud.")
        return

    # Build weighted word list
    word_weights = Counter()
    for topic_col, weight in active_topics.items():
        topic_id = int(topic_col.split("_")[1])
        print(f"\nProcessing topic {topic_id} with weight {weight}")

        topic_row = df_topics[df_topics["Topic"] == topic_id]
        if topic_row.empty:
            print(f"No topic info found for topic ID {topic_id}")
            continue

        raw_keywords = topic_row["Name"].values[0]
        keywords = parse_keywords(raw_keywords)
        print(f" Keywords parsed: {keywords}")

        for w in keywords:
            word_weights[w] += weight

    if len(word_weights) == 0:
        print("No keywords generated — word cloud will be empty.")
        return

    # Generate wordcloud
    wc = WordCloud(
        width=1000,
        height=600,
        background_color="white"
    ).generate_from_frequencies(word_weights)

    # Save to file
    out_path = os.path.join(out_dir, f"{subreddit_name}_wordcloud.png")
    wc.to_file(out_path)
    print(f"\n Saved wordcloud to: {out_path}")

    # Show preview
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Word Cloud for r/{subreddit_name}")
    plt.show()


# ------------------------------
# Load your data
# ------------------------------
df_posts = pd.read_csv(r"results/one_hot_encoding/posts_one_hot.csv")  # posts with topic columns
df_topics = pd.read_csv(r"results\topic_modelling_output\topic_info.csv")  # topic IDs + keywords
print(df_topics.columns)

# ------------------------------
# Loop over all subreddits and generate word clouds
# ------------------------------
subreddits = df_posts['subreddit'].unique()

for sub in subreddits:
    make_subreddit_wordcloud(sub, df_posts, df_topics, out_dir="images")
