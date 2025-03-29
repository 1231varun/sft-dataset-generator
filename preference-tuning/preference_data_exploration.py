import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datasets import load_dataset, Dataset
import textstat
from tqdm.auto import tqdm
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()

# Get environment variables
HF_USERNAME = os.environ.get("HF_USERNAME", "your-username")
BASE_PREFERENCE_DATASET = os.environ.get("BASE_PREFERENCE_DATASET", f"{HF_USERNAME}/base-preference-dataset")
EVOLVED_PREFERENCE_DATASET = os.environ.get("EVOLVED_PREFERENCE_DATASET", f"{HF_USERNAME}/evolved-preference-dataset")
CRITIQUE_DATASET = os.environ.get("CRITIQUE_DATASET", f"{HF_USERNAME}/critique-preference-dataset")
MULTITURN_DATASET = os.environ.get("MULTITURN_DATASET", f"{HF_USERNAME}/multiturn-preference-dataset")
FILTERED_PREFERENCE_DATASET = os.environ.get("FILTERED_PREFERENCE_DATASET", f"{HF_USERNAME}/filtered-preference-dataset")

# Configure similarity threshold for filtering (adjust as needed)
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.8"))
MIN_SCORE_DIFF = float(os.environ.get("MIN_SCORE_DIFF", "0.5"))

# Create output directory
OUTPUT_DIR = "preference-tuning/analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output will be saved to {OUTPUT_DIR}/ directory")

# Download NLTK data if needed
nltk.download('punkt', quiet=True)

# Helper function for text analysis
def calculate_text_features(text):
    try:
        tokens = word_tokenize(text.lower())
    except:
        tokens = text.lower().split()
    
    word_counts = Counter(tokens)
    features = {
        'num_tokens': len(tokens),
        'unique_tokens': len(word_counts),
        'lexical_diversity': len(word_counts) / len(tokens) if tokens else 0
    }
    return features

# Calculate similarity between two responses
def calculate_similarity(text1, text2):
    """Calculate a simple Jaccard similarity between two texts"""
    if not text1 or not text2:
        return 0
    
    # Tokenize and convert to sets
    try:
        set1 = set(word_tokenize(text1.lower()))
        set2 = set(word_tokenize(text2.lower()))
    except:
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0

# Main analysis function
def analyze_preference_dataset():
    print("\n" + "="*50)
    print("STARTING PREFERENCE DATA EXPLORATION")
    print("="*50 + "\n")
    
    # Step 1: Load base preference dataset
    print(f"Loading base preference dataset: {BASE_PREFERENCE_DATASET}")
    try:
        base_dataset = load_dataset(BASE_PREFERENCE_DATASET, split="train")
        base_df = pd.DataFrame(base_dataset)
        print(f"Successfully loaded {len(base_df)} preference pairs")
    except Exception as e:
        print(f"Error loading base dataset: {e}")
        print("Using mock data for demonstration")
        # Create mock data
        base_df = pd.DataFrame({
            "instruction": ["Explain neural networks"] * 3,
            "response_a": ["A basic explanation of neural networks."] * 3,
            "response_b": ["A detailed explanation of neural networks with examples."] * 3,
            "chosen": ["B"] * 3
        })
    
    # Step 2: Basic statistics
    print("Computing basic statistics...")
    stats = {}
    stats["total_preference_pairs"] = len(base_df)
    stats["chosen_a_percentage"] = (base_df["chosen"] == "A").mean() * 100
    stats["chosen_b_percentage"] = (base_df["chosen"] == "B").mean() * 100
    
    # Add length statistics
    base_df["response_a_length"] = base_df["response_a"].apply(len)
    base_df["response_b_length"] = base_df["response_b"].apply(len)
    base_df["length_difference"] = base_df["response_b_length"] - base_df["response_a_length"]
    
    stats["avg_response_a_length"] = base_df["response_a_length"].mean()
    stats["avg_response_b_length"] = base_df["response_b_length"].mean()
    stats["avg_length_difference"] = base_df["length_difference"].mean()
    
    # Step 3: Text quality analysis
    print("Analyzing text quality...")
    for idx, row in tqdm(base_df.iterrows(), total=len(base_df)):
        # Calculate readability scores
        base_df.loc[idx, "readability_a"] = textstat.flesch_reading_ease(row["response_a"])
        base_df.loc[idx, "readability_b"] = textstat.flesch_reading_ease(row["response_b"])
        
        # Calculate text features
        features_a = calculate_text_features(row["response_a"])
        features_b = calculate_text_features(row["response_b"])
        
        for feature, value in features_a.items():
            base_df.loc[idx, f"a_{feature}"] = value
        
        for feature, value in features_b.items():
            base_df.loc[idx, f"b_{feature}"] = value
        
        # Calculate similarity between responses
        base_df.loc[idx, "response_similarity"] = calculate_similarity(row["response_a"], row["response_b"])
    
    # Step 4: Try to load critique dataset if available
    has_critique_data = False
    try:
        critique_dataset = load_dataset(CRITIQUE_DATASET, split="train")
        critique_df = pd.DataFrame(critique_dataset)
        print(f"Successfully loaded critique dataset with {len(critique_df)} examples")
        
        # Extract scores
        critique_df["score_a_overall"] = critique_df["score_a"].apply(
            lambda x: x["overall"] if isinstance(x, dict) else 
                      (json.loads(x)["overall"] if isinstance(x, str) else 0)
        )
        critique_df["score_b_overall"] = critique_df["score_b"].apply(
            lambda x: x["overall"] if isinstance(x, dict) else 
                      (json.loads(x)["overall"] if isinstance(x, str) else 0)
        )
        critique_df["score_difference"] = critique_df["score_b_overall"] - critique_df["score_a_overall"]
        
        # Add stats
        stats["avg_score_a"] = critique_df["score_a_overall"].mean()
        stats["avg_score_b"] = critique_df["score_b_overall"].mean()
        stats["avg_score_difference"] = critique_df["score_difference"].mean()
        
        has_critique_data = True
    except Exception as e:
        print(f"Critique dataset not available: {e}")
    
    # Step 5: Create visualizations
    print("Creating visualizations...")
    
    # Plot length distributions
    plt.figure(figsize=(10, 6))
    length_plot_df = pd.DataFrame({
        "response_length": pd.concat([base_df["response_a_length"], base_df["response_b_length"]]),
        "response_type": ["Response A"] * len(base_df) + ["Response B"] * len(base_df)
    })
    sns.histplot(data=length_plot_df, x="response_length", hue="response_type", element="step")
    plt.title("Distribution of Response Lengths")
    plt.xlabel("Length (characters)")
    plt.savefig(os.path.join(OUTPUT_DIR, "length_distributions.png"))
    
    # Plot readability comparison
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x="readability_a", y="readability_b", hue="chosen", data=base_df)
    plt.plot([0, 100], [0, 100], 'k--', alpha=0.5)
    plt.title("Readability Comparison")
    plt.xlabel("Response A Readability")
    plt.ylabel("Response B Readability")
    
    plt.subplot(1, 2, 2)
    readability_diff = base_df["readability_b"] - base_df["readability_a"]
    plot_df = pd.DataFrame({
        "readability_diff": readability_diff,
        "chosen": base_df["chosen"]
    })
    sns.histplot(data=plot_df, x="readability_diff", hue="chosen")
    plt.title("Readability Difference Distribution (B - A)")
    plt.xlabel("Difference in Readability Score")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "readability_comparison.png"))
    
    # Plot similarity distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(data=base_df, x="response_similarity")
    plt.axvline(SIMILARITY_THRESHOLD, color='r', linestyle='--', label=f'Threshold: {SIMILARITY_THRESHOLD}')
    plt.title("Response Similarity Distribution")
    plt.xlabel("Jaccard Similarity")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "response_similarity.png"))
    
    # Plot critique scores if available
    if has_critique_data:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        
        # Create a proper dataframe for the overall scores
        scores_plot_df = pd.DataFrame({
            "overall_score": pd.concat([critique_df["score_a_overall"], critique_df["score_b_overall"]]),
            "response_type": ["Response A"] * len(critique_df) + ["Response B"] * len(critique_df)
        })
        sns.histplot(data=scores_plot_df, x="overall_score", hue="response_type")
        plt.title("Distribution of Overall Scores")
        plt.xlabel("Overall Score")
        
        plt.subplot(1, 2, 2)
        sns.histplot(data=critique_df, x="score_difference", hue="chosen")
        plt.title("Score Difference Distribution (B - A)")
        plt.xlabel("Difference in Overall Score")
        plt.axvline(MIN_SCORE_DIFF, color='r', linestyle='--', label=f'Min Difference: {MIN_SCORE_DIFF}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "critique_scores.png"))
    
    # Step 6: Filter the dataset
    print("Filtering preference dataset...")
    
    # Remove pairs with high similarity
    filtered_df = base_df[base_df["response_similarity"] < SIMILARITY_THRESHOLD].copy()
    
    # Keep significant length differences (chosen response should be longer in most cases)
    for idx, row in filtered_df.iterrows():
        if row["chosen"] == "A" and row["response_a_length"] < row["response_b_length"] * 0.7:
            filtered_df.loc[idx, "flag_length_inconsistency"] = True
        elif row["chosen"] == "B" and row["response_b_length"] < row["response_a_length"] * 0.7:
            filtered_df.loc[idx, "flag_length_inconsistency"] = True
        else:
            filtered_df.loc[idx, "flag_length_inconsistency"] = False
    
    # Apply criterion from critique data if available
    if has_critique_data:
        # Merge critique data with filtered data
        filtered_df = filtered_df.merge(
            critique_df[["instruction", "score_a_overall", "score_b_overall", "score_difference"]], 
            on="instruction", 
            how="left"
        )
        
        # Flag inconsistent scores
        filtered_df["flag_score_inconsistency"] = False
        for idx, row in filtered_df.iterrows():
            if not pd.isna(row["score_difference"]):
                if row["chosen"] == "A" and row["score_difference"] > 0:
                    filtered_df.loc[idx, "flag_score_inconsistency"] = True
                elif row["chosen"] == "B" and row["score_difference"] < 0:
                    filtered_df.loc[idx, "flag_score_inconsistency"] = True
    
    # Create final filtered dataset
    if has_critique_data:
        final_filtered_df = filtered_df[
            ~filtered_df["flag_length_inconsistency"] & 
            ~filtered_df["flag_score_inconsistency"] &
            (filtered_df["response_similarity"] < SIMILARITY_THRESHOLD)
        ].copy()
    else:
        final_filtered_df = filtered_df[
            ~filtered_df["flag_length_inconsistency"] &
            (filtered_df["response_similarity"] < SIMILARITY_THRESHOLD)
        ].copy()
    
    # If filtering was too strict, use a subset of the original data
    if len(final_filtered_df) < 10:
        print("Warning: Filtering was too strict, only found", len(final_filtered_df), "examples")
        print("Using top examples by readability difference")
        
        # Sort by readability difference that matches the chosen response
        base_df["effective_readability_diff"] = base_df.apply(
            lambda row: row["readability_b"] - row["readability_a"] if row["chosen"] == "B" else row["readability_a"] - row["readability_b"], 
            axis=1
        )
        final_filtered_df = base_df.sort_values("effective_readability_diff", ascending=False).head(20)
    
    # Save filtered dataset
    print(f"Saving filtered dataset with {len(final_filtered_df)} examples")
    final_filtered_dataset = Dataset.from_pandas(final_filtered_df)
    final_filtered_dataset.push_to_hub(FILTERED_PREFERENCE_DATASET)
    
    # Save to CSV for easy inspection
    final_filtered_df.to_csv(os.path.join(OUTPUT_DIR, "filtered_preference_dataset.csv"), index=False)
    
    # Step 7: Generate analysis report
    print("Generating analysis report...")
    with open(os.path.join(OUTPUT_DIR, "preference_analysis_results.md"), "w") as f:
        f.write("# Preference Dataset Analysis Results\n\n")
        
        f.write("## Basic Statistics\n\n")
        for k, v in stats.items():
            if isinstance(v, float):
                f.write(f"- **{k}**: {v:.2f}\n")
            else:
                f.write(f"- **{k}**: {v}\n")
        
        f.write("\n## Response Length Analysis\n\n")
        f.write("![Length Distributions](length_distributions.png)\n\n")
        f.write(f"Average length of chosen responses: {base_df.apply(lambda row: row['response_a_length'] if row['chosen'] == 'A' else row['response_b_length'], axis=1).mean():.2f} characters\n")
        f.write(f"Average length of rejected responses: {base_df.apply(lambda row: row['response_b_length'] if row['chosen'] == 'A' else row['response_a_length'], axis=1).mean():.2f} characters\n\n")
        
        f.write("\n## Readability Analysis\n\n")
        f.write("![Readability Comparison](readability_comparison.png)\n\n")
        f.write(f"Average readability of chosen responses: {base_df.apply(lambda row: row['readability_a'] if row['chosen'] == 'A' else row['readability_b'], axis=1).mean():.2f}\n")
        f.write(f"Average readability of rejected responses: {base_df.apply(lambda row: row['readability_b'] if row['chosen'] == 'A' else row['readability_a'], axis=1).mean():.2f}\n\n")
        
        f.write("\n## Response Similarity Analysis\n\n")
        f.write("![Response Similarity](response_similarity.png)\n\n")
        f.write(f"Average similarity between responses: {base_df['response_similarity'].mean():.2f}\n")
        f.write(f"Percentage of examples with high similarity (>{SIMILARITY_THRESHOLD}): {(base_df['response_similarity'] > SIMILARITY_THRESHOLD).mean() * 100:.2f}%\n\n")
        
        if has_critique_data:
            f.write("\n## Critique Analysis\n\n")
            f.write("![Critique Scores](critique_scores.png)\n\n")
            f.write(f"Average score of chosen responses: {critique_df.apply(lambda row: row['score_a_overall'] if row['chosen'] == 'A' else row['score_b_overall'], axis=1).mean():.2f}\n")
            f.write(f"Average score of rejected responses: {critique_df.apply(lambda row: row['score_b_overall'] if row['chosen'] == 'A' else row['score_a_overall'], axis=1).mean():.2f}\n\n")
        
        f.write("\n## Filtering Results\n\n")
        f.write(f"Original dataset size: {len(base_df)}\n")
        f.write(f"Filtered dataset size: {len(final_filtered_df)}\n")
        f.write(f"Percentage retained: {len(final_filtered_df)/len(base_df)*100:.2f}%\n\n")
        
        f.write("### Filtering Criteria\n\n")
        f.write(f"- Response similarity < {SIMILARITY_THRESHOLD}\n")
        f.write(f"- Length consistency with preference choice\n")
        if has_critique_data:
            f.write(f"- Score consistency with preference choice\n")
            f.write(f"- Minimum score difference: {MIN_SCORE_DIFF}\n")
    
    # Step 8: Try to analyze multi-turn data if available
    try:
        multiturn_dataset = load_dataset(MULTITURN_DATASET, split="train")
        multiturn_df = pd.DataFrame(multiturn_dataset)
        print(f"Successfully loaded multi-turn dataset with {len(multiturn_df)} examples")
        
        # Create visualization for multi-turn analysis
        plt.figure(figsize=(8, 6))
        multiturn_df["chosen_count"] = multiturn_df["chosen"].value_counts().to_dict()
        plt.pie([multiturn_df["chosen"].value_counts().get("A", 0), 
                 multiturn_df["chosen"].value_counts().get("B", 0)], 
                labels=["Conversation A", "Conversation B"], 
                autopct='%1.1f%%')
        plt.title("Multi-turn Preference Distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "multiturn_distribution.png"))
        
        # Add to the report
        with open(os.path.join(OUTPUT_DIR, "preference_analysis_results.md"), "a") as f:
            f.write("\n## Multi-turn Conversation Analysis\n\n")
            f.write("![Multi-turn Distribution](multiturn_distribution.png)\n\n")
            f.write(f"Total multi-turn conversations: {len(multiturn_df)}\n")
            f.write(f"Percentage preferring conversation A: {multiturn_df['chosen'].value_counts().get('A', 0)/len(multiturn_df)*100:.2f}%\n")
            f.write(f"Percentage preferring conversation B: {multiturn_df['chosen'].value_counts().get('B', 0)/len(multiturn_df)*100:.2f}%\n")
    
    except Exception as e:
        print(f"Multi-turn dataset not available or error in analysis: {e}")
    
    print("\n" + "="*50)
    print("PREFERENCE DATA EXPLORATION COMPLETE")
    print(f"All output has been saved to the {OUTPUT_DIR}/ directory")
    print("="*50 + "\n")

if __name__ == "__main__":
    analyze_preference_dataset() 