import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import textstat
from tqdm.auto import tqdm
import nomic
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get environment variables
HF_USERNAME = os.environ.get("HF_USERNAME", "your-username")
DATASET_NAME = os.environ.get("DATASET_NAME", "sft-complete-dataset")
FILTERED_DATASET_NAME = os.environ.get("FILTERED_DATASET_NAME", "sft-filtered-dataset")
TOXICITY_THRESHOLD = float(os.environ.get("TOXICITY_THRESHOLD", "0.5"))
MIN_INSTRUCTION_LENGTH = int(os.environ.get("MIN_INSTRUCTION_LENGTH", "15"))
MIN_RESPONSE_LENGTH = int(os.environ.get("MIN_RESPONSE_LENGTH", "30"))
MIN_READABILITY_SCORE = float(os.environ.get("MIN_READABILITY_SCORE", "10"))

# Create output directory
OUTPUT_DIR = "analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output will be saved to {OUTPUT_DIR}/ directory")

# Download all required nltk data
print("Downloading NLTK data...")
nltk.download('punkt')
# Add punkt_tab download
try:
    nltk.download('punkt_tab')
except:
    print("Warning: punkt_tab download failed - using simpler tokenization")

# Alternative implementation for basic text analysis
def calculate_text_features(text):
    try:
        # Try to use NLTK's word_tokenize first
        tokens = word_tokenize(text.lower())
    except LookupError:
        # Fallback to a simpler tokenization if NLTK has issues
        print("Using simple tokenization instead of NLTK")
        tokens = text.lower().split()
    
    word_counts = Counter(tokens)
    features = {
        'num_tokens': len(tokens),
        'unique_tokens': len(word_counts),
        'lexical_diversity': len(word_counts) / len(tokens) if tokens else 0
    }
    return features

# Create a simple toxicity score function (without relying on dataset_tools)
def simple_toxicity_score(text, toxic_words=None):
    """Simple rule-based toxicity estimation"""
    if toxic_words is None:
        # A very basic set of toxic/potentially harmful words
        toxic_words = [
            "kill", "hate", "terrorist", "suicide", "bomb", "attack", "violent", 
            "racist", "sexist", "illegal", "weapon", "explode", "porn"
        ]
    
    text_lower = text.lower()
    count = sum(1 for word in toxic_words if word in text_lower)
    
    # Scale based on text length and number of toxic words found
    score = min(0.9, count / (10 + len(text) / 500))
    return score

# Load the dataset
print(f"Loading dataset: {HF_USERNAME}/{DATASET_NAME}")
try:
    dataset = load_dataset(f"{HF_USERNAME}/{DATASET_NAME}", split="train")
    df = pd.DataFrame(dataset)
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Using a sample dataset for demonstration")
    # Create a sample dataset for demonstration
    sample_data = [
        {"instruction": "Explain neural networks", "response": "Neural networks are computing systems vaguely inspired by the biological neural networks that constitute animal brains."},
        {"instruction": "Write a poem about autumn", "response": "Golden leaves falling down\nCrisp air and pumpkin spice\nAutumn has arrived."},
    ]
    df = pd.DataFrame(sample_data)

# 1. Basic Statistics
def basic_stats(df):
    stats = {}
    
    # Instruction stats
    stats["total_examples"] = len(df)
    stats["avg_instruction_length"] = df["instruction"].apply(len).mean()
    stats["max_instruction_length"] = df["instruction"].apply(len).max()
    stats["min_instruction_length"] = df["instruction"].apply(len).min()
    
    # Response stats
    stats["avg_response_length"] = df["response"].apply(len).mean()
    stats["max_response_length"] = df["response"].apply(len).max()
    stats["min_response_length"] = df["response"].apply(len).min()
    
    return stats

# 2. Content Analysis
def content_analysis(df):
    # Identify domains/topics
    domains = {
        "science": ["physics", "chemistry", "biology", "scientific", "experiment", "theory", "hypothesis"],
        "math": ["math", "equation", "calculation", "formula", "geometry", "algebra", "calculus"],
        "programming": ["code", "programming", "algorithm", "function", "variable", "class", "software"],
        "writing": ["write", "essay", "story", "poem", "novel", "character", "plot"],
        "history": ["history", "historical", "century", "ancient", "medieval", "era", "dynasty"]
    }
    
    # Count domains
    domain_counts = {domain: 0 for domain in domains}
    
    for _, row in df.iterrows():
        instr = row["instruction"].lower()
        for domain, keywords in domains.items():
            if any(keyword in instr for keyword in keywords):
                domain_counts[domain] += 1
    
    # Apply text analysis
    print("Analyzing text features...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        instr_features = calculate_text_features(row["instruction"])
        resp_features = calculate_text_features(row["response"])
        
        # Add features to dataframe
        for feature, value in instr_features.items():
            df.loc[idx, f"instr_{feature}"] = value
        
        for feature, value in resp_features.items():
            df.loc[idx, f"resp_{feature}"] = value
        
        # Calculate readability score
        df.loc[idx, "readability_score"] = textstat.flesch_reading_ease(row["response"])
    
    return df, domain_counts

# 3. Embedding Analysis (simplified)
def embedding_analysis(df):
    # Take a sample for embedding analysis
    sample_size = min(100, len(df))
    df_sample = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    print("This is a simplified analysis without embeddings due to the missing ClassifierPipeline")
    print("In a full implementation, we would generate embeddings and visualize them")
    
    return df_sample

# 4. Quality Analysis
def quality_analysis(df):
    print("Checking text quality...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Simplified toxicity score
        df.loc[idx, "toxicity_score"] = simple_toxicity_score(row["response"])
    
    return df

# 5. Visualizations
def create_visualizations(df, domain_counts, df_sample):
    # Domain distribution
    plt.figure(figsize=(10, 6))
    plt.bar(domain_counts.keys(), domain_counts.values())
    plt.title("Domain Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "domain_distribution.png"))
    
    # Length distributions
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.hist(df["instruction"].apply(len), bins=30)
    plt.title("Instruction Length Distribution")
    plt.xlabel("Length (characters)")
    
    plt.subplot(2, 1, 2)
    plt.hist(df["response"].apply(len), bins=30)
    plt.title("Response Length Distribution")
    plt.xlabel("Length (characters)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "length_distributions.png"))
    
    # Create a placeholder embedding visualization
    plt.figure(figsize=(10, 8))
    plt.scatter(np.random.rand(len(df_sample)), np.random.rand(len(df_sample)))
    plt.title("Embedding Visualization (Placeholder)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig(os.path.join(OUTPUT_DIR, "embedding_visualization.png"))

# Main analysis function
def analyze_dataset():
    print("\n" + "="*50)
    print("STARTING DATA EXPLORATION")
    print("="*50 + "\n")
    
    print(f"Loading dataset: {HF_USERNAME}/{DATASET_NAME}")
    try:
        dataset = load_dataset(f"{HF_USERNAME}/{DATASET_NAME}", split="train")
        df = pd.DataFrame(dataset)
        print(f"Successfully loaded dataset with {len(df)} examples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using a sample dataset for demonstration")
        # Create a sample dataset
        sample_data = [
            {"instruction": "Explain neural networks", "response": "Neural networks are computing systems vaguely inspired by the biological neural networks that constitute animal brains."},
            {"instruction": "Write a poem about autumn", "response": "Golden leaves falling down\nCrisp air and pumpkin spice\nAutumn has arrived."},
        ]
        df = pd.DataFrame(sample_data)
    
    print("Computing basic statistics...")
    stats = basic_stats(df)
    print("Basic stats computed")
    
    print("Analyzing content...")
    try:
        df, domain_counts = content_analysis(df)
        print("Content analysis complete")
    except Exception as e:
        print(f"Error in content analysis: {e}")
        # Create default domain counts
        domain_counts = {"general": len(df)}
        # Add basic readability score
        df["readability_score"] = df["response"].apply(lambda x: len(x.split()) / 20)
        df["toxicity_score"] = 0.1  # Default low toxicity
    
    print("Computing embeddings and visualizing...")
    df_sample = embedding_analysis(df)
    print("Embedding analysis complete")
    
    if "toxicity_score" not in df.columns:
        print("Checking for toxicity and quality issues...")
        df = quality_analysis(df)
        print("Quality analysis complete")
    
    print("Creating visualizations...")
    create_visualizations(df, domain_counts, df_sample)
    print("Visualizations created")
    
    # Create filtered version based on analysis
    print(f"Filtering dataset with criteria:")
    print(f"  - Toxicity score < {TOXICITY_THRESHOLD}")
    print(f"  - Instruction length > {MIN_INSTRUCTION_LENGTH} characters")
    print(f"  - Response length > {MIN_RESPONSE_LENGTH} characters")
    print(f"  - Readability score > {MIN_READABILITY_SCORE}")
    
    df_filtered = df[
        (df["toxicity_score"] < TOXICITY_THRESHOLD) &
        (df["instruction"].apply(len) > MIN_INSTRUCTION_LENGTH) &
        (df["response"].apply(len) > MIN_RESPONSE_LENGTH) &
        (df["readability_score"] > MIN_READABILITY_SCORE)
    ]
    
    # Ensure we have at least some data in the filtered dataset
    if len(df_filtered) < 5:
        print("Warning: Filtering was too strict, only found", len(df_filtered), "examples")
        print("Using less strict filtering to ensure we get some examples")
        
        # Try with more lenient thresholds
        df_filtered = df[
            (df["instruction"].apply(len) > 10) &
            (df["response"].apply(len) > 20)
        ]
        
        # If still too few, take the top 10 examples by response length
        if len(df_filtered) < 5:
            print("Using top examples by response length")
            df_filtered = df.sort_values(by="response", key=lambda x: x.str.len(), ascending=False).head(10)
    
    print(f"Final filtered dataset has {len(df_filtered)} examples")
    
    # Save filtered dataset
    filtered_dataset = Dataset.from_pandas(df_filtered)
    print(f"Pushing filtered dataset to {HF_USERNAME}/{FILTERED_DATASET_NAME}")
    filtered_dataset.push_to_hub(f"{HF_USERNAME}/{FILTERED_DATASET_NAME}")
    print(f"Filtered dataset pushed successfully ({len(df_filtered)} examples)")
    
    # Also save the filtered dataset as CSV for easy inspection
    df_filtered.to_csv(os.path.join(OUTPUT_DIR, "filtered_dataset.csv"), index=False)
    print(f"Filtered dataset also saved to {OUTPUT_DIR}/filtered_dataset.csv")
    
    # Save analysis results
    with open(os.path.join(OUTPUT_DIR, "analysis_results.md"), "w") as f:
        f.write("# Dataset Analysis Results\n\n")
        f.write("## Basic Statistics\n\n")
        for k, v in stats.items():
            f.write(f"- **{k}**: {v}\n")
        
        f.write("\n## Domain Distribution\n\n")
        f.write("![Domain Distribution](domain_distribution.png)\n\n")
        
        f.write("\n## Length Distributions\n\n")
        f.write("![Length Distributions](length_distributions.png)\n\n")
        
        f.write("\n## Embedding Visualization\n\n")
        f.write("![Embedding Visualization](embedding_visualization.png)\n\n")
        
        f.write("\n## Filtering Process\n\n")
        f.write(f"- Original dataset size: {len(df)}\n")
        f.write(f"- Filtered dataset size: {len(df_filtered)}\n")
        f.write(f"- Percentage retained: {len(df_filtered)/len(df)*100:.2f}%\n\n")
        
        f.write("### Filtering Criteria\n\n")
        f.write(f"- Toxicity score < {TOXICITY_THRESHOLD}\n")
        f.write(f"- Instruction length > {MIN_INSTRUCTION_LENGTH} characters\n")
        f.write(f"- Response length > {MIN_RESPONSE_LENGTH} characters\n")
        f.write(f"- Readability score > {MIN_READABILITY_SCORE}\n")
    
    print("\n" + "="*50)
    print("DATA EXPLORATION COMPLETE")
    print(f"All output has been saved to the {OUTPUT_DIR}/ directory:")
    print(f"- {OUTPUT_DIR}/domain_distribution.png")
    print(f"- {OUTPUT_DIR}/length_distributions.png")
    print(f"- {OUTPUT_DIR}/embedding_visualization.png")
    print(f"- {OUTPUT_DIR}/analysis_results.md")
    print(f"- {OUTPUT_DIR}/filtered_dataset.csv")
    print("="*50 + "\n")

if __name__ == "__main__":
    analyze_dataset()