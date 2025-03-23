import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import textstat
from dataset_tools.models import ClassifierPipeline
from tqdm.auto import tqdm
import nomic
from text_descriptives.descriptives import fd_tokens_basic

# Load the dataset
dataset = load_dataset("yourusername/sft-complete-dataset", split="train")
df = pd.DataFrame(dataset)

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
    
    # Multi-turn stats (if applicable)
    if "conversation" in df.columns:
        stats["multi_turn_count"] = df["conversation"].apply(lambda x: len(x["messages"]) > 2).sum()
    
    return stats

# 2. Content Analysis
def content_analysis(df):
    # Load domain classifier
    domain_classifier = ClassifierPipeline("dataset_tools/domain_classifier")
    
    # Get domains for instructions
    domains = []
    for instr in tqdm(df["instruction"]):
        domains.append(domain_classifier(instr)[0]["label"])
    
    df["domain"] = domains
    
    # Count domains
    domain_counts = df["domain"].value_counts()
    
    # Complexity analysis
    df["readability_score"] = df["instruction"].apply(lambda x: textstat.flesch_reading_ease(x))
    df["complexity_score"] = df["instruction"].apply(lambda x: textstat.text_standard(x, float_output=True))
    
    return df, domain_counts

# 3. Embedding Analysis
def embedding_analysis(df, sample_size=1000):
    # Sample for efficiency
    if len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df
    
    # Use fast embedder
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
    
    # Get embeddings
    embeddings = []
    
    with torch.no_grad():
        for text in tqdm(df_sample["instruction"]):
            inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(embedding)
    
    # Dimensionality reduction
    pca = PCA(n_components=50)
    embeddings_pca = pca.fit_transform(embeddings)
    
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings_pca)
    
    df_sample["tsne_x"] = embeddings_tsne[:, 0]
    df_sample["tsne_y"] = embeddings_tsne[:, 1]
    
    return df_sample

# 4. Toxicity and Quality Analysis
def quality_analysis(df):
    # Load toxicity classifier
    toxicity_classifier = ClassifierPipeline("dataset_tools/toxicity_classifier")
    
    # Analyze toxicity
    toxicity_scores = []
    for text in tqdm(df["instruction"]):
        score = toxicity_classifier(text)[0]["score"]
        toxicity_scores.append(score)
    
    df["toxicity_score"] = toxicity_scores
    
    # Flag potentially problematic examples
    df["is_toxic"] = df["toxicity_score"] > 0.7
    
    return df

# 5. Visualizations
def create_visualizations(df, domain_counts, df_sample):
    # 1. Domain distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(x=domain_counts.index, y=domain_counts.values)
    plt.xticks(rotation=45, ha="right")
    plt.title("Domain Distribution")
    plt.tight_layout()
    plt.savefig("domain_distribution.png")
    
    # 2. Length distributions
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df["instruction"].apply(len), bins=30)
    plt.title("Instruction Length Distribution")
    
    plt.subplot(1, 2, 2)
    sns.histplot(df["response"].apply(len), bins=30)
    plt.title("Response Length Distribution")
    
    plt.tight_layout()
    plt.savefig("length_distributions.png")
    
    # 3. Embedding visualization
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df_sample["tsne_x"], df_sample["tsne_y"], c=df_sample["complexity_score"], cmap="viridis", alpha=0.6)
    plt.colorbar(scatter, label="Complexity Score")
    plt.title("Instruction Embedding Space (t-SNE)")
    plt.savefig("embedding_visualization.png")
    
    # 4. Readability vs. domain
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="domain", y="readability_score", data=df)
    plt.xticks(rotation=45, ha="right")
    plt.title("Readability Score by Domain")
    plt.tight_layout()
    plt.savefig("readability_by_domain.png")
    
    # 5. Nomic Atlas visualization (if you want to provide interactive visualization)
    try:
        atlas = nomic.map(df_sample["instruction"].tolist(), id_field="index")
        atlas.deploy("sft-instruction-atlas")
    except:
        print("Skipping Nomic Atlas visualization")

# Main analysis function
def analyze_dataset():
    dataset = load_dataset("yourusername/sft-complete-dataset", split="train")
    df = pd.DataFrame(dataset)
    
    print("Computing basic statistics...")
    stats = basic_stats(df)
    
    print("Analyzing content...")
    df, domain_counts = content_analysis(df)
    
    print("Computing embeddings and visualizing...")
    df_sample = embedding_analysis(df)
    
    print("Checking for toxicity and quality issues...")
    df = quality_analysis(df)
    
    print("Creating visualizations...")
    create_visualizations(df, domain_counts, df_sample)
    
    # Create filtered version based on analysis
    df_filtered = df[
        (df["toxicity_score"] < 0.3) &  # Remove toxic content
        (df["instruction"].apply(len) > 20) &  # Remove very short instructions
        (df["response"].apply(len) > 50) &  # Remove very short responses
        (df["readability_score"] > 30)  # Ensure reasonable readability
    ]
    
    # Save filtered dataset
    filtered_dataset = Dataset.from_pandas(df_filtered)
    filtered_dataset.push_to_hub("yourusername/sft-filtered-dataset")
    
    # Save analysis results
    with open("analysis_results.md", "w") as f:
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
        f.write("- Toxicity score < 0.3\n")
        f.write("- Instruction length > 20 characters\n")
        f.write("- Response length > 50 characters\n")
        f.write("- Readability score > 30\n")

if __name__ == "__main__":
    analyze_dataset() 