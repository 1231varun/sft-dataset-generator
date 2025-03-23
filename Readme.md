# SFT Instruction Dataset Generation Project

This project creates high-quality datasets for supervised fine-tuning (SFT) of large language models. It implements multiple generation techniques including SelfInstruct, EvolInstruct, and Magpie to create diverse, high-quality instruction-response pairs and multi-turn conversations.

## Project Overview

This repository contains code to:
1. Generate instruction datasets using multiple techniques
2. Analyze and filter the generated data
3. Visualize dataset properties
4. Push the datasets to the Hugging Face Hub

## Installation

### Requirements
- Python 3.9+
- PyTorch
- An account with Hugging Face (free)
- API access to OpenAI and/or Anthropic (optional but recommended)

### Setup

1. Clone this repository:
```bash
git clone https://github.com/1231varun/sft-dataset-generator.git
cd sft-dataset-generator
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up API keys for model providers (if using them):
```bash
# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# For Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

On Windows, use `set` instead of `export`.

5. Login to Hugging Face:
```bash
huggingface-cli login
```

## Step-by-Step Execution Guide

### 1. Customize the configuration

First, make sure to customize the settings in both files:

- In `instruction_dataset_generator.py`:
  - Update seed instructions
  - Set your Hugging Face username
  - Adjust model selection if needed
  - Modify number of examples to generate

- In `data_exploration.py`:
  - Set your Hugging Face username
  - Adjust filtering criteria if needed

### 2. Generate the Dataset

Run the instruction dataset generator:

```bash
python instruction_dataset_generator.py
```

This will:
- Generate instructions using SelfInstruct technique
- Evolve instructions using EvolInstruct
- Generate responses for instructions
- Create multi-turn conversations
- Push intermediate datasets to Hugging Face Hub

Note: The full generation may take several hours and might incur API costs if using commercial models.

### 3. Analyze and Filter the Dataset

Run the data analysis script:

```bash
python data_exploration.py
```

This will:
- Calculate dataset statistics
- Analyze content for domains, complexity, and toxicity
- Generate visualizations
- Create a filtered version of the dataset
- Push the filtered dataset to Hugging Face Hub

### 4. Review the Results

After running both scripts:

1. Check the terminal output for any errors
2. Review the generated visualizations in the project directory
3. Read the `analysis_results.md` file for a summary of findings
4. Visit your Hugging Face profile to see the published datasets

## Dataset Evaluation

To evaluate the quality of your dataset:

### 1. Basic Metrics Review

Check the `analysis_results.md` file for:
- Total number of examples
- Distribution across domains
- Length distributions
- Toxicity levels
- Filtering statistics

### 2. Sample Review

Review random samples from your dataset for quality:

```python
from datasets import load_dataset

# Load your dataset
dataset = load_dataset("yourusername/sft-filtered-dataset", split="train")

# Display random samples
for i in range(5):
    idx = random.randint(0, len(dataset) - 1)
    print(f"Sample {i+1}:")
    print(f"Instruction: {dataset[idx]['instruction']}")
    print(f"Response: {dataset[idx]['response']}")
    print("---")
```

### 3. Domain Distribution

Check if your dataset has good coverage across domains:

```python
from collections import Counter
import matplotlib.pyplot as plt

# Count domains
domains = [example['domain'] for example in dataset]
domain_counts = Counter(domains)

# Plot domain distribution
plt.figure(figsize=(12, 6))
plt.bar(domain_counts.keys(), domain_counts.values())
plt.xticks(rotation=45, ha="right")
plt.title("Domain Distribution")
plt.tight_layout()
plt.show()
```

## Pushing to Hugging Face Hub

The scripts automatically push datasets to the Hugging Face Hub. If you want to manually push updates:

```python
from datasets import load_dataset
from huggingface_hub import HfApi

# Update dataset card
api = HfApi()
api.upload_file(
    path_or_fileobj="dataset_card.md",
    path_in_repo="README.md",
    repo_id="yourusername/sft-filtered-dataset",
    repo_type="dataset"
)

# Upload visualizations
for img in ["domain_distribution.png", "length_distributions.png", "embedding_visualization.png"]:
    api.upload_file(
        path_or_fileobj=img,
        path_in_repo=img,
        repo_id="yourusername/sft-filtered-dataset",
        repo_type="dataset"
    )
```

## Customization Options

### Modify Seed Instructions

Update the `seed_instructions` list in `instruction_dataset_generator.py` to focus on specific domains or instruction types.

### Change Models

Modify the model selection in each pipeline to use different providers or model variants:

```python
# For example, to use a different OpenAI model:
llm = OpenAILLM(model="gpt-3.5-turbo-instruct")

# Or to use a local model:
llm = TransformersLLM(model="meta-llama/Llama-3-8B-Instruct")
```

### Adjust Filtering Criteria

Modify the filtering criteria in `data_exploration.py` to be more or less strict:

```python
df_filtered = df[
    (df["toxicity_score"] < 0.5) &  # More permissive toxicity threshold
    (df["instruction"].apply(len) > 10) &  # Allow shorter instructions
    (df["response"].apply(len) > 30) &  # Allow shorter responses
    (df["readability_score"] > 20)  # Lower readability requirement
]
```

## Troubleshooting

### API Rate Limits

If you encounter rate limit errors from API providers:
- Add delays between requests
- Reduce batch size
- Switch to different models or providers

### Memory Issues

If you encounter memory errors during embedding or analysis:
- Reduce sample size in `embedding_analysis`
- Process data in smaller batches
- Use a more memory-efficient embedding model

### Model Loading Errors

If you have issues loading large models:
- Use smaller models like `meta-llama/Llama-3-8B-Instruct` instead of larger variants
- Add quantization options to reduce memory usage

## Contributing

Contributions to improve the dataset generation or analysis are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the Apache 2.0 License.