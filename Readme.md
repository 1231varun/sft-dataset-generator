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
- Python 3.9-3.11 (Python 3.12 may have compatibility issues with some packages)
- PyTorch
- An account with Hugging Face (free)
- API access to OpenAI and/or Anthropic (optional but recommended)

### macOS-Specific Setup

If you're using macOS and encounter the `ModuleNotFoundError: No module named '_lzma'` error:

1. Install the xz library using Homebrew:
```bash
brew install xz
```

2. If using pyenv, reinstall Python with the proper dependencies:
```bash
# Uninstall the current Python version
pyenv uninstall 3.12.2

# Install with the proper dependencies
LDFLAGS="-L$(brew --prefix xz)/lib" CPPFLAGS="-I$(brew --prefix xz)/include" pyenv install 3.12.2
```

3. Alternatively, if using the system Python, you may need to install Python from python.org with proper dependencies.

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

If you're using Python 3.12 and encounter issues with some packages, consider:
```bash
# Create a virtual environment with Python 3.11 instead
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

4. If you're using Python 3.12, you'll need to install spaCy models manually:
```bash
python -m spacy download en_core_web_sm
```

5. Set up API keys for model providers (if using them):
```bash
# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# For Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

On Windows, use `set` instead of `export`.

6. Login to Hugging Face:
```bash
huggingface-cli login
```

## Troubleshooting

### Missing LZMA Module
If you encounter `ModuleNotFoundError: No module named '_lzma'`:

1. This is a common issue with Python installations on macOS when the xz library is missing
2. Follow the macOS-specific setup instructions above
3. After reinstalling Python, recreate your virtual environment:
```bash
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Other Common Issues

If you're having issues with package installations:

1. Ensure you have the latest pip: `pip install --upgrade pip`
2. Try installing packages one by one to identify problematic dependencies
3. Check system dependencies required by numerical packages:
```bash
# On macOS
brew install cmake libomp

# On Ubuntu
sudo apt-get install build-essential cmake
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

## Contributing

Contributions to improve the dataset generation or analysis are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the Apache 2.0 License.

# Environment Variables

This project uses environment variables for configuration. You can set them in your environment or use a `.env` file.

1. Copy the example environment file to create your own:
```bash
cp .env.example .env
```

2. Edit the `.env` file with your own values:
```bash
# Set your Hugging Face username
HF_USERNAME=your-username

# Set your API keys
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
```

3. Install python-dotenv:
```bash
pip install python-dotenv
```

4. The application will automatically load variables from the `.env` file when it runs.

## Available Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| HF_USERNAME | Your Hugging Face username | 1231varun |
| OPENAI_MODEL | OpenAI model to use | gpt-4o |
| ANTHROPIC_MODEL | Anthropic model to use | claude-3-opus-20240229 |
| TRANSFORMERS_MODEL | Local transformer model | meta-llama/Llama-3-8B-Instruct |
| OPENAI_API_KEY | OpenAI API key | (required) |
| ANTHROPIC_API_KEY | Anthropic API key | (required) |
| DATASET_NAME | Name of the complete dataset | sft-complete-dataset |
| FILTERED_DATASET_NAME | Name of the filtered dataset | sft-filtered-dataset |
| TOXICITY_THRESHOLD | Maximum toxicity score for filtered data | 0.3 |
| MIN_INSTRUCTION_LENGTH | Minimum instruction length | 20 |
| MIN_RESPONSE_LENGTH | Minimum response length | 50 |
| MIN_READABILITY_SCORE | Minimum readability score | 30 |
| CONVERSATION_SAMPLE_SIZE | Number of examples to use for conversations | 500 |