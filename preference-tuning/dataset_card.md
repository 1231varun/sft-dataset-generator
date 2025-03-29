# Preference Tuning Dataset

## Dataset Description

This dataset contains high-quality preference pairs for fine-tuning language models with reinforcement learning from human feedback (RLHF) techniques. Each example contains an instruction and two alternative responses, with a label indicating which response is preferred.

## Dataset Creation

This dataset was created through a multi-step pipeline:

1. **Base Preference Dataset**: Generated using model pooling to create diverse response pairs for each instruction.

2. **Evolved Preference Dataset**: Used EvolQuality techniques to improve responses while maintaining diverse options.

3. **Critique Dataset**: Added detailed critiques and scores for each response based on helpfulness, relevance, accuracy and depth.

4. **Multi-turn Conversation Dataset**: Extended preferences to cover multi-turn conversations.

5. **Filtered Preference Dataset**: Applied quality filtering based on response similarity, length consistency, and critique scores.

## Dataset Statistics

- Total preference pairs: [NUMBER]
- Average response length: [NUMBER] characters
- Average similarity between response pairs: [NUMBER]
- Percentage chosen A vs B: [NUMBER]% vs [NUMBER]%

## Dataset Structure

```json
{
  "instruction": "Write a poem about autumn leaves.",
  "response_a": "Golden leaves falling gently...",
  "response_b": "Autumn's tapestry unfolds as crimson and amber leaves...",
  "chosen": "B"
}
```

For the critique dataset, additional fields include:
```json
{
  "score_a": {"helpfulness": 3.2, "relevance": 4.0, "accuracy": 3.8, "depth": 2.9, "overall": 3.5},
  "score_b": {"helpfulness": 4.5, "relevance": 4.2, "accuracy": 4.0, "depth": 4.3, "overall": 4.3},
  "critique_a": "The response provides basic information but lacks...",
  "critique_b": "This response offers comprehensive coverage of the topic with..."
}
```

For the multi-turn dataset, the structure is:
```json
{
  "conversation_a": {
    "messages": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
  },
  "conversation_b": {
    "messages": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
  },
  "chosen": "A"
}
```

## Data Exploration

The dataset includes comprehensive analysis of response quality factors including:

1. Response length distribution
2. Readability metrics
3. Similarity between response options
4. Critique scores and their correlation with preferences

Detailed visualizations and analysis can be found in the dataset's analysis_output directory.

## Intended Uses

This dataset is designed for:

1. Training reward models for RLHF
2. Direct preference optimization (DPO) methods
3. Evaluating response quality and preference prediction

## Additional Information

This dataset was created as part of Project 2 (Preference Tuning) for Uplimit's course on synthetic data generation for LLM fine-tuning.

The dataset builds upon the instruction-tuning dataset from Project 1: [LINK TO PROJECT 1 DATASET] 