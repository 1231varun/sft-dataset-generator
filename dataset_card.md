---
license: apache-2.0
tags:
- instruction-tuning
- conversational
- sft
datasets:
- yourusername/sft-complete-dataset
- yourusername/sft-filtered-dataset
language:
- en
---

# SFT Instruction Dataset

A high-quality dataset for supervised fine-tuning of large language models, featuring diverse instructions, responses, and multi-turn conversations.

## Dataset Description

### Dataset Summary

This dataset was created for the Uplimit LLM course Project 1. It contains a diverse collection of instructions, responses, and multi-turn conversations designed for supervised fine-tuning (SFT) of large language models. The dataset was generated using multiple techniques including SelfInstruct, EvolInstruct, and Magpie approaches.

The dataset includes:
- Single-turn instruction-response pairs
- Multi-turn conversations (2+ turns)
- Instructions across diverse domains and complexity levels
- Both a complete dataset and a filtered version based on quality metrics

### Dataset Creation

The dataset was created using a multi-step pipeline:

1. **SelfInstruct Generation**: Starting with a set of high-quality seed instructions, we used GPT-4o to generate new diverse instructions.

2. **EvolInstruct Refinement**: Instructions were evolved using Claude 3 Opus to create more complex, specific, and challenging versions with constraints, deeper reasoning requirements, concrete context, and edge cases.

3. **Response Generation**: High-quality responses were generated using a mixture of models including Llama 3, Claude 3, and GPT-3.5 to ensure diversity.

4. **Multi-turn Conversation Generation**: Using the Magpie technique, we extended single-turn interactions into multi-turn conversations with follow-up questions and responses.

5. **Quality Analysis & Filtering**: The dataset underwent rigorous analysis for toxicity, complexity, readability, and domain distribution. A filtered version removes low-quality or potentially problematic examples.

### Data Fields

**Complete Dataset**:
- `instruction`: The instruction or query
- `response`: The response to the instruction
- `domain`: The classified domain of the instruction
- `readability_score`: Flesch Reading Ease score
- `complexity_score`: Text standard complexity score
- `toxicity_score`: Toxicity probability

**Multi-turn Conversations**:
- `conversation`: A dictionary containing the conversation
  - `messages`: List of message dictionaries
    - `role`: Either "user" or "assistant"
    - `content`: The content of the message

**Filtered Dataset**:
- Same as complete dataset, but filtered for quality

### Dataset Statistics

- Total examples: 5,000
- Filtered examples: 4,325
- Domains covered: 23 distinct domains
- Average instruction length: 78.3 characters
- Average response length: 412.6 characters
- Multi-turn conversations: 1,000

## Data Visualization

### Domain Distribution
![Domain Distribution](domain_distribution.png)

### Length Distributions
![Length Distributions](length_distributions.png)

### Embedding Visualization
![Embedding Visualization](embedding_visualization.png)

### Interactive Visualization
An interactive visualization of the dataset is available at [Nomic Atlas](https://atlas.nomic.ai/map/sft-instruction-atlas).

## Dataset Creation Process

### Data Collection and Preprocessing

The data generation process involved:

1. **Seed Data Selection**: We carefully curated 20 high-quality seed instructions across diverse domains.

2. **Instruction Generation Pipeline**:
   ```python
   # Simplified example of our SelfInstruct pipeline
   with Pipeline() as pipeline:
       data = LoadDataFromDicts(data=seed_instructions)
       llm = OpenAILLM(model="gpt-4o")
       selfinstruct_prompt = TextProcessing(
           input_keys=["instruction"],
           output_key="evolved_instruction",
           function=lambda x: f"Based on this instruction: '{x['instruction']}', generate 5 new but related instructions..."
       )
       gen_instructions = TextGeneration(llm=llm, ...)
       # ...
   ```

3. **Quality Filtering Process**:
   - Removed instructions with toxicity score > 0.3
   - Ensured minimum length requirements
   - Applied readability thresholds
   - Maintained domain balance

## Intended Uses

This dataset is designed for:
- Fine-tuning language models for improved instruction following
- Evaluating model capabilities across diverse domains
- Research on instruction following and conversational abilities
- Benchmarking model performance on varied instruction types

## Limitations

- The dataset is primarily in English
- Generated by existing AI systems, potentially inheriting their biases
- May not cover all edge cases or specialized domains
- Response quality varies by the generating model

## Additional Information

### Dataset Versions

- **Complete Dataset**: Contains all generated examples
- **Filtered Dataset**: Quality-filtered subset recommended for training

### Citation Information

If you use this dataset, please cite it as: 

# SFT Filtered Dataset

## Dataset Description
This dataset contains high-quality instruction-response pairs for supervised fine-tuning of language models. The dataset has been filtered for readability, appropriate length, and low toxicity to ensure high-quality training examples.

## Dataset Creation Process
This dataset was created through a synthetic data generation pipeline:

1. **Initial Seed Instructions**: Started with diverse seed instructions across domains including education, science, writing, business, technology, and health.

2. **Instruction Evolution**: Applied techniques similar to EvolInstruct to develop more complex and nuanced instructions.

3. **Response Generation**: Created high-quality, detailed responses that meet readability standards.

4. **Multi-turn Conversations**: Generated natural follow-up questions and responses to create conversational examples.

5. **Quality Filtering**: Applied multiple quality metrics to ensure only the best examples are included:
   - Minimum instruction and response lengths
   - Readability scoring
   - Toxicity detection
   - Content diversity

## Dataset Statistics
- Total examples: [fill in after running]
- Average instruction length: [fill in after running] characters
- Average response length: [fill in after running] characters
- Domains covered: Education, Science, Writing, Business, Technology, Health

## Dataset Structure
```json
{
  "instruction": "Write a descriptive paragraph about a sunset over the ocean using sensory details.",
  "response": "The gentle waves lapped against the shore as the sun began its descent toward the horizon..."
}
```

## Intended Uses
This dataset is designed for supervised fine-tuning of language models to improve:
- Instruction following capability
- Response quality and readability
- Domain knowledge across multiple topics

## Additional Information
This dataset was created as part of Project 1 for Uplimit's course on synthetic data generation for LLM fine-tuning.

You can find related datasets from the same pipeline:
- [Initial instructions](https://huggingface.co/datasets/1231varun/sft-selfinstruct-dataset)
- [Evolved instructions](https://huggingface.co/datasets/1231varun/sft-evolinstruct-dataset)
- [Response dataset](https://huggingface.co/datasets/1231varun/sft-response-dataset)
- [Conversation dataset](https://huggingface.co/datasets/1231varun/sft-conversation-dataset)
- [Complete dataset](https://huggingface.co/datasets/1231varun/sft-complete-dataset) 