import os
import sys
import random
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# Get environment variables or use defaults
HF_USERNAME = os.environ.get("HF_USERNAME", "your-username")
USE_MOCK_DATA = os.environ.get("USE_MOCK_DATA", "").lower() in ("true", "1", "yes")
SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", "50"))
USE_DISTILABEL = os.environ.get("USE_DISTILABEL", "").lower() in ("true", "1", "yes")

# Dataset names
PREVIOUS_DATASET = os.environ.get("PREVIOUS_DATASET", f"{HF_USERNAME}/sft-filtered-dataset")
BASE_PREFERENCE_DATASET = os.environ.get("BASE_PREFERENCE_DATASET", f"{HF_USERNAME}/base-preference-dataset")
EVOLVED_PREFERENCE_DATASET = os.environ.get("EVOLVED_PREFERENCE_DATASET", f"{HF_USERNAME}/evolved-preference-dataset")
CRITIQUE_DATASET = os.environ.get("CRITIQUE_DATASET", f"{HF_USERNAME}/critique-preference-dataset")
MULTITURN_DATASET = os.environ.get("MULTITURN_DATASET", f"{HF_USERNAME}/multiturn-preference-dataset")

print(f"Starting preference dataset generation for user: {HF_USERNAME}")
print(f"Using mock data: {USE_MOCK_DATA}")
print(f"Using Distilabel: {USE_DISTILABEL}")

# Function to load previous dataset
def load_previous_dataset():
    try:
        print(f"Loading dataset from {PREVIOUS_DATASET}")
        dataset = load_dataset(PREVIOUS_DATASET, split="train")
        print(f"Successfully loaded {len(dataset)} examples")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using sample instructions instead")
        return None

# Generate mock preference data
def generate_mock_preference_data(count=50):
    print(f"Generating {count} mock preference pairs")
    preference_data = []
    
    # Sample instructions (expand as needed)
    instructions = [
        "Explain the concept of neural networks to a beginner.",
        "Write a poem about autumn leaves.",
        "Compare and contrast renewable and non-renewable energy sources.",
        "Explain how blockchain technology works.",
        "Write a story about a robot that develops emotions.",
        "Describe the water cycle in nature.",
        "Explain quantum computing to a non-technical audience.",
        "Write a recipe for chocolate chip cookies.",
        "Describe the major events of World War II.",
        "Explain the theory of relativity in simple terms."
    ]
    
    # Expand instruction list if needed
    while len(instructions) < count:
        instructions.extend(instructions[:count-len(instructions)])
    
    # Generate preference pairs
    for i in range(count):
        instruction = instructions[i % len(instructions)]
        
        # Generate two responses of different quality
        response_weak = f"Here's a basic explanation of {instruction.split()[1]} {instruction.split()[2]}. It involves some concepts and processes. There are a few important points to remember."
        
        response_strong = f"""
        Thank you for asking about {instruction.split()[1]} {instruction.split()[2]}. 
        
        Let me provide a comprehensive explanation:
        
        {instruction.split()[1].capitalize()} {instruction.split()[2]} refers to a sophisticated concept with multiple dimensions. The fundamental principles include [detailed explanation with specific examples]. 
        
        To illustrate this further, imagine [concrete analogy or scenario]. This demonstrates how the process works in practice.
        
        Key aspects to understand:
        1. [First important principle with explanation]
        2. [Second important principle with details]
        3. [Third important principle with context]
        
        Common applications include [specific use cases] and [industry examples].
        
        I hope this helps your understanding! Would you like me to elaborate on any particular aspect?
        """
        
        # Sometimes swap to avoid position bias
        if random.random() < 0.2:
            chosen = "A"
            preference_data.append({
                "instruction": instruction,
                "response_a": response_strong,
                "response_b": response_weak,
                "chosen": chosen
            })
        else:
            chosen = "B"
            preference_data.append({
                "instruction": instruction,
                "response_a": response_weak,
                "response_b": response_strong,
                "chosen": chosen
            })
    
    return preference_data

# Generate evolved responses with EvolQuality (mock version)
def generate_mock_evolved_responses(preference_data):
    print(f"Generating evolved responses for {len(preference_data)} preference pairs")
    evolved_data = []
    
    for item in tqdm(preference_data):
        instruction = item["instruction"]
        chosen_response = item["response_a"] if item["chosen"] == "A" else item["response_b"]
        
        # Create an "evolved" version with more detail and better structure
        evolved_response = f"""
        {chosen_response.strip()}
        
        To expand on this topic further:
        
        Additional insights worth noting include [detailed expansion of key concepts]. Recent research has shown [recent developments or findings].
        
        To provide an even clearer picture, consider this example: [detailed, concrete example that illustrates the concept].
        
        Common misconceptions about this topic include [misconception 1] and [misconception 2]. It's important to understand that [clarification of misconceptions].
        
        For practical applications, you might consider [specific actionable applications or examples].
        
        I hope this comprehensive explanation has been helpful. Please let me know if you have any other questions!
        """
        
        # Create a new preference pair with the evolved response
        if random.random() < 0.1:  # 10% chance the evolved response isn't better
            chosen = "A"
            evolved_data.append({
                "instruction": instruction,
                "response_a": chosen_response,
                "response_b": evolved_response,
                "chosen": chosen
            })
        else:
            chosen = "B"
            evolved_data.append({
                "instruction": instruction,
                "response_a": chosen_response,
                "response_b": evolved_response,
                "chosen": chosen
            })
    
    return evolved_data

# Generate critiques (mock version)
def generate_mock_critiques(preference_data):
    print(f"Generating critiques for {len(preference_data)} preference pairs")
    critique_data = []
    
    for item in tqdm(preference_data):
        instruction = item["instruction"]
        response_a = item["response_a"]
        response_b = item["response_b"]
        chosen = item["chosen"]
        
        # Generate mock scores
        score_a = {
            "helpfulness": random.uniform(1, 5) if chosen != "A" else random.uniform(3, 5),
            "relevance": random.uniform(1, 5) if chosen != "A" else random.uniform(3, 5),
            "accuracy": random.uniform(1, 5) if chosen != "A" else random.uniform(3, 5),
            "depth": random.uniform(1, 5) if chosen != "A" else random.uniform(3, 5),
            "overall": 0  # Will calculate below
        }
        
        score_b = {
            "helpfulness": random.uniform(1, 5) if chosen != "B" else random.uniform(3, 5),
            "relevance": random.uniform(1, 5) if chosen != "B" else random.uniform(3, 5),
            "accuracy": random.uniform(1, 5) if chosen != "B" else random.uniform(3, 5),
            "depth": random.uniform(1, 5) if chosen != "B" else random.uniform(3, 5),
            "overall": 0  # Will calculate below
        }
        
        # Calculate overall scores
        score_a["overall"] = sum([score_a["helpfulness"], score_a["relevance"], score_a["accuracy"], score_a["depth"]]) / 4
        score_b["overall"] = sum([score_b["helpfulness"], score_b["relevance"], score_b["accuracy"], score_b["depth"]]) / 4
        
        # Generate critiques
        critique_a = f"""
        Critique for Response A:
        
        Helpfulness ({score_a['helpfulness']:.1f}/5): The response {'provides useful information and answers the question directly' if score_a['helpfulness'] > 3 else 'could be more helpful by providing clearer information'}.
        
        Relevance ({score_a['relevance']:.1f}/5): The content is {'highly relevant to the instruction' if score_a['relevance'] > 3 else 'somewhat off-topic and doesn\'t fully address the instruction'}.
        
        Accuracy ({score_a['accuracy']:.1f}/5): The information provided is {'accurate and factually correct' if score_a['accuracy'] > 3 else 'contains some inaccuracies or misleading statements'}.
        
        Depth ({score_a['depth']:.1f}/5): The response {'explores the topic in sufficient depth' if score_a['depth'] > 3 else 'is quite shallow and doesn\'t delve into important details'}.
        
        Overall assessment: This response {'effectively addresses the instruction and provides valuable information' if score_a['overall'] > 3 else 'has significant room for improvement in addressing the instruction'}.
        """
        
        critique_b = f"""
        Critique for Response B:
        
        Helpfulness ({score_b['helpfulness']:.1f}/5): The response {'provides useful information and answers the question directly' if score_b['helpfulness'] > 3 else 'could be more helpful by providing clearer information'}.
        
        Relevance ({score_b['relevance']:.1f}/5): The content is {'highly relevant to the instruction' if score_b['relevance'] > 3 else 'somewhat off-topic and doesn\'t fully address the instruction'}.
        
        Accuracy ({score_b['accuracy']:.1f}/5): The information provided is {'accurate and factually correct' if score_b['accuracy'] > 3 else 'contains some inaccuracies or misleading statements'}.
        
        Depth ({score_b['depth']:.1f}/5): The response {'explores the topic in sufficient depth' if score_b['depth'] > 3 else 'is quite shallow and doesn\'t delve into important details'}.
        
        Overall assessment: This response {'effectively addresses the instruction and provides valuable information' if score_b['overall'] > 3 else 'has significant room for improvement in addressing the instruction'}.
        """
        
        critique_data.append({
            "instruction": instruction,
            "response_a": response_a,
            "response_b": response_b,
            "chosen": chosen,
            "score_a": score_a,
            "score_b": score_b,
            "critique_a": critique_a,
            "critique_b": critique_b
        })
    
    return critique_data

# Generate multi-turn conversation pairs (mock version)
def generate_mock_multiturn_preferences(preference_data):
    print(f"Generating multi-turn conversations for {len(preference_data)} preference pairs")
    multiturn_data = []
    
    for item in tqdm(preference_data):
        instruction = item["instruction"]
        response_a = item["response_a"]
        response_b = item["response_b"]
        chosen = item["chosen"]
        
        # Generate a follow-up question
        if "explain" in instruction.lower():
            followup = "Can you provide a concrete example to illustrate this concept?"
        elif "write" in instruction.lower():
            followup = "Could you explain the creative choices you made in this piece?"
        elif "compare" in instruction.lower():
            followup = "Which option do you think is more sustainable for the future?"
        else:
            followup = "Can you elaborate on one of the key points you mentioned?"
        
        # Generate second-turn responses
        response_a2 = f"""
        Thank you for your follow-up question.
        
        {random.choice([
            "Here's a simple example that might help illustrate the concept.",
            "I'll try to elaborate on the key points I mentioned earlier.",
            "Let me provide some additional context that might be helpful."
        ])}
        
        {random.choice([
            "The basic idea is that...",
            "To put it simply...",
            "In essence, what this means is..."
        ])}
        
        I hope that clarifies things a bit.
        """
        
        response_b2 = f"""
        Thank you for your excellent follow-up question!
        
        To address your question about {followup.lower().replace('can you ', '').replace('could you ', '')}:
        
        Let me provide a detailed example: [specific, concrete example with relevant details and context]. This example demonstrates [key principle] because [explanation of relationship].
        
        Additionally, it's worth noting that [additional insight or perspective]. This connects to our earlier discussion by [relationship to previous points].
        
        Some practical applications of this include:
        1. [First application with explanation]
        2. [Second application with explanation]
        3. [Third application with explanation]
        
        Does this example help clarify the concept? I'm happy to provide further elaboration or a different type of example if needed.
        """
        
        # Create conversation objects
        conversation_a = {
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response_a},
                {"role": "user", "content": followup},
                {"role": "assistant", "content": response_a2}
            ]
        }
        
        conversation_b = {
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response_b},
                {"role": "user", "content": followup},
                {"role": "assistant", "content": response_b2}
            ]
        }
        
        # Determine multi-turn chosen (usually follows single-turn, but can differ)
        multiturn_chosen = chosen
        if random.random() < 0.2:  # 20% chance to flip preference in multi-turn
            multiturn_chosen = "A" if chosen == "B" else "B"
        
        multiturn_data.append({
            "conversation_a": conversation_a,
            "conversation_b": conversation_b,
            "chosen": multiturn_chosen
        })
    
    return multiturn_data

# Function to implement preference generation with Distilabel
def generate_with_distilabel(instructions):
    try:
        from distilabel.models.llms.huggingface import InferenceEndpointsLLM
        from distilabel.pipeline import Pipeline
        from distilabel.steps import GroupColumns, LoadDataFromDicts
        from distilabel.steps.tasks import TextGeneration
        
        print("Generating preference pairs using Distilabel")
        
        base_url = "https://api-inference.huggingface.co/models"
        
        # Create input data
        input_data = [{"instruction": instr} for instr in instructions]
        
        with Pipeline() as pipeline:
            data = LoadDataFromDicts(data=input_data)
            
            # Use different models for different quality responses
            llm_a = InferenceEndpointsLLM(
                base_url=f"{base_url}/meta-llama/Llama-3.2-3B-Instruct",  # Smaller model
                headers={"Authorization": f"Bearer {os.environ.get('HF_TOKEN')}"}
            )
            gen_a = TextGeneration(
                llm=llm_a, 
                output_mappings={"generation": "response_a"},
                generation_kwargs={"temperature": 0.3}  # Lower temperature
            )
            
            llm_b = InferenceEndpointsLLM(
                base_url=f"{base_url}/meta-llama/Llama-3.3-70B-Instruct",  # Larger model
                headers={"Authorization": f"Bearer {os.environ.get('HF_TOKEN')}"}
            )
            gen_b = TextGeneration(
                llm=llm_b, 
                output_mappings={"generation": "response_b"},
                generation_kwargs={"temperature": 0.7}  # Higher temperature
            )
            
            group = GroupColumns(columns=["response_a", "response_b"])
            data >> [gen_a, gen_b] >> group
        
        # Run the pipeline
        distiset = pipeline.run()
        
        # Post-process to add 'chosen' field (assume the larger model is better)
        preference_data = []
        for item in distiset:
            preference_data.append({
                "instruction": item["instruction"],
                "response_a": item["response_a"],
                "response_b": item["response_b"],
                "chosen": "B"  # Assuming B (larger model) is usually better
            })
        
        return preference_data
    
    except Exception as e:
        print(f"Error using Distilabel: {e}")
        print("Falling back to mock data generation")
        return generate_mock_preference_data(len(instructions))

# Main function
if __name__ == "__main__":
    try:
        # Step 1: Load instructions from previous dataset
        dataset = load_previous_dataset()
        
        if dataset:
            # Sample instructions from our previous dataset
            sample_size = min(SAMPLE_SIZE, len(dataset))
            sample_indices = random.sample(range(len(dataset)), sample_size)
            instructions = [dataset[i]["instruction"] for i in sample_indices]
        else:
            # Generate mock instructions if needed
            instructions = [
                "Explain the concept of neural networks to a beginner.",
                "Write a poem about autumn leaves.",
                "Compare and contrast renewable and non-renewable energy sources.",
                "Explain how blockchain technology works.",
                "Write a story about a robot that develops emotions.",
                "Describe the water cycle in nature.",
                "Explain quantum computing to a non-technical audience.",
                "Write a recipe for chocolate chip cookies.",
                "Describe the major events of World War II.",
                "Explain the theory of relativity in simple terms."
            ]
            # Expand if needed
            while len(instructions) < SAMPLE_SIZE:
                instructions.extend(instructions[:SAMPLE_SIZE - len(instructions)])
            instructions = instructions[:SAMPLE_SIZE]
        
        # Step 2: Generate base preference dataset
        if USE_DISTILABEL and os.environ.get("HF_TOKEN"):
            preference_data = generate_with_distilabel(instructions)
        else:
            preference_data = generate_mock_preference_data(len(instructions))
        
        # Save base preference dataset
        base_dataset = Dataset.from_list(preference_data)
        base_dataset.push_to_hub(BASE_PREFERENCE_DATASET)
        print(f"Saved base preference dataset to {BASE_PREFERENCE_DATASET}")
        
        # Step 3: Generate evolved preference dataset
        evolved_data = generate_mock_evolved_responses(preference_data)
        evolved_dataset = Dataset.from_list(evolved_data)
        evolved_dataset.push_to_hub(EVOLVED_PREFERENCE_DATASET)
        print(f"Saved evolved preference dataset to {EVOLVED_PREFERENCE_DATASET}")
        
        # Step 4: Generate critiques
        critique_data = generate_mock_critiques(preference_data)
        critique_dataset = Dataset.from_list(critique_data)
        critique_dataset.push_to_hub(CRITIQUE_DATASET)
        print(f"Saved critique dataset to {CRITIQUE_DATASET}")
        
        # Step 5: Generate multi-turn preferences
        multiturn_data = generate_mock_multiturn_preferences(preference_data)
        multiturn_dataset = Dataset.from_list(multiturn_data)
        multiturn_dataset.push_to_hub(MULTITURN_DATASET)
        print(f"Saved multi-turn dataset to {MULTITURN_DATASET}")
        
        print("\nPreference dataset generation complete!")
        print(f"Generated datasets:")
        print(f"- Base preference pairs: {len(preference_data)}")
        print(f"- Evolved preference pairs: {len(evolved_data)}")
        print(f"- Critique data: {len(critique_data)}")
        print(f"- Multi-turn conversations: {len(multiturn_data)}")
        
    except Exception as e:
        print(f"Error in preference dataset generation: {e}")
        import traceback
        traceback.print_exc() 