import pandas as pd
import os
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
import random
import sys

# Load environment variables from .env file
load_dotenv()

# Get environment variables or use defaults
HF_USERNAME = os.environ.get("HF_USERNAME", "your-username")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-3-opus-20240229")
TRANSFORMERS_MODEL = os.environ.get("TRANSFORMERS_MODEL", "meta-llama/Llama-3-8B-Instruct")
SAMPLE_SIZE = int(os.environ.get("CONVERSATION_SAMPLE_SIZE", "500"))
USE_MOCK_DATA = os.environ.get("USE_MOCK_DATA", "").lower() in ("true", "1", "yes")

# Check for API keys and enable mock data if needed
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

if not OPENAI_API_KEY and not ANTHROPIC_API_KEY and not USE_MOCK_DATA:
    print("ERROR: No API keys found and mock data mode is not enabled.")
    print("Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables")
    print("or set USE_MOCK_DATA=true to use mock data for testing.")
    print("\nTo use mock data, run with:")
    print("USE_MOCK_DATA=true python instruction_dataset_generator.py")
    sys.exit(1)

if not OPENAI_API_KEY and not ANTHROPIC_API_KEY:
    print("WARNING: No API keys found. Using mock data mode.")
    USE_MOCK_DATA = True
elif not OPENAI_API_KEY:
    print("WARNING: No OpenAI API key found. Using only Anthropic for API calls.")
elif not ANTHROPIC_API_KEY:
    print("WARNING: No Anthropic API key found. Using only OpenAI for API calls.")

# Print Distilabel version for reference
import distilabel
print(f"Using Distilabel version: {distilabel.__version__}")
print("Using direct API implementation instead of Distilabel pipeline")

# 1. Initial Seed Data
seed_instructions = [
    {"instruction": "Explain the concept of neural networks to a 10-year-old."},
    {"instruction": "Write a poem about autumn leaves."},
    {"instruction": "Describe the process of photosynthesis."},
    {"instruction": "Compare and contrast renewable and non-renewable energy sources."},
    {"instruction": "Explain how to solve a quadratic equation."},
    # Add more diverse seed instructions across different domains
]

# Helper function to load data from Hugging Face
def load_hf_data_to_dicts(dataset_name, split="train"):
    try:
        dataset = load_dataset(dataset_name, split=split)
        return [dict(item) for item in dataset]
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        # Return a default example to prevent pipeline failures
        return [{"instruction": "Default instruction after dataset loading error"}]

# Mock data generator for testing without API keys
def generate_mock_data(count=5, type="instruction"):
    """Generate mock data for testing without API keys"""
    mock_instructions = [
        "Explain the concept of artificial intelligence",
        "Write a short story about space exploration",
        "Describe how climate change affects biodiversity",
        "Explain the process of photosynthesis in detail",
        "Compare different economic systems (capitalism, socialism, etc.)",
        "Describe the water cycle and its importance",
        "Explain how to solve a system of linear equations",
        "Write about the history of the internet",
        "Explain the concept of quantum computing",
        "Describe the process of evolution by natural selection"
    ]
    
    mock_responses = [
        "Artificial intelligence (AI) refers to computer systems designed to perform tasks that typically require human intelligence...",
        "The year was 2157, and humanity had finally developed the technology needed for deep space exploration...",
        "Climate change significantly impacts biodiversity by altering habitats, changing seasonal patterns, and increasing extreme weather events...",
        "Photosynthesis is the process used by plants, algae and certain bacteria to convert light energy, usually from the sun, into chemical energy...",
        "Economic systems vary in how they organize production, distribution, and consumption of goods and services..."
    ]
    
    if type == "instruction":
        return [{"instruction": random.choice(mock_instructions)} for _ in range(count)]
    elif type == "response":
        return [{"instruction": random.choice(mock_instructions), "response": random.choice(mock_responses)} for _ in range(count)]
    elif type == "conversation":
        return [{
            "conversation": {
                "messages": [
                    {"role": "user", "content": random.choice(mock_instructions)},
                    {"role": "assistant", "content": random.choice(mock_responses)},
                    {"role": "user", "content": "Can you elaborate on that?"},
                    {"role": "assistant", "content": "Certainly! " + random.choice(mock_responses)}
                ]
            }
        } for _ in range(count)]
    
    return []

# A simpler version of the pipeline that works with minimal Distilabel functionality
def generate_instructions(input_instructions):
    """Generate new instructions using OpenAI API directly"""
    if USE_MOCK_DATA:
        print("Using mock data for instructions")
        return generate_mock_data(25, "instruction")
    
    from openai import OpenAI
    
    if not OPENAI_API_KEY:
        print("No OpenAI API key available. Cannot generate instructions.")
        return []
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    results = []
    
    for instr in input_instructions:
        prompt = f"Based on this instruction: '{instr['instruction']}', generate 5 new but related instructions that are more specific, challenging or explore different aspects of the topic."
        
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse the response into individual instructions
            generated_text = response.choices[0].message.content
            new_instructions = []
            for line in generated_text.strip().split('\n'):
                if line.strip():
                    new_instructions.append({"instruction": line.strip()})
            
            results.extend(new_instructions)
        except Exception as e:
            print(f"Error generating instructions: {e}")
    
    return results

def evolve_instructions(input_instructions):
    """Evolve instructions using Anthropic API directly"""
    if USE_MOCK_DATA:
        print("Using mock data for evolved instructions")
        return generate_mock_data(15, "instruction")
    
    from anthropic import Anthropic
    
    if not ANTHROPIC_API_KEY:
        print("No Anthropic API key available. Cannot evolve instructions.")
        return []
    
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    results = []
    
    for instr in input_instructions:
        prompt = f"""
        Evolve the following instruction into a more complex, specific, and challenging version:
        INSTRUCTION: {instr['instruction']}
        
        Make it more specific by:
        1. Adding constraints or limitations
        2. Requiring deeper reasoning or analysis
        3. Adding concrete context or examples
        4. Including complications or edge cases
        
        EVOLVED INSTRUCTION:
        """
        
        try:
            response = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            evolved_instruction = response.content[0].text.strip()
            results.append({"instruction": evolved_instruction})
        except Exception as e:
            print(f"Error evolving instructions: {e}")
    
    return results

def generate_responses(input_instructions):
    """Generate responses using a mix of models"""
    if USE_MOCK_DATA:
        print("Using mock data for responses")
        return generate_mock_data(20, "response")
    
    # Determine which APIs are available
    apis_available = []
    if OPENAI_API_KEY:
        apis_available.append("openai")
    if ANTHROPIC_API_KEY:
        apis_available.append("anthropic")
    
    if not apis_available:
        print("No API keys available. Cannot generate responses.")
        return []
    
    openai_client = None
    anthropic_client = None
    
    if "openai" in apis_available:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    if "anthropic" in apis_available:
        from anthropic import Anthropic
        anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    
    results = []
    
    for instr in input_instructions:
        # Choose a random API from available ones
        model_choice = random.choice(apis_available)
        
        try:
            if model_choice == "openai" and openai_client:
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": instr["instruction"]}]
                )
                response_text = response.choices[0].message.content
            elif model_choice == "anthropic" and anthropic_client:
                response = anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": instr["instruction"]}]
                )
                response_text = response.content[0].text
            else:
                continue
                
            results.append({
                "instruction": instr["instruction"],
                "response": response_text
            })
        except Exception as e:
            print(f"Error generating response: {e}")
    
    return results

def generate_conversations(input_data):
    """Generate multi-turn conversations"""
    if USE_MOCK_DATA:
        print("Using mock data for conversations")
        return generate_mock_data(10, "conversation")
    
    from openai import OpenAI
    
    if not OPENAI_API_KEY:
        print("No OpenAI API key available. Cannot generate conversations.")
        return []
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    results = []
    
    for item in input_data:
        instruction = item["instruction"]
        response = item["response"]
        
        # Format the context for follow-up
        context = f"""User: {instruction}
        
Assistant: {response}

User:"""
        
        try:
            # Generate follow-up question
            followup_response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "Generate a natural follow-up question that the user might ask."},
                    {"role": "user", "content": context}
                ]
            )
            followup_question = followup_response.choices[0].message.content.strip()
            
            # Generate response to follow-up
            final_response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": followup_question}
                ]
            )
            followup_answer = final_response.choices[0].message.content.strip()
            
            # Add to results
            results.append({
                "conversation": {
                    "messages": [
                        {"role": "user", "content": instruction},
                        {"role": "assistant", "content": response},
                        {"role": "user", "content": followup_question},
                        {"role": "assistant", "content": followup_answer}
                    ]
                }
            })
        except Exception as e:
            print(f"Error generating conversation: {e}")
    
    return results

# Main execution
if __name__ == "__main__":
    print(f"Using Hugging Face username: {HF_USERNAME}")
    print(f"OpenAI model: {OPENAI_MODEL}" if OPENAI_API_KEY else "OpenAI API key not found")
    print(f"Anthropic model: {ANTHROPIC_MODEL}" if ANTHROPIC_API_KEY else "Anthropic API key not found")
    
    try:
        # Using direct API calls instead of Distilabel pipeline
        print("\n=== Generating Initial Instructions ===")
        # Start with seed instructions
        selfinstruct_results = generate_instructions(seed_instructions)
        if selfinstruct_results:
            selfinstruct_dataset = Dataset.from_list(selfinstruct_results)
            selfinstruct_dataset.push_to_hub(f"{HF_USERNAME}/sft-selfinstruct-dataset")
            print(f"Generated {len(selfinstruct_results)} instructions")
        else:
            print("No instructions were generated")
            selfinstruct_results = seed_instructions
        
        print("\n=== Evolving Instructions ===")
        # Take a sample to evolve
        sample_size = min(50, len(selfinstruct_results))
        sample_to_evolve = random.sample(selfinstruct_results, sample_size)
        evolinstruct_results = evolve_instructions(sample_to_evolve)
        if evolinstruct_results:
            evolinstruct_dataset = Dataset.from_list(evolinstruct_results)
            evolinstruct_dataset.push_to_hub(f"{HF_USERNAME}/sft-evolinstruct-dataset")
            print(f"Evolved {len(evolinstruct_results)} instructions")
        else:
            print("No instructions were evolved")
            evolinstruct_results = selfinstruct_results[:10]  # Use a subset of original instructions
        
        print("\n=== Generating Responses ===")
        # Generate responses for evolved instructions
        response_results = generate_responses(evolinstruct_results)
        if response_results:
            response_dataset = Dataset.from_list(response_results)
            response_dataset.push_to_hub(f"{HF_USERNAME}/sft-response-dataset")
            print(f"Generated {len(response_results)} responses")
        else:
            print("No responses were generated")
            # Create mock responses if needed
            response_results = generate_mock_data(10, "response")
            response_dataset = Dataset.from_list(response_results)
            response_dataset.push_to_hub(f"{HF_USERNAME}/sft-response-dataset")
        
        print("\n=== Generating Conversations ===")
        # Create conversations from a subset of responses
        conv_sample_size = min(SAMPLE_SIZE, len(response_results))
        conv_sample = random.sample(response_results, conv_sample_size)
        conversation_results = generate_conversations(conv_sample)
        if conversation_results:
            conversation_dataset = Dataset.from_list(conversation_results)
            conversation_dataset.push_to_hub(f"{HF_USERNAME}/sft-conversation-dataset")
            print(f"Generated {len(conversation_results)} conversations")
        else:
            print("No conversations were generated")
            # Create mock conversations if needed
            conversation_results = generate_mock_data(5, "conversation")
            conversation_dataset = Dataset.from_list(conversation_results)
            conversation_dataset.push_to_hub(f"{HF_USERNAME}/sft-conversation-dataset")
        
        print("\n=== Creating Final Dataset ===")
        # Combine response and conversation datasets
        response_df = pd.DataFrame(response_dataset)
        conversation_df = pd.DataFrame(conversation_dataset)
        
        # Add dummy 'instruction' and 'response' columns to conversation_df if needed
        if 'instruction' not in conversation_df.columns:
            conversation_df['instruction'] = conversation_df['conversation'].apply(
                lambda x: x['messages'][0]['content']
            )
        if 'response' not in conversation_df.columns:
            conversation_df['response'] = conversation_df['conversation'].apply(
                lambda x: x['messages'][1]['content']
            )
        
        # Combine and create final dataset
        combined_df = pd.concat([response_df, conversation_df], ignore_index=True)
        final_dataset = Dataset.from_pandas(combined_df)
        final_dataset.push_to_hub(f"{HF_USERNAME}/sft-complete-dataset")
        
        print("\n=== All steps completed successfully! ===")
        print(f"Generated a total of {len(final_dataset)} examples")
        print("\nNote: If you ran in mock data mode, the datasets contain placeholder data.")
        print("Set OPENAI_API_KEY and ANTHROPIC_API_KEY in your .env file for real data generation.")
        
    except Exception as e:
        print(f"\n=== ERROR: {e} ===")
        import traceback
        traceback.print_exc() 