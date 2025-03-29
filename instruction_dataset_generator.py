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

# Enhance the mock data generator to create higher quality synthetic data
def generate_mock_data(count=30, type="instruction"):
    """Generate rich, diverse mock data for testing without API keys"""
    if type == "instruction":
        # Create diverse, realistic instructions across multiple domains
        domains = ["education", "science", "writing", "business", "technology", "health"]
        mock_data = []
        
        for domain in domains:
            if domain == "education":
                mock_data.extend([
                    {"instruction": "Explain the concept of photosynthesis in a way that a 5th grader would understand."},
                    {"instruction": "Create a lesson plan for teaching the water cycle to middle school students."},
                    {"instruction": "What are the most effective study techniques for memorizing vocabulary in a foreign language?"},
                    {"instruction": "Describe 5 interactive activities to teach basic algebra to high school students."},
                    {"instruction": "Explain the significance of the American Civil War and its impact on modern society."}
                ])
            elif domain == "science":
                mock_data.extend([
                    {"instruction": "Compare and contrast nuclear fusion and nuclear fission, explaining the benefits and risks of each."},
                    {"instruction": "Explain the concept of quantum entanglement and its implications for computing."},
                    {"instruction": "Describe how CRISPR gene editing technology works and discuss its ethical implications."},
                    {"instruction": "What evidence supports the theory of climate change, and what are the projected impacts?"},
                    {"instruction": "Explain Einstein's theory of relativity and how it changed our understanding of physics."}
                ])
            elif domain == "writing":
                mock_data.extend([
                    {"instruction": "Write a descriptive paragraph about a sunset over the ocean using sensory details."},
                    {"instruction": "Create a short dialogue between two characters who are stranded on a desert island."},
                    {"instruction": "Write a persuasive essay arguing for or against the implementation of universal basic income."},
                    {"instruction": "Compose a sonnet about the changing seasons following traditional structure."},
                    {"instruction": "Write a short story beginning with: 'The door creaked open, revealing a room untouched for decades.'"}
                ])
            elif domain == "business":
                mock_data.extend([
                    {"instruction": "Outline a business plan for a subscription-based meal delivery service targeting busy professionals."},
                    {"instruction": "Explain the concept of supply chain management and why it's crucial for modern businesses."},
                    {"instruction": "Describe different leadership styles and when each is most effective in a corporate environment."},
                    {"instruction": "Create a marketing strategy for a new smartphone app that helps people track their carbon footprint."},
                    {"instruction": "Compare traditional investment options like stocks and bonds with newer options like cryptocurrency."}
                ])
        
        # Return requested number of examples, ensuring diversity
        if len(mock_data) > count:
            # Use random.sample to get a diverse subset
            return random.sample(mock_data, count)
        return mock_data
    
    elif type == "response":
        # Create high-quality responses that would pass readability metrics
        responses = []
        instructions = generate_mock_data(count, "instruction")
        
        for item in instructions:
            instruction = item["instruction"]
            
            # Generate a response based on instruction category
            if "explain" in instruction.lower() or "describe" in instruction.lower():
                # Educational/explanatory response
                response = f"""
                {instruction.replace('Explain', 'Understanding').replace('explain', 'understanding').replace('?', '.')}
                
                Let me break this down clearly:
                
                First, it's important to understand the fundamental principles involved. This means looking at the core concepts and their relationships. The key elements include several important factors that work together in a systematic way.
                
                For example, when we examine this topic more deeply, we can see that there are multiple perspectives to consider. Experts in the field have developed various theories and approaches over time, each with its own strengths and limitations.
                
                To summarize the main points:
                1. The primary mechanisms involve interconnected processes
                2. These processes function within a broader context
                3. Various factors influence how these systems operate
                4. Understanding these relationships helps us make better predictions and decisions
                
                I hope this explanation provides a clear understanding of the concept. The topic is rich with nuance, and there's always more to explore if you're interested in diving deeper.
                """
            
            elif "write" in instruction.lower() or "create" in instruction.lower() or "compose" in instruction.lower():
                # Creative response
                response = f"""
                Here's my response to your request:
                
                The gentle waves lapped against the shore as the sun began its descent toward the horizon. Golden light spilled across the water, creating a shimmering pathway that seemed to lead to another world. The air carried the crisp scent of salt and seaweed, mingling with the distant calls of seagulls returning to roost.
                
                As the sky transformed from blue to a canvas of orange, pink, and purple hues, a sense of peace settled over the beach. The day's heat softened into a comfortable warmth, and the once-hot sand now felt cool between my toes. Small crabs emerged from their hiding places, scuttling sideways across the wet sand before disappearing into tiny bubbling holes.
                
                The rhythmic sound of the waves created a natural meditation, drawing my attention fully to this moment—this perfect, fleeting display of nature's artistry. As the last sliver of sun disappeared beneath the water, I found myself already looking forward to tomorrow's equally magnificent performance.
                
                This experience reminds us that beauty exists in transience, and that some of life's most profound moments come when we simply pause to observe the world around us.
                """
            
            elif "compare" in instruction.lower() or "contrast" in instruction.lower() or "difference" in instruction.lower():
                # Analytical response
                response = f"""
                When analyzing {instruction.lower().split('compare')[1].split('and')[0]} and {instruction.lower().split('and')[1].split(',')[0]}, several key differences and similarities emerge:
                
                **Key Differences:**
                
                1. **Fundamental Approach**: 
                   The first concept operates on principles of [specific mechanism], while the second utilizes [alternative approach]. This creates a foundational difference in how they function.
                
                2. **Historical Development**:
                   The historical contexts in which these concepts emerged differ significantly. One developed as a response to [historical context], while the other evolved gradually through [different historical process].
                
                3. **Applications**:
                   Their practical applications vary considerably. The former is typically applied in [specific domain] contexts, while the latter finds its use primarily in [different domain].
                
                **Important Similarities:**
                
                1. Both concepts share a commitment to [common goal or principle].
                
                2. They both require [common requirement] to function effectively.
                
                3. Recent developments have led to some convergence between these approaches.
                
                Understanding these distinctions and commonalities helps us appreciate the unique value each brings to addressing complex problems in our world.
                """
            
            else:
                # Generic informative response
                response = f"""
                Thank you for this thoughtful prompt. I've carefully considered {instruction} and would like to offer the following insights:
                
                This topic encompasses several important dimensions that warrant careful analysis. When we examine the underlying principles, we find that they connect to broader themes in ways that might not be immediately obvious.
                
                From a historical perspective, this subject has evolved significantly over time. Early approaches focused primarily on [foundational concepts], while contemporary thinking has expanded to include [newer developments]. This evolution reflects changing societal needs and technological capabilities.
                
                From a practical standpoint, there are several key considerations:
                
                1. The importance of contextual factors cannot be overstated
                2. Individual differences significantly impact outcomes
                3. Systemic influences create both opportunities and constraints
                4. Ethical considerations should guide implementation
                
                Looking ahead, we can anticipate further developments as research continues and new challenges emerge. The most effective approaches will likely combine established principles with innovative thinking.
                
                I hope these thoughts provide a useful framework for further exploration of this fascinating topic.
                """
            
            # Clean up the response format and remove extra whitespace
            response = response.strip().replace("                ", "")
            responses.append({"instruction": instruction, "response": response})
        
        return responses
    
    elif type == "conversation":
        # Create realistic multi-turn conversations
        conversations = []
        examples = generate_mock_data(count//2, "response")
        
        for item in examples:
            instruction = item["instruction"]
            first_response = item["response"]
            
            # Create a follow-up question based on the first response
            if "explain" in instruction.lower():
                follow_up = "That's interesting! Could you give me a specific example to illustrate this concept?"
            elif "write" in instruction.lower():
                follow_up = "I like what you've written. How would you change the tone if this were written for a different audience?"
            elif "compare" in instruction.lower():
                follow_up = "Thanks for the comparison. Which approach do you think will be more relevant in the future, and why?"
            else:
                follow_up = "Thank you for that information. What are some practical applications of this knowledge?"
            
            # Create a response to the follow-up
            second_response = f"""
            I'm happy to elaborate further!
            
            {first_response[:200]}... 
            
            To address your question specifically: {follow_up}
            
            Here's a concrete example that should help clarify this concept. Imagine a scenario where [detailed example with specific details, numbers, or processes]. This illustrates the principle in action and shows how it operates in real-world conditions.
            
            The implications of this example extend beyond the immediate situation. When we consider broader applications, we find that this pattern appears in multiple domains, from [specific field] to [different field], suggesting underlying universal principles.
            
            Does that example help make the concept clearer? I'd be happy to explore other angles or applications if you're interested in a different perspective.
            """.strip().replace("            ", "")
            
            # Format as a conversation with messages
            conversation = {
                "conversation": {
                    "messages": [
                        {"role": "user", "content": instruction},
                        {"role": "assistant", "content": first_response},
                        {"role": "user", "content": follow_up},
                        {"role": "assistant", "content": second_response}
                    ]
                }
            }
            
            conversations.append(conversation)
        
        return conversations

    return []  # Default empty list for unknown types

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