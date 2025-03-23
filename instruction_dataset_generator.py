import pandas as pd
from datasets import load_dataset, Dataset
from distilabel.llms import OpenAILLM, TransformersLLM, AnthropicLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts, LoadDataFromHF
from distilabel.steps.tasks import TextGeneration
from distilabel.steps.processing import TextProcessing
from distilabel.utils.functions import merge_dicts
import random

# 1. Initial Seed Data
seed_instructions = [
    {"instruction": "Explain the concept of neural networks to a 10-year-old."},
    {"instruction": "Write a poem about autumn leaves."},
    {"instruction": "Describe the process of photosynthesis."},
    {"instruction": "Compare and contrast renewable and non-renewable energy sources."},
    {"instruction": "Explain how to solve a quadratic equation."},
    # Add more diverse seed instructions across different domains
]

# 2. SelfInstruct Pipeline for Generating from Seeds
def create_selfinstruct_pipeline():
    with Pipeline(name="selfinstruct_pipeline") as pipeline:
        # Load seed data
        data = LoadDataFromDicts(data=seed_instructions)
        
        # Use a stronger model for generation
        llm = OpenAILLM(model="gpt-4o")
        
        # Generate new instructions
        selfinstruct_prompt = TextProcessing(
            input_keys=["instruction"],
            output_key="evolved_instruction",
            function=lambda x: f"Based on this instruction: '{x['instruction']}', generate 5 new but related instructions that are more specific, challenging or explore different aspects of the topic."
        )
        
        gen_instructions = TextGeneration(
            llm=llm, 
            input_mappings={"prompt": "evolved_instruction"},
            output_mappings={"generation": "new_instructions"}
        )
        
        # Parser to split the generated instructions
        instruction_parser = TextProcessing(
            input_keys=["new_instructions"],
            output_key="parsed_instructions",
            function=lambda x: [{"instruction": instr.strip()} for instr in x["new_instructions"].split("\n") if instr.strip()]
        )
        
        data >> selfinstruct_prompt >> gen_instructions >> instruction_parser
    
    return pipeline

# 3. EvolInstruct Pipeline for Evolving Instructions
def create_evolinstruct_pipeline():
    with Pipeline(name="evolinstruct_pipeline") as pipeline:
        # We'll load data from previous pipeline
        data = LoadDataFromHF(dataset="path_to_selfinstruct_output")
        
        # Evolution prompt
        evol_prompt = TextProcessing(
            input_keys=["instruction"],
            output_key="evol_prompt",
            function=lambda x: f"""
            Evolve the following instruction into a more complex, specific, and challenging version:
            INSTRUCTION: {x['instruction']}
            
            Make it more specific by:
            1. Adding constraints or limitations
            2. Requiring deeper reasoning or analysis
            3. Adding concrete context or examples
            4. Including complications or edge cases
            
            EVOLVED INSTRUCTION:
            """
        )
        
        # Use Anthropic model for evolution
        llm = AnthropicLLM(model="claude-3-opus-20240229")
        
        # Generate evolved instructions
        gen_evolved = TextGeneration(
            llm=llm,
            input_mappings={"prompt": "evol_prompt"},
            output_mappings={"generation": "evolved_instruction"}
        )
        
        data >> evol_prompt >> gen_evolved
    
    return pipeline

# 4. Response Generation Pipeline
def create_response_pipeline():
    with Pipeline(name="response_generation") as pipeline:
        # Load data from evolinstruct output
        data = LoadDataFromHF(dataset="path_to_evolinstruct_output")
        
        # Use different models for generating responses
        llm_options = [
            TransformersLLM(model="meta-llama/Llama-3-8B-Instruct"),
            AnthropicLLM(model="claude-3-sonnet-20240229"),
            OpenAILLM(model="gpt-3.5-turbo")
        ]
        
        # Randomly select model for each instruction
        llm_selector = TextProcessing(
            input_keys=["instruction"],
            output_key="selected_model",
            function=lambda x: random.choice(llm_options)
        )
        
        # Generate responses
        gen_response = TextGeneration(
            llm=lambda x: x["selected_model"],
            input_mappings={"prompt": "instruction"},
            output_mappings={"generation": "response"}
        )
        
        data >> llm_selector >> gen_response
    
    return pipeline

# 5. Multi-turn Conversation Generation using Magpie
def create_conversation_pipeline():
    with Pipeline(name="conversation_pipeline") as pipeline:
        # Start with some responses we generated
        data = LoadDataFromHF(dataset="path_to_response_output", split="train[:500]")
        
        # Create magpie prompt for follow-up
        magpie_prompt = TextProcessing(
            input_keys=["instruction", "response"],
            output_key="magpie_prompt",
            function=lambda x: f"""<|im_start|>user
{x['instruction']}
<|im_end|>
<|im_start|>assistant
{x['response']}
<|im_end|>
<|im_start|>user
"""
        )
        
        # Generate follow-up questions
        llm = OpenAILLM(model="gpt-4o")
        gen_followup = TextGeneration(
            llm=llm,
            input_mappings={"prompt": "magpie_prompt"},
            output_mappings={"generation": "followup_question"}
        )
        
        # Generate assistant response to follow-up
        followup_prompt = TextProcessing(
            input_keys=["magpie_prompt", "followup_question"],
            output_key="followup_prompt",
            function=lambda x: x["magpie_prompt"] + x["followup_question"] + "\n<|im_end|>\n<|im_start|>assistant\n"
        )
        
        gen_followup_response = TextGeneration(
            llm=llm,
            input_mappings={"prompt": "followup_prompt"},
            output_mappings={"generation": "followup_response"}
        )
        
        # Format the conversation
        format_conversation = TextProcessing(
            input_keys=["instruction", "response", "followup_question", "followup_response"],
            output_key="conversation",
            function=lambda x: {
                "messages": [
                    {"role": "user", "content": x["instruction"]},
                    {"role": "assistant", "content": x["response"]},
                    {"role": "user", "content": x["followup_question"]},
                    {"role": "assistant", "content": x["followup_response"]}
                ]
            }
        )
        
        data >> magpie_prompt >> gen_followup >> followup_prompt >> gen_followup_response >> format_conversation
    
    return pipeline

# Main execution
if __name__ == "__main__":
    # Generate data using SelfInstruct
    selfinstruct_pipeline = create_selfinstruct_pipeline()
    selfinstruct_dataset = selfinstruct_pipeline.run(use_cache=False)
    selfinstruct_dataset.push_to_hub("yourusername/sft-selfinstruct-dataset")
    
    # Evolve instructions using EvolInstruct
    evolinstruct_pipeline = create_evolinstruct_pipeline()
    evolinstruct_dataset = evolinstruct_pipeline.run(use_cache=False)
    evolinstruct_dataset.push_to_hub("yourusername/sft-evolinstruct-dataset")
    
    # Generate responses
    response_pipeline = create_response_pipeline()
    response_dataset = response_pipeline.run(use_cache=False)
    response_dataset.push_to_hub("yourusername/sft-response-dataset")
    
    # Generate multi-turn conversations
    conversation_pipeline = create_conversation_pipeline()
    conversation_dataset = conversation_pipeline.run(use_cache=False)
    conversation_dataset.push_to_hub("yourusername/sft-conversation-dataset")
    
    # Merge all datasets for the final comprehensive dataset
    # Will be filtered in the analysis step
    # ... code to merge datasets ...
    
    final_dataset = merge_datasets()
    final_dataset.push_to_hub("yourusername/sft-complete-dataset") 