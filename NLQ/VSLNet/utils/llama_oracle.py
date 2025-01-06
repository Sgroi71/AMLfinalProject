import argparse
import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Constants for the small LLaMA model (change model name if needed)
MODEL_NAME = "TinyLlama/TinyLlama_v1.1"

def load_llama_model():
    """Load a small LLaMA model and tokenizer."""
    print("Downloading and loading the LLaMA model...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return model, tokenizer

def generate_query_answer_pairs(model, tokenizer, annotations, max_length=128):
    """
    Generate NLQ-style query-answer pairs based on narrations.
    
    Args:
    - model: Pretrained LLaMA model.
    - tokenizer: Corresponding tokenizer.
    - annotations: List of video annotations with narrations.
    - max_length: Maximum length for the generated query.
    
    Returns:
    - List of query-answer pairs in NLQ format.
    """

    print(f"Loading LLaMA Model: {model}")

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    result = {}

    for video_uid, narration_list in annotations.items():
        result[video_uid] = []
      
        for narration_entry in narration_list:
            narrations = [narration[3:] for narration in narration_entry]
            narration_text = " ".join(narrations)

            for _ in range(3):  # Generate 3 query-answer pairs for each narration
                prompt = f"Given the following narration: {narration_text}\nGenerate a query and its answer:"
                
                # Generate the query and answer
                generated_output = generator(prompt, num_return_sequences=1)[0]["generated_text"]
                
                # Extract the query and answer from the generated output
                query_answer = generated_output.replace(prompt, "").strip()
                query, answer = query_answer.split("Answer:", 1) if "Answer:" in query_answer else (query_answer, "N/A")
                
                # Store the query-answer pair
                result[video_uid].append({
                    "start_sec": narration_entry["start_sec"],
                    "end_sec": narration_entry["end_sec"],
                    "query": query.strip(),
                    "answer": answer.strip()
                })

    return result

def main(args):
    # Load the LLaMA model and tokenizer
    model, tokenizer = load_llama_model()
    
    # Load the filtered narrations from the provided JSON file
    with open(args.narration_filename, 'r') as f:
        annotations = json.load(f)
    
    # Generate query-answer pairs
    query_answer_pairs = generate_query_answer_pairs(model, tokenizer, annotations)
    
    # Save the output to a JSON file
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "query_answer_pairs.json")
    
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(query_answer_pairs, file, indent=4)
    
    print(f"Query-answer pairs saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NLQ-style query-answer pairs using a LLaMA model.")
    parser.add_argument("--narration_filename", type=str, required=True, help="Path to the filtered narrations JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the generated query-answer pairs.")
    
    args = parser.parse_args()
    main(args)
