import argparse
import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Constants for the small LLaMA model (change model name if needed)
MODEL_NAME = "TinyLlama/TinyLlama_v1.1"

def load_llama_model():
    """Load a small LLaMA model and tokenizer on GPU if available."""
    print("Downloading and loading the LLaMA model...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    return model, tokenizer, device

def generate_query_answer_pairs(model, tokenizer, annotations, device, max_length, temperature):
    """
    Generate NLQ-style query-answer pairs based on narrations.
    
    Args:
    - model: Pretrained LLaMA model.
    - tokenizer: Corresponding tokenizer.
    - annotations: List of video annotations with narrations.
    - device: Device to run the model on (CPU or GPU).
    - max_length: Maximum length for the generated output.
    
    Returns:
    - Dictionary of query-answer pairs in NLQ format.
    """

    print(f"Running text generation on device: {device}")

    # Create the text-generation pipeline with the specified device
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)

    result = {}

    for video_uid, narration_list in annotations.items():
        result[video_uid] = []
      
        for narration_entry in narration_list:
            # Remove #C and prepare narrations
            narrations = [narration[3:].strip() for narration in narration_entry["narrations"]]
            narration_text = " ".join(narrations)

            prompt = f"""
Given the following narrations describing the actions of a person, generate a set of queries and their corresponding answers that could be answered by watching the video segment. Provide 3-5 query-answer pairs in the following format:

Query: <question>
Answer: <answer>

Narrations:
{narration_text}
            """

            # Generate the query and answer pairs
            generated_output = generator(prompt, num_return_sequences=1, max_length=max_length, temperature=temperature)[0]["generated_text"]

            # Extract query-answer pairs
            split_output = generated_output.replace(prompt, "").strip().split("\n")
            query = None
            answer = None

            for line in split_output:
                if line.startswith("Query:"):
                    query = line.replace("Query:", "").strip()
                elif line.startswith("Answer:"):
                    answer = line.replace("Answer:", "").strip()
                    if query and answer:
                        # Store the query-answer pair
                        result[video_uid].append({
                            "start_sec": narration_entry["start_sec"],
                            "end_sec": narration_entry["end_sec"],
                            "query": query,
                            "answer": answer
                        })
                        query = None  # Reset for the next pair
                        answer = None

    return result

def main(args):
    # Load the LLaMA model and tokenizer
    model, tokenizer, device = load_llama_model()
    
    # Load the filtered narrations from the provided JSON file
    with open(args.narration_filename, 'r') as f:
        annotations = json.load(f)
    
    # Generate query-answer pairs
    query_answer_pairs = generate_query_answer_pairs(model, tokenizer, annotations, device, args.max_length, args.temperature)
    
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
    parser.add_argument("--max_length", type=int, default=800, help="Maximum length for the generated output.")
    parser.add_argument("--temperature", type=float, default=0.4, help="Sampling temperature for text generation.")
    
    args = parser.parse_args()
    main(args)
