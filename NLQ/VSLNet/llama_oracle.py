from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch
import options
import json


# Load the tokenizer and model



#we define a method to ask any prompt to llama
def ask_llama(device,tokenizer, model,prompt, maxl=800, temp=0.7):
    """
    Send a prompt to the Llama model and get a response.

    Args:
    - configs (Namespace): The configuration options for the model.
    - prompt (str): The input question or statement to the model.
    - max_length (int): The maximum length of the response.
    - temperature (float): Controls randomness in the model's output.

    Returns:
    - str: The model's generated response.
    """
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    inputs.to(device)

    # Generate the output
    outputs = model.generate(
        inputs['input_ids'],  # Tokenized input
        max_length=maxl,         # Limit response length to avoid extra text
        temperature=temp,        # Lower temperature to reduce randomness
        do_sample=True,        # Disable sampling for deterministic output
        pad_token_id=tokenizer.eos_token_id  # Ensure the model doesn't go beyond the end token
    )

    # Decode and return the response
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

if __name__ == "__main__":

    login(token="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")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs, parser = options.read_command_line()
    model_id = "meta-llama/Llama-3.2-3B"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = model.to(device)

    #create prompt
    base_prompt = "Given the following narrations describing the actions of a person, generate a set of simple queries (one per line) that could be answered by looking at the video segments corresponding to these narrations:"

    narration_filename = configs.narration_filename
    #read the json file as a dictionary
    with open(narration_filename) as f:
        narrations = json.load(f)
    
    for video_uid, video in narrations.items():
        for narrationblock in video:
            prompt = base_prompt
            for n in narrationblock['narrations']:
                prompt += f"\n{n[3:]}"
            print(ask_llama(device,tokenizer,model,prompt))
        
    

