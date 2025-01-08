from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch
import options
import json
import time
from tqdm import tqdm

# Load the tokenizer and model


def process_responses(response):
    # Split the response into individual lines
    lines = response.split("\n")
    
    valid_end = "?"

    # Process each line
    processed_lines = []
    for line in lines:
        # Remove leading numbers and dots
        line = line.lstrip("0123456789. ").strip()
        
        # Check if the line is in the correct format
        if line.endswith(valid_end):
            processed_lines.append(line)
    
    # eliminate duplicates
    processed_lines = list(set(processed_lines))
    return processed_lines

#we define a method to ask any prompt to llama
def ask_llama(
    device, tokenizer, model, prompt,
    max_new_tokens=80,  # or use 'max_length' if needed
    temperature=0.4,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1
):
    """
    Send a prompt to the Llama model and get a response.

    Args:
        device (torch.device): The device (CPU or GPU).
        tokenizer (AutoTokenizer): The tokenizer.
        model (AutoModelForCausalLM): The Llama model.
        prompt (str): The input question or statement to the model.
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling threshold.
        top_k (int): Top-k sampling cutoff.
        repetition_penalty (float): Penalty for repeated phrases/tokens.

    Returns:
        str: The model's generated response.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate the output
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

if __name__ == "__main__":

    
    start_time = time.time()

    login(token="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")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs, parser = options.read_command_line()
    model_id = "meta-llama/Llama-3.2-3B"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = model.to(device)

    end_time = time.time()
    print(f"Time required to load the model: {end_time - start_time} seconds")

    #create prompt

    base_prompt = """
        Given the following narrations describing the actions of a person, generate a set of simple queries (one per narration). Each query should be a question that can be answered based on the information provided in the corresponding narration.
        You must refer to the person doing the actions in the narrations as "C". There must be one question per narration.
        Output the list of questions maximum 5 line in a clear and concise manner. Do not include duplicates or irrelevant terms do not produce othe examples.

        **Example 1:**
        Narrations:
        C cuts vegetables with a knife
        C places the knife on the table
        C takes a bowl from the cupboard
        C pours the vegetables into the bowl
        C washes their hands in the sink
        Response:
        What is C using to cut vegetables?
        Where does C place the knife?
        What does C take from the cupboard?
        What is being poured into the bowl?
        Where does C wash their hands?

        ---

        **Example 2:**
        Narrations:
        C opens a book
        C flips through the pages
        C picks up a pen from the table
        C writes something in the book
        C closes the book
        Response:
        What does C open at the start?
        What is C doing with the pages?
        What item does C pick up from the table?
        What does C use to write in the book?
        What does C do after writing?

        ---

        **Example 3:**
        Narrations:
        C ties their shoelaces
        C puts on a jacket
        C picks up an umbrella from the floor
        C opens the door
        C steps outside
        Response:
        What is C tying?
        What does C put on after tying their shoelaces?
        What does C pick up from the floor?
        What does C open before stepping outside?
        Where does C go after opening the door?

        ---
        Now, it's your turn! Generate a set of simple queries based on the following narrations:
        """
    narration_filename = configs.narration_filename
    #read the json file as a dictionary
    with open(narration_filename) as f:
        narrations = json.load(f)
    results = {}
    i=1
    for video_uid, video in tqdm(narrations.items()):
        objectres=[]
        for narrationblock in video:
            narrares=[]
            narrationobject = {
                "start_sec":narrationblock["start_sec"],
                "end_sec":narrationblock["end_sec"],
                "questions":[],
                "answers":narrationblock['narrations'],
            }
            prompt = base_prompt
            for n in narrationblock['narrations']:
                prompt += f"\n{n[3:]}"
            prompt += "\nAnswer:\n"
            response=ask_llama(device,tokenizer,model,prompt)
            response = response.split("Answer:")[1]
            print (f"response{i}: {response}")
            i+=1
            narrares.extend(process_responses(response))
            narrationobject["questions"]=narrares
            objectres.append(narrationobject)
        results[video_uid]=objectres
    with open(configs.output_dir, 'w') as f:
        json.dump(results, f, indent=4)
            
        
    


