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
    
    return processed_lines

#we define a method to ask any prompt to llama
def ask_llama(device,tokenizer, model,prompt, maxl=1500, temp=0.7):
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
        Given the following narrations describing the actions of a person, generate a set of simple queries (one per narration) that could be answered by looking at the video segments corresponding to these narrations
        Output the list of questions maximum 5 line in a clear and concise manner. Do not include duplicates or irrelevant terms do not produce othe examples.

        **Example 1:**
        Narrations:
        the person cuts vegetables with a knife
        the person places the knife on the table
        the person takes a bowl from the cupboard
        the person pours the vegetables into the bowl
        the person washes their hands in the sink
        Response:
        What is the person using to cut vegetables?
        Where does the person place the knife?
        What does the person take from the cupboard?
        What is being poured into the bowl?
        Where does the person wash their hands?

        ---

        **Example 2:**
        Narrations:
        the person opens a book
        the person flips through the pages
        the person picks up a pen from the table
        the person writes something in the book
        the person closes the book
        Response:
        What does the person open at the start?
        What is the person doing with the pages?
        What item does the person pick up from the table?
        What does the person use to write in the book?
        What does the person do after writing?

        ---

        **Example 3:**
        Narrations:
        the person ties their shoelaces
        the person puts on a jacket
        the person picks up an umbrella from the floor
        the person opens the door
        the person steps outside
        Response:
        What is the person tying?
        What does the person put on after tying their shoelaces?
        What does the person pick up from the floor?
        What does the person open before stepping outside?
        Where does the person go after opening the door?

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
            
        
    

