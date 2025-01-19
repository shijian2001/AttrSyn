from PIL import Image
from .query_utils import generate_dsg
import openai
from .xiaoai_utils import openai_setup, openai_completion
from .parse_utils import parse_tuple_output, parse_dependency_output, parse_question_output

import json
import os

def save_qid_data_to_json(qid_data, filename='qid_data.json'):
    """
    Save the qid_data to a JSON file if the same prompt doesn't exist.
    
    Parameters:
    - qid_data: The dictionary containing the data to save.
    - filename: The name of the file to save the data to.
    """
    # Read existing JSON data
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    else:
        # If the file doesn't exist, initialize it as an empty list
        existing_data = []

    # Check if the same prompt already exists
    prompt_exists = any(item['prompt'] == qid_data['prompt'] for item in existing_data)

    if not prompt_exists:
        # If the same prompt doesn't exist, add the data to the existing data
        existing_data.append(qid_data)

        # Write the updated data back to the file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)
        
        print(f"Data with prompt '{qid_data['prompt']}' has been written to {filename}.")
    else:
        print(f"Data with prompt '{qid_data['prompt']}' already exists. No data written.")

def load_qid_data_from_json(prompt, filename='qid_data.json'):
    """
    Load the qid_data from a JSON file based on the prompt.
    
    Parameters:
    - prompt: The prompt to search in the JSON file.
    - filename: The name of the file to load the data from.
    
    Returns:
    - The qid_data dictionary if the prompt is found, otherwise None.
    """
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        
        # Find the matching prompt and return the corresponding data
        for item in existing_data:
            if item['prompt'] == prompt:
                return item
    return None

def generate_qid_data_from_prompt(input_text_prompt):
    """
    Generate questions based on the descriptive text and return a dictionary 
    containing prompt, qid2tuple, qid2dependency, and qid2question.
    
    Parameters:
    - input_text_prompt: A prompt describing the image to be generated.
    
    Returns:
    - qid_data: A dictionary containing prompt, qid2tuple, qid2dependency, and qid2question.
    """
    # Get the API key
    api_key = ''

    assert api_key is not None, "API key not found. Please ensure it's in the key file."
    
    # Try to load qid_data from the JSON file first
    qid_data = load_qid_data_from_json(input_text_prompt)
    
    # If the data for the prompt is found in the file, return it
    if qid_data:
        print(f"Data for prompt '{input_text_prompt}' found in JSON file. Returning existing data.")
        return qid_data

    # If the data for the prompt is not in the file, generate new qid_data
    id2prompts = {'custom_0': {'input': input_text_prompt}}

    id2tuple_outputs, id2question_outputs, id2dependency_outputs = generate_dsg(
        id2prompts,
        generate_fn=lambda prompt: openai_completion(prompt, model="gpt-4", temperature=0, max_tokens=500, api_key=api_key)
    )

    qid2tuple = parse_tuple_output(id2tuple_outputs['custom_0']['output'])
    qid2dependency = parse_dependency_output(id2dependency_outputs['custom_0']['output'])
    qid2question = parse_question_output(id2question_outputs['custom_0']['output'])

    qid_data = {
        'prompt': input_text_prompt,
        'qid2tuple': qid2tuple,
        'qid2dependency': qid2dependency,
        'qid2question': qid2question
    }

    # Save the generated data to the JSON file
    save_qid_data_to_json(qid_data)

    return qid_data
