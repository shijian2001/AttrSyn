import os
import re
import json
from collections import OrderedDict
from autogen import OpenAIWrapper

# get LLM by OpenAIWrapper
os.environ["OPENAI_API_KEY"] = ""
client = OpenAIWrapper()

# get class_names
classes_file = "/linxindisk/AttrSyn/data/real/CUB-200-Painting/cub_list_drawing.txt"
CLASSES = []

with open(classes_file, "r") as file:
    for line in file:
        line = line.strip()
        words = line.split()
        class_name = " ".join(words[:-1])
        CLASSES.append(class_name)

# define system prompt template
prompt_template = "List 5 most common background environments related to {}, strictly follow the format of one lowercase word with commas. Example: flying, sky, ..."

# get 5 attr values through LLM
def get_answer(bird_class):
    prompt = prompt_template.format(bird_class)
    response = client.create(
        messages=[
            {"role": "user", "content": prompt}
        ], 
        model="gpt-4-0125-preview",
        cache_seed = None
    )
    answer = client.extract_text_or_completion_object(response)[0]
    return answer

# check answer format
def check_format(answer):
    pattern = r'^([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)$'
    match = re.match(pattern, answer)
    if match:
        words = [word.strip() for word in match.groups()]
        return words
    return None

# re-generate if wrong format 
def get_valid_answer(bird_class):
    answer = get_answer(bird_class)
    if check_format(answer):
        return check_format(answer)
    else:
        print("!!!NOT MEET THE FORMAT!!!!", answer)
        return get_valid_answer(bird_class)

result = {}
for bird_class in CLASSES:
    print("*******************************")
    answer = get_valid_answer(bird_class)
    # print(answer)
    result[bird_class] = answer
    print(result[bird_class])

with open('class_background.json', 'w') as file:
    json.dump(result, file, indent=4)