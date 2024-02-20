import re
import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


path_to_json = 'config.json'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

f = open(path_to_json, encoding="utf8")     
config = json.load(f)
 
# Cache dir and inference parameters.
max_length = config['max_length']
temperature = config['temperature']
top_k = config['top_k']
custom_cache_directory = config['cache_dir']
model_path = config['model_name']


# Load model and tokenizer.
fine_tuned_model = AutoModelForSeq2SeqLM.from_pretrained(model_path, cache_dir=custom_cache_directory)
fine_tuned_model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=custom_cache_directory)


# Beautify the response with regex, by leaving only the completed sentences.
def post_process_response(query) -> str:
    query = re.sub('\n\d+\.', '', query)
    return re.search('(.|\n)+\.', query).group(0)


while True:
    user_input = input('Please type something:')
    input_ids = tokenizer.encode(user_input, return_tensors="pt").to(device)
    
    context = ''
    context_tensor = tokenizer.encode(context, return_tensors='pt').to(device)
    
    input_ids  = torch.cat([input_ids, context_tensor], dim=1)
    response_ids = fine_tuned_model.generate(input_ids, max_length=max_length,num_beams=5, pad_token_id=tokenizer.eos_token_id, temperature=temperature, top_k=top_k, do_sample=True)
    response_ids = response_ids.to(device)
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    
    response = post_process_response(response)    
    print(response + '\n\n')
