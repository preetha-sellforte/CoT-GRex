# ########### Python 3.2 #############
import requests
import argparse
import pandas as pd
import time
from tqdm import tqdm
from pathlib import Path

parser = argparse.ArgumentParser(description='Define experiment arguments')
parser.add_argument('--prompt_dir', type=str, default='path/to/prompt', help='define the path to the prompt')
parser.add_argument('--data', type=str, default='HyperRED', help='what dataset are you using')
parser.add_argument('--model', type=str, default='GPT3.5', help='type of model we are using')


args = parser.parse_args()


with open(args.prompt_dir, 'r', encoding='utf-8') as file:
  PROMPT = file.read()




if args.data == 'close-domain':
  test_data = pd.read_csv('path/to/dataset', sep='\t', encoding='latin-1')
elif args.data == 'HyperRED':
  test_data = pd.read_csv('path/to/HyperRED/dataset', sep='\t', encoding='utf-8')

RE_pred = []

url = "https://aalto-openai-apigw.azure-api.net/v1/chat"

headers = {
    "Content-Type": "application/json",
    "Ocp-Apim-Subscription-Key": "$API_KEY"
}

data = {
    "messages": [
        {
            "role": "system",
            "content": PROMPT
        },
        {
            "role": "user",
            "content": ""
        },
        {
            "role": "user",
            "content": "Print the entities and relations in this sentence in the format (Entity1, Entity2, Relation, Qualifier Key: Qualifier Value ). You are expected to strictly adhere to this format."
        }
    ]
}

for index, row in tqdm(test_data.iterrows(), total=len(test_data)):
    # Access the data in each row using column names
    context = row['Context']
    data['messages'][1]['content'] = context
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        try:
            RE_pred.append(response.json()['choices'][0]['message']['content'])
        except:
           RE_pred.append("None")
    else:
        print(f"Request failed with status code {response.status_code}")
        RE_pred.append("None")
    time.sleep(7)

test_data[f'RE_pred'] = RE_pred

pred_name = f'/scratch/work/{args.user}/knowledge-graph/predictions/{args.data}/random_prediction_{args.model}_{Path(args.prompt_dir).stem}.tsv'
print(pred_name, flush=True)
test_data.to_csv(pred_name, sep='\t', encoding='utf-8')

print("CSV created", flush=True)

