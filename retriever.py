import torch
import transformers
from transformers import AutoModel, AutoTokenizer, XLMRobertaTokenizer
import faiss

import os
import logging

import pandas as pd
import numpy as np
import json

from encoder import BiEncoder

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/log.log",
    level=logging.DEBUG,
    format="[%(asctime)s | %(funcName)s @ %(pathname)s] %(message)s",
)
logger = logging.getLogger()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_training_state(model, optimizer, scheduler, model_path, optim_path) -> None:
    """모델, optimizer와 기타 정보를 로드합니다"""
    model.load(model_path)
    training_state = torch.load(optim_path)
    optimizer.load_state_dict(training_state["optimizer_state"])
    scheduler.load_state_dict(training_state["scheduler_state"])

def retrieval(tokenizer, model, data_path, index, top_k):
    dataset = pd.read_csv(data_path)

    output={'query': [], 'ref':[], 'pred': [], 'ref_content':[], 'pred_content':[]}
    queries = dataset['question'].tolist()
    output['query'] = queries
    output['ref'] = dataset['jo_id'].tolist()
    output['ref_content'] = dataset['johang'].tolist()

    for query in queries: 
        dict = tokenizer.batch_encode_plus([query], return_tensors='pt')
        q_ids = dict['input_ids'].to(device)
        q_atten = dict['attention_mask'].to(device)
        model.eval()
        with torch.no_grad():
            query_embedding = model(q_ids, q_atten, "query")
        search_index = list(index.search(np.float32(query_embedding.detach().cpu()), top_k)[1][0])
        output['pred'].append('\n'.join([str(i) for i in search_index]))
        output['pred_content'].append(get_content(search_index))

    output=pd.DataFrame(output)
    output_dir = 'retrieval_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output.to_csv(f"{output_dir}/{data_path[8:-4]}_{top_k}.csv", index=False) 

def get_content(search_index):
    with open("dataset/sangbub_jo_prompt.json") as f:
        reference = json.load(f)

    ref_texts=[]
    for idx in search_index:
        ref_texts.append(reference[idx])

    return '\n\n'.join(ref_texts)

if __name__ == "__main__":
    config_dict = {
        "model_path": 'legal_dpr.pt',
        "optim_path" : 'legal_dpr_optim.pt',
        "lr": 1e-5,
        "betas": (0.9, 0.99),
        "num_warmup_steps" : 1000,
        "num_training_steps": 2670,
        "output_path": 'legal_dpr.index',
        "test_set": 'dataset/test_with_id.csv',
        'top_k': 3
    }

    '''모델 로드하기 '''
    model = BiEncoder().to(device)
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base") 
    optimizer = torch.optim.Adam(model.parameters(), lr=config_dict['lr'], betas=config_dict['betas'])
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, config_dict["num_warmup_steps"], config_dict["num_training_steps"]
    )
    load_training_state(model, optimizer, scheduler, config_dict["model_path"], config_dict["optim_path"])

    '''index 불러오기'''
    index = faiss.read_index(config_dict['output_path'])
    retrieval(tokenizer, model, config_dict['test_set'], index, config_dict["top_k"])