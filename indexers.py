import torch
import transformers
from transformers import AutoModel, AutoTokenizer, XLMRobertaTokenizer
import faiss

import os
import logging

import json
import numpy as np

from encoder import BiEncoder

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/log.log",
    level=logging.DEBUG,
    format="[%(asctime)s | %(funcName)s @ %(pathname)s] %(message)s",
)
logger = logging.getLogger()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') ## TODO 다른 GPU 지정해서 쓸 수 있도록 지우기 

def load_training_state(model, optimizer, scheduler, model_path, optim_path) -> None:
    """모델, optimizer와 기타 정보를 로드합니다"""
    model.load(model_path)
    training_state = torch.load(optim_path)
    optimizer.load_state_dict(training_state["optimizer_state"])
    scheduler.load_state_dict(training_state["scheduler_state"])

def get_faiss_index(dataset, tokenizer, model):

    ## 학습된 model의 passage로 임베딩 만들기
    vectors = []

    for line in dataset: ## To-DO batch 처리 
        dict = tokenizer.batch_encode_plus([line], return_tensors='pt', max_length=512, truncation=True)
        p_ids = dict['input_ids'].to(device)
        p_atten = dict['attention_mask'].to(device)
        model.eval()
        with torch.no_grad():
            out = model(p_ids, p_atten, "passage")

        vectors.append(np.array(out.detach().cpu()))

    vectors = np.array(vectors)
    vectors = vectors.reshape(1180, 768)

    vector_dimension = vectors.shape[1]
    print(vector_dimension)
    print(vectors)
    index = faiss.IndexFlatL2(vector_dimension)
    faiss.normalize_L2(vectors)
    index.add(vectors)

    return index

def save_faiss_index(index, path):
    faiss.write_index(index, path)

if __name__ == "__main__":
    config_dict={
        "model_path": 'legal_dpr.pt',
        "optim_path" : 'legal_dpr_optim.pt',
        "lr" : 1e-5,
        "betas" : (0.9, 0.99),
        "num_warmup_steps" : 1000,
        "num_training_steps" : 2670,
        "output_path": 'legal_dpr.index',
    }

    '''모델 로드하기 '''
    model = BiEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config_dict['lr'], betas=config_dict['betas'])
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, config_dict["num_warmup_steps"], config_dict["num_training_steps"]
    )

    load_training_state(model, optimizer, scheduler, config_dict["model_path"], config_dict["optim_path"])

    '''데이터셋과 토크나이저 불러오기'''
    with open("dataset/sangbub_jo_prompt.json") as f:
        sangbub = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base") 

    '''모델에서 임베딩 얻어서 faiss index 얻기'''
    index = get_faiss_index(sangbub, tokenizer, model)
    print(index)
    print()

    '''faiss index writing 하기'''
    save_faiss_index(index, config_dict['output_path'])

    print(config_dict['output_path'])