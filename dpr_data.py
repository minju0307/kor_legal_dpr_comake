from tqdm import tqdm
import torch
from torch import tensor as T
from torch.nn.utils.rnn import pad_sequence
import os
import json
import re
import logging
from typing import Iterator, List, Sized, Tuple
import pickle
from transformers import AutoTokenizer
import pandas as pd

# set logger
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/log.log",
    level=logging.DEBUG,
    format="[%(asctime)s | %(funcName)s @ %(pathname)s] %(message)s",
)
logger = logging.getLogger()

# tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class DPRDataset:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.data_tuples = []
        self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
        self.pad_token_id = self.tokenizer.pad_token_id
        self.load()

    @property
    def dataset(self) -> List[Tuple]:
        return self.tokenized_tuples

    def load(self):
        '''데이터를 불러와서 토크나이징 하여 저장합니다.'''
        self._load_data()
        logger.debug("successfully loaded data_tuples into self.data_tuples")

        # tokenizing raw dataset
        self.tokenized_tuples = [
            (self.tokenizer.encode(q, max_length=512, truncation=True), id,
             self.tokenizer.encode(p, max_length=512, truncation=True))
            for q, id, p in tqdm(self.data_tuples, desc="tokenize")
        ]

        self._save_processed_dataset()
        logger.debug("finished tokenization")

    def _load_data(self):
        '''데이터를 로드하기 (이미 매칭이 되어있는 dataframe을 불러와서 self.data_tupels에 넣어줍니다.)'''
        df = pd.read_csv(f'{self.dataset_path}')
        self.raw_csv = df
        logger.debug("data loaded into self.raw_csv")

        questions = df['question'].to_list()
        johangs = df['johang'].to_list()
        ids = df['jo_id'].to_list()

        for idx, q in tqdm(enumerate(questions), desc="making data_tuples"):
            self.data_tuples.append((q, int(ids[idx]), johangs[idx]))

    def _save_processed_dataset(self):
        '''전처리한 데이터를 저장합니다.'''
        with open(f'{self.dataset_path[:-4]}', "wb") as f:
            pickle.dump(self.tokenized_tuples, f)
        logger.debug(
            f"successfully saved self.tokenized_tuples into {self.dataset_path}"
        )


class DPRSampler_train(torch.utils.data.BatchSampler):
    '''in-batch negative학습을 위해 batch 내에 중복 answer를 갖지 않도록 batch를 구성합니다.
    sample 일부를 pass하기 때문에 전체 data 수보다 iteration을 통해 나오는 데이터 수가 몇십개 정도 적습니다.'''
    '''torch utils의 batchsampler를 상속받는다.'''

    def __init__(
            self,
            data_source: Sized,
            batch_size: int,
            drop_last: bool = False,
            shuffle: bool = True,
            generator=None,
            sampler=None
    ) -> None:

        if shuffle:  ## shuffle을 선택한 경우 Randomsampler를 사용하기
            sampler = torch.utils.data.RandomSampler(
                data_source, replacement=False, generator=generator
            )
        else:  ## 아닌 경우 sequentialsampler 사용하기
            sampler = torch.utils.data.SequentialSampler(data_source)

        
        super(DPRSampler_train, self).__init__(  ## torch utils의 batchsampler 정의하기
            sampler=sampler, batch_size=batch_size, drop_last=drop_last
        )

    def __iter__(self) -> Iterator[List[int]]:  ## batch sampler의 iter 함수 정의하기
        sampled_p_id = []
        sampled_idx = []
        for idx in self.sampler:
            item = self.sampler.data_source[idx]
            if item[1] in sampled_p_id:
                continue  # 만일 같은 answer passage가 이미 뽑혔다면 pass
            sampled_idx.append(idx)
            sampled_p_id.append(item[1])
            if len(sampled_idx) >= self.batch_size:
                yield sampled_idx
                sampled_p_id = []
                sampled_idx = []
        if len(sampled_idx) > 0 and not self.drop_last:
            yield sampled_idx

class DPRSampler_test(torch.utils.data.BatchSampler):
    '''test 시에는 같은 배치 안에 같은 passage가 정답인 경우가 있으므로 이를 고려해야 한다.'''
    '''torch utils의 batchsampler를 상속받는다.'''

    def __init__(
            self,
            data_source: Sized,
            batch_size: int,
            drop_last: bool = False,
            shuffle: bool = True,
            generator=None,
            sampler=None
    ) -> None:

        if shuffle:  ## shuffle을 선택한 경우 Randomsampler를 사용하기
            sampler = torch.utils.data.RandomSampler(
                data_source, replacement=False, generator=generator
            )
        else:  ## 아닌 경우 sequentialsampler 사용하기
            sampler = torch.utils.data.SequentialSampler(data_source)

        super(DPRSampler_test, self).__init__(  ## torch utils의 batchsampler 정의하기
            sampler=sampler, batch_size=batch_size, drop_last=drop_last
        )

    def __iter__(self) -> Iterator[List[int]]:  ## batch sampler의 iter 함수 정의하기
        sampled_p_id = []
        sampled_idx = []
        for idx in self.sampler:
            item = self.sampler.data_source[idx]
            sampled_idx.append(idx)
            sampled_p_id.append(item[1])
            if len(sampled_idx) >= self.batch_size:
                yield sampled_idx
                sampled_p_id = []
                sampled_idx = []
        if len(sampled_idx) > 0 and not self.drop_last:
            yield sampled_idx

def DPR_collator(batch: List[Tuple], padding_value: int) -> Tuple[torch.Tensor]:
    '''query, p_id, gold_passage를 batch로 반환합니다.'''
    batch_q = pad_sequence(
        [T(e[0]) for e in batch], batch_first=True, padding_value=padding_value
    )
    batch_q_attn_mask = (batch_q != padding_value).long()
    batch_p_id = T([e[1] for e in batch])[:, None]
    batch_p = pad_sequence(
        [T(e[2]) for e in batch], batch_first=True, padding_value=padding_value
    )
    batch_p_attn_mask = (batch_p != padding_value).long()
    return (batch_q, batch_q_attn_mask, batch_p_id, batch_p, batch_p_attn_mask)

if __name__ == "__main__":

    ### train
    print("train")
    ds_train = DPRDataset(dataset_path="dataset/train_with_id.csv")
    loader = torch.utils.data.DataLoader(
        dataset=ds_train.dataset,
        batch_sampler=DPRSampler_train(ds_train.dataset, batch_size=32, drop_last=False),
        collate_fn=lambda x: DPR_collator(x, padding_value=ds_train.pad_token_id)
    )
    # print(len(_dataset.tokenized_tuples))
    torch.manual_seed(123412341235)
    cnt = 0
    for batch in tqdm(loader):
        '''batch[0] : q의 input_ids, batch[1] : q의 attention mask, batch[2] : gold_passage_id , batch[3] : p의 input_ids, batch[4]: p의 attention mask '''
        cnt += batch[0].size(0)
    print(cnt)  ## 조금 적은 개수가 나올 수 있음. 최종적으로 train에 포함되는 개수
    print()


    ### dev
    print("dev")
    ds_dev = DPRDataset(dataset_path="dataset/dev_with_id.csv")
    loader = torch.utils.data.DataLoader(
        dataset=ds_dev.dataset,
        batch_sampler=DPRSampler_test(ds_dev.dataset, batch_size=32, drop_last=False),
        collate_fn=lambda x: DPR_collator(x, padding_value=ds_dev.pad_token_id),
    )
    # print(len(_dataset.tokenized_tuples))
    torch.manual_seed(123412341235)
    cnt = 0
    for batch in tqdm(loader):
        '''batch[0] : q의 input_ids, batch[1] : q의 attention mask, batch[2] : gold_passage_id , batch[3] : p의 input_ids, batch[4]: p의 attention mask '''
        cnt += batch[0].size(0)
    print(cnt)  ## 조금 적은 개수가 나올 수 있음. 최종적으로 train에 포함되는 개수
    print()

    ### test
    print("test")
    ds_test = DPRDataset(dataset_path="dataset/test_with_id.csv")
    loader = torch.utils.data.DataLoader(
        dataset=ds_test.dataset,
        batch_sampler=DPRSampler_test(ds_test.dataset, batch_size=32, drop_last=False),
        collate_fn=lambda x: DPR_collator(x, padding_value=ds_test.pad_token_id)
    )
    # print(len(_dataset.tokenized_tuples))
    torch.manual_seed(123412341235)
    cnt = 0
    for batch in tqdm(loader):
        '''batch[0] : q의 input_ids, batch[1] : q의 attention mask, batch[2] : gold_passage_id , batch[3] : p의 input_ids, batch[4]: p의 attention mask '''
        cnt += batch[0].size(0)
    print(cnt)  ## 조금 적은 개수가 나올 수 있음. 최종적으로 train에 포함되는 개수
    print()
