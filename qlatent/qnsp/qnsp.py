import torch
import pandas as pd
import numpy as np
import torch
import time
from pprint import pprint
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import pipeline
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer
import scipy
import sklearn as sk
import itertools
from ..qabstract.qabstract import *
from ..qabstract.qabstract import SCALE, DIMENSIONS, FILTER, IDXSELECT, _filter_data_frame

from transformers import BertTokenizer, BertForNextSentencePrediction
import torch


class NextSentencePredictionPipeline():
  def __init__(self, model_name='bert-base-uncased', device=0):
    self.tokenizer = BertTokenizer.from_pretrained(model_name)
    self.model = BertForNextSentencePrediction.from_pretrained(model_name, return_dict=True)
    self.model.to(device)

  def __call__(self, sentence_pairs, device=0):
    results = []
#     print(sentence_pairs)
#     for prompt, next_sentence in sentence_pairs:
    prompt = sentence_pairs[0]
    next_sentence = sentence_pairs[1]
    encoding = self.tokenizer(prompt, next_sentence, return_tensors='pt').to(device)
    next_sentence_label = torch.LongTensor([1]).to(device)
    outputs = self.model(**encoding, labels=next_sentence_label)
#     outputs = self.model(**encoding, next_sentence_label=torch.LongTensor([1]))
    logits = outputs.logits
    results.append(logits)
    return torch.vstack(results).detach()


class QNSP(QABSTRACT):
    # Static property QNSP._qregister is defined after this class
    # It will contain one instance (the last one) of every subtype of QNSP.
    # This will allow to automatically register questions and reuse them with various models when the time comes.

    def __init__(self, next_sentence: str, prompt: str, dimensions:DIMENSIONS = {},
                model: pipeline = None,
                 p=None,
                 index=None,
                 scale='intensifier',
                 descriptor = {}):
        super().__init__(dimensions, model, p, index, scale, descriptor)
#         self.nsp = model
        self._next_sentence = next_sentence
        self._prompt = prompt
        self._descriptor['query'] = prompt+"->"+next_sentence
        QNSP._qregister[self.__class__.__name__] = self

    def ans_logits(self, result):
        return torch.nn.functional.softmax(result, dim=1)[:,0]

    def run(self, model=None):
        super().run(model)
        T = time.time()
        coo = []
        p = []  
        for kmap,kcoo in zip(self._keywords_map,self._keywords_grid_idx):
            prompt = self._prompt.format_map(kmap)
            next_sentence = self._next_sentence.format_map(kmap)
            val = torch.softmax(self.model([prompt, next_sentence]), dim=1)[0]
            prob = val[0].item()
            p.append(torch.Tensor([prob]).squeeze().cpu().item())
            coo.append(kcoo)

        coo = torch.stack(coo).T
        assert torch.all(torch.eq(coo.T, self._keywords_grid_idx))

        self._pdf["P"] = p
#         print(self._pdf)
        self._T = time.time() - T
        self.result = self
#         print(self.result)
        return self.result


from typing import Dict

# The static property QNSP._qregister will contain one instance (the last one) of every subtype of QNSP.
# This will allow to automatically registering questions and reusing them with various models when the time comes.
QNSP._qregister: Dict[str, QNSP] = {}
