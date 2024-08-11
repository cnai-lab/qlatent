import torch
import pandas as pd
import numpy as np
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


class QCOLA(QABSTRACT):

    # Static property QCOLA._qregister is defined after this class
    # It will contain one instance (the last one) of every subtype of QCOLA.
    # This will allow to automatically register questions and reuse them with various models when the time comes.

    def __init__(self,
                 template: str,
                 dimensions:DIMENSIONS = {},
                 model: pipeline = None,
                 p=None,
                 index=None,
                 scale='intensifier',
                 descriptor = {}):
        super().__init__(dimensions, model, p, index, scale, descriptor=descriptor)
        
        self._index = index
        self._scale = scale
        self._descriptor['query'] = template
        self._template = template
#         self.intensifier_names = list(dimensions[scale].keys())
#         self.emotions = list(dimensions[index].keys())
        QCOLA._qregister[self.__class__.__name__] = self

        
    def run(self, model= None):
        super().run(model)
        T = time.time()
        coo = []
        p = []
        for kmap,kcoo in zip(self._keywords_map,self._keywords_grid_idx):
            query = self._template.format_map(kmap)
            ans = self.model(query)[0].get('score')
            p.append(torch.Tensor([ans]).squeeze().cpu().item())
            coo.append(kcoo)

        coo = torch.stack(coo).T
        assert torch.all(torch.eq(coo.T, self._keywords_grid_idx))

        self._pdf["P"] = p
        
        self._T = time.time() - T
        self.result = self
        return self.result


    def ans_logits(self, result):
        """
        result = [{'label': 'LABEL_1', 'score': 0.8850699663162231}, ... ]
        """
        ans = [r['score'] if r['label'] == 'LABEL_1' else 1 - r['score'] for r in result]
        return ans


from typing import Dict

# The static property QCOLA._qregister will contain one instance (the last one) of every subtype of QCOLA.
# This will allow to automatically registering questions and reusing them with various models when the time comes.
QCOLA._qregister: Dict[str, QCOLA] = {}

