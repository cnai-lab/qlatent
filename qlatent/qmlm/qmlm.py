import torch
import pandas as pd
import numpy as np
from pprint import pprint
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import pipeline
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer
import scipy
import itertools
import re

from ..qabstract.qabstract import *
from ..qabstract.qabstract import SCALE, DIMENSIONS, FILTER, IDXSELECT, _filter_data_frame


class QMLM(QABSTRACT):    
    def __init__(self,
                 template:str,
                 dimensions:DIMENSIONS = {},
                 model:pipeline = None,
                 p=None,
                 index=None,
                 scale='intensifier',
                 descriptor = {}):
        super().__init__(dimensions, model, p, index, scale, descriptor=descriptor)
        
        self._index = index
        self._scale = scale
        self._descriptor['query'] = template
        self._template = template
        QMLM._qregister[self.__class__.__name__] = self
        
    
    def ans_logits(self, result):
        ans = [r['token_str'] for r in result]
        prob = torch.tensor([r['score'] for r in result])
        return dict(zip(ans, prob))

    
    def mlm_ans(mlm_esult):
        return sorted([r['token_str'] for r in mlm_esult])


    def chain_prob(self, query, word, j=0):
        tokens = self.model.tokenizer(word, add_special_tokens = False)
        token_str = self.model.tokenizer.convert_ids_to_tokens(tokens['input_ids'])
        token_count = len(token_str)
        mask_token = self.model.tokenizer.mask_token  

        if token_count > 1 and token_str[1].startswith('##'):                    
            mask = token_count*mask_token
        else:
            mask = ' '.join(token_count*[mask_token])

        mask_num = query.count(mask_token) 
        start_index = [m.start() for m in re.finditer(re.escape(mask_token), query)][j]
        query = query[:start_index] + mask + query[start_index + len(mask_token):]

        ans = self.model(query, targets=token_str[:1])

        if token_count > 1:
            probs = [ans[j][0]["score"]]
            for i in range(1, token_count): 
                start_index = [m.start() for m in re.finditer(re.escape(mask_token), query)][j] 
                token = token_str[i - 1]
                mask_len = len(mask_token)
                query = query[:start_index] + token + query[start_index + mask_len:]
                ans = self.model(query, targets=token_str[i:i+1])

                if mask_num == 1:
                    if token_count - i == 1:
                        probs.append(ans[j]["score"])
                    else:
                        probs.append(ans[j][0]["score"])
                else:
                    probs.append(ans[j][0]["score"])
        else:
            if mask_num == 1:
                probs = [ans[0]["score"]]
            else:
                probs = [ans[j][0]["score"]]
        prob = np.prod(probs)
        return prob
    
    def run(self, model= None):
        super().run(model)
        tokens_to_ids_dict = {}
        str_to_tokens_dict = {}
        mask_token = self.model.tokenizer.mask_token
        for key, value in self._keywords.items():
            for i in value:
                tokens = self.model.tokenizer(i, add_special_tokens = False)
                token_str = self.model.tokenizer.convert_ids_to_tokens(tokens['input_ids'])
                tokens_to_ids_dict[tuple(token_str)] = tokens['input_ids'];
                str_to_tokens_dict[i] = tuple(token_str)
        T = time.time()
        coo = []
        p = []
        for kmap,kcoo in zip(self._keywords_map,self._keywords_grid_idx):
            dimensions_probs = []
#             dimension_name = self._index[0]
            dimension_name = self._scale
            dimension_value = kmap[dimension_name]
            index_value = kmap[self._index[0]]
            masked_kmap = kmap.copy()
            masked_kmap[dimension_name] = self.model.tokenizer.mask_token 
            prob = self.chain_prob(self._template.format_map(masked_kmap), dimension_value, 0)
            
            dimensions_probs.append(prob)
            kmap[dimension_name] = dimension_value
                
            # avg_probs_dimensions is the averege of the propability of the sentence
            avg_prob_dimensions = np.mean(dimensions_probs)
            p.append(avg_prob_dimensions)
            coo.append(kcoo)

        coo = torch.stack(coo).T
        assert torch.all(torch.eq(coo.T, self._keywords_grid_idx))

        self._pdf["P"] = p
        
        self._T = time.time() - T
        self.result = self
        return self.result


from typing import Dict

# The static property QCOLA._qregister will contain one instance (the last one) of every subtype of QCOLA.
# This will allow to automatically registering questions and reusing them with various models when the time comes.
QMLM._qregister: Dict[str, QMLM] = {}

class QQQ():
    pass

class QQW():
    pass



