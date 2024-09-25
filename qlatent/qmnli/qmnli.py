import torch
import pandas as pd
import numpy as np
import time
import sklearn as sk
import itertools
import sys
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from functools import partial
from overrides import override
from typing import *
import copy
from typeguard import check_type
from numbers import Number
from transformers import pipeline
import os
from transformers.tokenization_utils import TruncationStrategy

from ..qabstract.qabstract import *
from ..qabstract.qabstract import SCALE, DIMENSIONS, FILTER, IDXSELECT, _filter_data_frame

## Wait for colab to upgrade to Python 3.11
##IDXSELECT_for_consistency_checks = Annotated[Tuple[Union[slice,Annotated[List[int], MinLen(2)]], MinLen(2)]


class QMNLI(QABSTRACT):

    
    # Static property QMNLI._qregister is defined after this class
    # It will contain one instance (the last one) of every subtype of QMNLI.
    # This will allow to automatically register questions and reuse them with various models when the time comes.

    def __init__(self,
                 context_template: str = " ",
                 answer_template: str = " ",
                 dimensions:DIMENSIONS = {},
                 model: pipeline = None,
                 p=None,
                 index=None,
                 scale='intensifier',
                 descriptor = {}):
        """
        Templates (both context and answer) are parsed before execution to prepare the list of template fields used in each string.
        During execution a list of strings and a list of answer strings are created basaed on cartesian product of the respective field values.
        The field values are extracted from dimensions and intensifiers.
        """        
        super().__init__(dimensions, model, p, index, scale, descriptor)
        
        self._descriptor['query'] = context_template+"->"+answer_template

        self._context_template = context_template
        self._answer_template = answer_template
        QMNLI._qregister[self.__class__.__name__]=self
        

    def run(self, model=None):
        super().run(model)
        if self.model.entailment_id == -1:
            raise Exception("""The entailment id of the MNLI model is not determine.  please update label name to {"CONTRADICTION", "ENTAILMENT", "NEUTRAL"} in self.model.config""")
        T = time.time()
        coo = []
        p = []      

        ## prepare all premise hypothesis pairs
        sequences = []
        for kmap,kcoo in zip(self._keywords_map,self._keywords_grid_idx):
            context = self._context_template.format_map(kmap)
            answer = self._answer_template.format_map(kmap)
            sequences.append((context, answer))
            coo.append(kcoo)
        
        ## Calculate batch entailment probabilities
        inputs = self.model.tokenizer(
                sequences,
                add_special_tokens=True,
                return_tensors="pt",
                padding=True,
                truncation=TruncationStrategy.ONLY_FIRST,
            )
        
        model_inputs = {k: inputs[k].to(self.model.device) for k in self.model.tokenizer.model_input_names}
        results = self.model.model(**model_inputs)
        
        ## Apply softmax on logits
        for res, (context, answer) in zip(results.logits, sequences):  
            output = {
                'logits': res.detach().cpu().view(1, -1),
                "candidate_label": answer,
                "sequence": context,
            }
            p.append(self.model.postprocess([output], multi_label=False)['scores'][0])

        
        coo = torch.stack(coo).T
        assert torch.all(torch.eq(coo.T, self._keywords_grid_idx))
        
        self._pdf["P"] = p
        
        self._T = time.time() - T
        self.result = self
        return self.result

    
# The static property QMNLI._qregister will contain one instance (the last one) of every subtype of QMNLI.
# This will allow to automatically registering questions and reusing them with various models when the time comes.


QMNLI._qregister:Dict[str,QMNLI]={}


class _QMNLI(QMNLI):
    
    index = ["emotion"]
    scale = "intensifier"

    def __init__(self, context, template, emo_pos, emo_neg, intensifiers, **kwargs):
        
        super().__init__(
            context_template=context,
            answer_template=template,
            dimensions={"emotion":dict_pos_neg(emo_pos, emo_neg,1.0), "intensifier":intensifiers, },
            **kwargs
        )

        # self.scale = "intensifier"
        # self.index = "emotion"
        self.emo_pos = emo_pos
        self.emo_neg = emo_neg
        self.grouping = [{"emotion":emo_pos},{"emotion":emo_neg}]
        

    def report(self,scale:Union[str,int]="intensifier", index="emotion", filters:Dict[str,FILTER]={"unfiltered":{}}, grouping:List[FILTER]=[],):
        if len(grouping)==0:
            grouping = self.grouping
        return super().report(scale=scale,index=index,filters=filters,grouping=grouping)


    def softmax(self, dim:Union[str,int,List[str],List[int]]="intensifier"):
        return super().softmax(dim=dim)

    
    def to_dataframe(self, scale:Union[str,int]="intensifier", index:List[str]="emotion", filter:FILTER={}, categories:Dict[str,Dict[str,str]]={}):
        return super().to_dataframe(scale=scale, index=index, filter=filter, categories=categories )
