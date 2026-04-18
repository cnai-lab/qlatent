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

from qlatent.qabstract.qabstract_torch import *
from qlatent.qabstract.qabstract_torch import SCALE, DIMENSIONS, FILTER, IDXSELECT, _filter_data_frame

## Wait for colab to upgrade to Python 3.11
##IDXSELECT_for_consistency_checks = Annotated[Tuple[Union[slice,Annotated[List[int], MinLen(2)]], MinLen(2)]


def tensor_postprocess(model_outputs, entailment_id=0, framework='pt', multi_label=False):
    candidate_labels = [outputs["candidate_label"] for outputs in model_outputs]
    sequences = [outputs["sequence"] for outputs in model_outputs]
    # Concatenate logits while preserving tensors
    if framework == "pt":
        logits = torch.cat([output["logits"].float() for output in model_outputs], dim=0)
    else:
        raise ValueError("Unsupported framework. Only 'pt' (PyTorch) is supported.")

    N = logits.shape[0]
    n = len(candidate_labels)
    num_sequences = N // n
    reshaped_outputs = logits.view(num_sequences, n, -1)

    if multi_label or len(candidate_labels) == 1:
        contradiction_id = -1 if entailment_id == 0 else 0
        entail_contr_logits = reshaped_outputs[..., [contradiction_id, entailment_id]]
        scores = torch.softmax(entail_contr_logits, dim=-1)[..., 1]
    else:
        entail_logits = reshaped_outputs[..., entailment_id]
        scores = torch.softmax(entail_logits, dim=-1)

    top_inds = torch.argsort(scores[0], descending=True)
    return {
        "sequence": sequences[0],
        "labels": [candidate_labels[i] for i in top_inds],
        "scores": scores[0][top_inds],
    }


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
#         print("QMNLI RUN 1")
        super().run(model)
#         print("QMNLI RUN 2")
        if self.model.entailment_id == -1:
            raise Exception("""The entailment id of the MNLI model is not determine.  please update label name to {"CONTRADICTION", "ENTAILMENT", "NEUTRAL"} in self.model.config""")
        
        T = time.time()
        coo = []
        p = []      
        p_torch = torch.empty(0)
        p_torch_list = []
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

        ## Apply softmax on logits for torch version
        for res, (context, answer) in zip(results.logits, sequences):  
            output = {
                'logits': res.cpu().view(1, -1).clone().requires_grad_(), # Use requires_grad_() to track gradients
                "candidate_label": answer,
                "sequence": context,
            }
            # Instead of passing to postprocess, you might want to use torch.softmax or any other operation
            scores = tensor_postprocess([output], entailment_id = self.model.entailment_id, framework = self.model.framework, multi_label=False)['scores']
            p_torch_list.append(scores[0]) 

        p_torch = torch.stack(p_torch_list)
        self._t = p_torch
        
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