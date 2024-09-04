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
import sys
from functools import partial
from overrides import override
from abc import *
import copy
from string import Formatter
from typing import * 
from typeguard import check_type
from numbers import Number
from typing import Dict
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import unittest
import seaborn as sns
import gc

from ..qlatent.qmnli import qmnli
from ..qlatent.qmnli.qmnli import *


device = 0 if torch.cuda.is_available() else -1
print(device)


SCALE = Dict[str,Number] 
DIMENSIONS = Dict[str,SCALE]
FILTER = Dict[str,Collection[str]]
IDXSELECT = Tuple[Union[slice,List[int]]]

intensifier_weights:SCALE = {
    'a bit':1,
    'slightly':1,
    'somewhat':1,
    'modereately':2,
    'rather':2,
    'pretty':3,
    'quite':3,
    'very':4,
    'extremely':4,
    'completely':5,
    'absolutely':5,
    'totally':5
}

frequency_weights:SCALE = {
    'never':1,
    'very rarely':1,
    'seldom':2,
    'rarely':2, 
    'occasionally':3,
    'sometimes':3,
    'usually':3,
    'frequently':4,
    'often':4,
    'very frequently':5,
    'always':5,
    'constantly':5,
}

frequency_weights_after_emo:SCALE = {
    'during non of the days':1,
    'during a day or two':2,
    'for several days':2, 
    'during some of the days':3,
    'several times':3,
    'less than half the time':3,
    'about half the time':3,
    'nearly half the time':3,
    'more than half the time':4,
    'often':4,
    'nearly every day':5,
    'every day':5,
    'very often':5,
    'all the time':5,
}

v2_frequency_weights:SCALE = {
    'during none of the days':1,
    'for several days':2, 
    'more than half the time':3,
    'nearly every day':4,
}

frequency_weights_before_emo:SCALE = {
    'never':1,
    'very rarely':1,
    'during non of the days':1,
    'seldom':2,
    'rarely':2,
    'during a day or two':2,
    'for several days':2,
    'occasionally':3,
    'sometimes':3,
    'during some of the days':3,
    'several times':3,
    'less than half the time':3,
    'about half the time':3,
    'nearly half the time':3,
    'frequently':4,
    'more than half the time':4,
    'often':4,
    'very frequently':5,
    'always':5,
    'constantly':5,
    'nearly every day':5,
    'every day':5,
    'very often':5,
    'all the time':5,
}


intensifiers_fraction_without_none:SCALE={
            "few":1,
            "some":2,
            "many":3,
            "most":4,
            "all":5,
        }

intensifiers_fraction:SCALE={
            "none of the":1,                            
            "few":2,
            "some":3,
            "many":4,
            "most":4,
            "all":5,
        }


def print_gradient(df):
    cm = sns.light_palette("green", as_cmap=True)
    s = df.style.background_gradient(cmap=cm)
    s = s.set_precision(4)
    return s


def dict_same_weight(w,ks):
    return dict(zip(ks,[w]*len(ks)))

mnli = pipeline("zero-shot-classification",device=device, model="typeform/distilbert-base-uncased-mnli")
mnli.model_identifier = "typeform/distilbert-base-uncased-mnli"


kw_long_male_plural = ["men","boys","gentelmen"]
kw_long_male_singular = ["man","boy","gentelman"]
kw_long_female_plural = ["women","girls","ladies"]
kw_long_female_singular = ["woman","girl","lady"]
kw_short_male_plural = ["men"]
kw_short_male_singular = ["man"]
kw_short_female_plural = ["women"]
kw_short_female_singular = ["woman"]

dict_long_gender_plural      = dict(dict_same_weight(1,kw_long_male_plural),    **dict_same_weight(1,kw_long_female_plural))
dict_long_gender_singluar    = dict(dict_same_weight(1,kw_long_male_singular),  **dict_same_weight(1,kw_long_female_singular))
dict_short_gender_plural     = dict(dict_same_weight(1,kw_short_male_plural),   **dict_same_weight(1,kw_short_female_plural))
dict_short_gender_singular   = dict(dict_same_weight(1,kw_short_male_singular), **dict_same_weight(1,kw_short_female_singular))



ASI = []

def split_question(Q, index, scales, softmax, filters):
    result = []
    for s in scales: 
        q = QCACHE(Q(index=index, scale=s))
        for sf in softmax:
            for f in filters:
                if sf:
                    qsf = QSOFTMAX(q,dim=[s])
                else:
                    qsf = QPASS(q,descupdate={'softmax':''})
                qsf_f = QFILTER(qsf,filters[f],filtername=f)
                print(s,sf,f)
                result.append(qsf_f)
    return result


gc.collect()
torch.cuda.empty_cache()


kw_fairness = ["fairness", "justice", "equal chances", "equality"]
kw_favors   = ['favors', "better chances", "benefits", "assistance"]
dict_objective = dict(dict_same_weight(1,kw_fairness), **dict_same_weight(-1,kw_favors))


asiq2_dimensions = {'objective': {'fairness': 1,
  'justice': 1,
  'equal chances': 1,
  'equality': 1,
  'favors': -1,
  'better chances': -1,
  'benefits': -1,
  'assistance': -1},
 'gender': {'men': 1, 'women': 1},
 'frequency': {'never': 1,
  'very rarely': 1,
  'seldom': 2,
  'rarely': 2,
  'occasionally': 3,
  'sometimes': 3,
  'usually': 3,
  'frequently': 4,
  'often': 4,
  'very frequently': 5,
  'always': 5,
  'constantly': 5},
 'fraction': {'none of the': 1,
  'few': 2,
  'some': 3,
  'many': 4,
  'most': 4,
  'all': 5}}

asiq2_field_names = ['objective', 'gender', 'frequency', 'fraction']
asiq2_scale = 'frequency'
asiq2_index = ['objective']
asiq2_context_template = '{fraction} {gender} are asking for gender equality.'
asiq2_answer_template = '{frequency} to get {objective}'



asiq2_descriptor = {'Questionnair': 'ASI',
 'Factor': 'H',
 'Ordinal': 2,
 'Original': 'H 2. Many women are actually seeking special favors, such as hiring policies that favor them over men, under the guise of asking for "equality."',
 'scale': 'frequency',
 'index': "['objective']",
 'query': '{fraction} {gender} are asking for gender equality.->{frequency} to get {objective}'}



asiq2_keywords = {'fraction': ['none of the', 'few', 'some', 'many', 'most', 'all'],
 'frequency': ['never',
               'very rarely',
               'seldom',
               'rarely',
               'occasionally',
               'sometimes',
               'usually',
               'frequently',
               'often',
               'very frequently',
               'always',
               'constantly'],
 'gender': ['men', 'women'],
 'objective': ['fairness',
               'justice',
               'equal chances',
               'equality',
               'favors',
               'better chances',
               'benefits',
               'assistance']}


asiq_keywords_indices = {'fraction': {'all': 5,
              'few': 1,
              'many': 3,
              'most': 4,
              'none of the': 0,
              'some': 2},
 'frequency': {'always': 10,
               'constantly': 11,
               'frequently': 7,
               'never': 0,
               'occasionally': 4,
               'often': 8,
               'rarely': 3,
               'seldom': 2,
               'sometimes': 5,
               'usually': 6,
               'very frequently': 9,
               'very rarely': 1},
 'gender': {'men': 0, 'women': 1},
 'objective': {'assistance': 7,
               'benefits': 6,
               'better chances': 5,
               'equal chances': 2,
               'equality': 3,
               'fairness': 0,
               'favors': 4,
               'justice': 1}}

asiq2_dimshape = (8, 2, 12, 6)

asiq2_keywords_grid = list(itertools.product(*[asiq2_keywords[f] for f in asiq2_field_names]))

asiq2_keyword_map = [dict(zip(asiq2_field_names, k)) for k in asiq2_keywords_grid ]

asiq2_keywords_grid_idx= torch.Tensor(
            [tuple([asiq_keywords_indices[f][ktuple[i]] for i, f in enumerate(asiq2_field_names)]) for ktuple in
             asiq2_keywords_grid])

asiq2W = [tuple([asiq2_dimensions[f][ktuple[i]] for i, f in enumerate(asiq2_field_names)]) for ktuple in
             asiq2_keywords_grid]

asiq2_weights_grid = pd.DataFrame(asiq2W, columns=asiq2_field_names)

asiq2_weights_flat = asiq2_weights_grid.prod(axis=1)

asiq2_pdf = pd.DataFrame(asiq2_keywords_grid, columns=asiq2_field_names)
asiq2_pdf = asiq2_pdf.assign(P=0)
asiq2_pdf = asiq2_pdf.assign(W=asiq2_weights_flat)




class ASIQ2(QMNLI):
    def __init__(self, **kwargs):
        super().__init__(
            context_template="{fraction} {gender} are asking for gender equality.",
            answer_template="{frequency} to get {objective}",
            dimensions={"objective":dict_objective,
                        "gender":dict_short_gender_plural,
                        "frequency":frequency_weights,
                        "fraction":intensifiers_fraction,
            },
            descriptor = {"Questionnair":"ASI",
                          "Factor":"H",
                          "Ordinal":2,
                          "Original":'H 2. Many women are actually seeking special favors, such as hiring policies that favor them over men, under the guise of asking for "equality."'
            },             
            **kwargs)
q = QCACHE(ASIQ2(index=["objective"], scale='frequency'))



class TestQmnli1(unittest.TestCase):
    def test_regular_fields(self):
        self.assertEqual(q._dimensions,asiq2_dimensions)
        self.assertEqual(q._field_names, asiq2_field_names)
        self.assertEqual(q._scale, asiq2_scale)
        self.assertEqual(q._context_template, asiq2_context_template)
        self.assertEqual(q._answer_template, asiq2_answer_template)
        self.assertIsNone(q.result)
        self.assertIsNone(q._p)
        self.assertIsNone(q.model)

    def test_descriptor(self):
        self.assertDictEqual(q._descriptor,asiq2_descriptor)

    def test_keywords(self):
        self.assertDictEqual(q._keywords, asiq2_keywords)

    def test_keywords_indices(self):
        self.assertDictEqual(q._keywords_indices, asiq_keywords_indices)

    def test_dimshape(self):
        self.assertTupleEqual(q._dimshape, asiq2_dimshape)
    
    def test_keyword_grid(self):
        self.assertListEqual(q._keywords_grid,asiq2_keywords_grid)

    def test_keyword_map(self):
        self.assertListEqual(q._keywords_map,asiq2_keyword_map)

    def test_keywords_grid_idx(self):
        self.assertTrue(torch.equal(q._keywords_grid_idx,asiq2_keywords_grid_idx))

    def test_weights_grid(self):
        self.assertTrue(asiq2_weights_grid.equals(q._weights_grid))

    def test_pdf(self):
        self.assertTrue(asiq2_pdf.equals(q._pdf))
    
    
    
    

class TestQmnliFilterDataFrame(unittest.TestCase):
    def test_filter_data_frame(self):
        data = [['Avi', 5], ['Niv', 20], ['Ron', 35]]
        df = pd.DataFrame(data, columns=['Name', 'Age'])
        filter:FILTER = {"Age":[20,35]}
        df = qmnli._filter_data_frame(df, filter)
        data2 =[[False],[True],[True]]
        df2 = pd.DataFrame(data2).all(axis=1)
        self.assertTrue(df.equals(df2))
        
        
        

mnli = pipeline("zero-shot-classification",device=device, model="typeform/distilbert-base-uncased-mnli")
mnli.model_identifier = "typeform/distilbert-base-uncased-mnli"



example1_dims = {"objective":dict_objective,
                    "gender":dict_short_gender_plural,
                    "fraction":intensifiers_fraction}

class EXAMPLE1(QMNLI):
    def __init__(self, **kwargs):
        super().__init__(
        context_template="{fraction} {gender} are asking for gender equality.",
        answer_template="always to get {objective}",
        dimensions=example1_dims,
        descriptor = {"Questionnair":"ASI",
                      "Factor":"H",
                      "Ordinal":2,
                      "Original":'H 2. Many women are actually seeking special favors, such as hiring policies that favor them over men, under the                            guise of asking for "equality."'},             
        **kwargs)
        
example1 = QCACHE(EXAMPLE1(index=["objective"], scale='fraction'))
example1.run(mnli)






class TestQmnliSoftMax(unittest.TestCase):
    """
    Rami's comment:
    - We need to add test that checks if the values sums to 1 but it is always the same matrix that the functions returns,
    and not the matrix the torch's softmax return. (!!!)
    """
    def test_pdf_stays_the_same(self):
        """
        After using softmax method we expect the ogirinal self._pdf to not change.
        Softmax should create a copy without modifying the original field.
        """
        before_softmax = example1._pdf.copy()
        result = example1.softmax('objective')._pdf
        after_softmax = example1._pdf
        self.assertTrue(before_softmax.equals(after_softmax))

    def test_one_dim_change(self):
        """ 
        Check that the sum of probabilities along one dimension('objective') for the rest of the dimensions ('fraction' and 'gender') sums to 1.:
        """
        result = example1.softmax('objective')._pdf
        dims_copy = copy.deepcopy(example1_dims)
        dims_copy.pop('objective', None) # Delete 'objective' from the copy of dimensions
        other_dims=list(dims_copy.keys())
        after_groupby = result.groupby(by=other_dims).sum()
        p_column = after_groupby['P'].values
        for p_val in p_column:
            self.assertAlmostEqual(first=1,second=p_val,places=4)

    def test_two_dims_change(self):
        """
        Test case: check that softmax over multiple dimensions works iteratively.
        Input: softmax over 'gender' and 'objective' dimensions
        Expect: probabilities over all dimensions except 'objective' sums to 1.
        """
        dims = ['gender', 'objective']
        result = example1.softmax(dims)._pdf
        after_groupby = result.groupby(by=['fraction','gender']).sum()
        p_column = after_groupby['P'].values

        for p_val in p_column:
            self.assertAlmostEqual(first=1,second=p_val,places=4)

    def test_two_dims_change2(self):
        """
        Test case: check that softmax over multiple dimensions works iteratively.
        Input: softmax over 'gender' and 'objective' dimensions
        Expect: probabilities over one dimension ('fraction') is the size of 'gender' dimension.
        """
        dims = ['gender', 'objective']
        result = example1.softmax(dims)._pdf
        after_groupby = result.groupby(by=['fraction']).sum()
        p_column = after_groupby['P'].values
        for p_val in p_column:
            self.assertAlmostEqual(first=2,second=p_val,places=4)
            
            

class TestQmnli_pd_index_sort_key(unittest.TestCase):
    def test_pd_index_sort_key(self):
        """
        Test case: check that _pd_index_sort_key sorts data frame by axises('objective' and 'fraction') weights (intensifiers).
        Input: Probability dataframe
        Expect: Sorted dataframe by axis1(columns) and than sort by axis0(rows).
        """
        df=example1._pdf
        index=["objective"]
        scale='fraction'
        df = pd.pivot_table(df, values='P', index=index, columns=[scale], aggfunc=np.mean)
        df = df.sort_index(axis=1, key=example1._pd_index_sort_key)
        columns_names = df.columns.values.tolist()
        for i in range(len(columns_names)-1):
            self.assertGreaterEqual(intensifiers_fraction[columns_names[i+1]], intensifiers_fraction[columns_names[i]])

        df = df.sort_index(axis=0, key=example1._pd_index_sort_key)
        rows_names = list(df.index.values.tolist())


        for i in range(len(rows_names)-1):
            self.assertGreaterEqual(dict_objective[rows_names[i+1]], dict_objective[rows_names[i]])


# print(unittest.main(argv=[''], verbosity=2, exit=False))