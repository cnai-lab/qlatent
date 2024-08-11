import numpy as np
import pandas as pd
import seaborn as sns
import torch
import scipy
import time
import sklearn as sk
from pprint import pprint
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import pipeline
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer
from abc import ABC, abstractmethod
import copy
from overrides import overrides
from typing import * 
from numbers import Number
import itertools
from typeguard import check_type
from functools import partial
import pingouin as pg


SCALE = Dict[str,Number]
DIMENSIONS = Dict[str,SCALE]
FILTER = Dict[str,Collection[str]]
IDXSELECT = Tuple[Union[slice,List[int]]]


def dict_same_weight(w,ks):
  return dict(zip(ks,[w]*len(ks)))


def dict_pos_neg(pos, neg, w):
  return dict(dict_same_weight(1.0*w/len(pos),pos), **dict_same_weight(-1.0*w/len(neg),neg))


def _filter_tensor(t, slices):
    for i in range(len(slices)):
        t = t[slices[i]]
    return t


def _filter_data_frame(df, filter:FILTER):
    select = df.copy()
    for f in filter: #only select rows where df[f] in filter[f]
        select[f] = select[f].apply(lambda x: x in filter[f])
    select = select.all(axis=1)
    return select
    

def fixed_check_type(var, expected_type):
    try:
        check_type(var, expected_type)
        return True
    except:
        return False


# def check_type(var, expected_type):
#   return isinstance(var, expected_type)


def print_gradient(df):
    import seaborn as snsintensifier_names
    cm = sns.light_palette("green", as_cmap=True)
    s = df.style.background_gradient(cmap=cm, axis=None) 
    s = s.format(precision = 4)
#     s = s.set_precision(4)
    return s


def wrap_replace_callable(c, f, with_copy=False, *fargs, **fkwargs):
    def wrapper(*cargs, **ckwargs):
        rc = c(*cargs, **ckwargs)
        rf = f(rc, *fargs, **fkwargs)
        return rf
    if with_copy:
        q = copy.deepcopy(c)
    else:
        q = c
    q.__call__ = wrapper
    return q
        

class QABSTRACT(ABC):

    
    def __init__(self, dimensions:DIMENSIONS = {}, model: pipeline = None, p=None, index=None, scale='intensifier', descriptor = {}):
        
        self._dimensions = dimensions
        self._field_names = list(self._dimensions.keys())
        self._scale = scale
        self.result = None
        self._index = index if index is not None else list(set(self._field_names) - set([scale]))
        self._descriptor = descriptor
        self._descriptor['scale'] = str(self._scale)
        self._descriptor['index'] = str(self._index)

        self._keywords = {d:list(scale.keys()) for d,scale in dimensions.items()}
        #print(f"_keywords={self._keywords}")
        self._keywords_indices = {d:dict([(k,i) for i,k in enumerate(self._keywords[d])]) for d in self._keywords}
        #print(f"_keywords_indices={self._keywords_indices}")
        self._dimshape = tuple([len(self._keywords[d]) for d in self._field_names])
        #print(f"_dimshape={self._dimshape}")

        self._keywords_grid = list(itertools.product(*[self._keywords[f] for f in self._field_names]))
        #print(f"_keywords_grid={self._keywords_grid}")
        self._keywords_map = [dict(zip(self._field_names,k)) for k in self._keywords_grid]
        #print(f"_keywords_map={self._keywords_map}")
        self._keywords_grid_idx = torch.Tensor([tuple([self._keywords_indices[f][ktuple[i]] for i,f in enumerate(self._field_names)]) for ktuple in self._keywords_grid])
        #print(f"_keywords_grid_idx={self._keywords_grid_idx[:3]}")


        W = [tuple([self._dimensions[f][ktuple[i]] for i,f in enumerate(self._field_names)]) for ktuple in self._keywords_grid]
        #print(f"W={W}")
        self._weights_grid = pd.DataFrame(W, columns=self._field_names)
        #print(f"_weights_grid={self._weights_grid}")
        self._weights_flat = self._weights_grid.prod(axis=1)
        #print(f"_weights_flat={self._weights_flat}")
        self._weights_dense = torch.sparse_coo_tensor(self._keywords_grid_idx.T, self._weights_flat, self._dimshape).to_dense()
        #print(f"_weights={self._weights.shape}")


        self._pdf = pd.DataFrame(self._keywords_grid, columns=self._field_names)
        self._pdf = self._pdf.assign(P=0)
        self._pdf = self._pdf.assign(W=self._weights_flat)
        #print(f"_pdf={self._pdf}")
        

        self._p = p
        self.model = model
        

    @abstractmethod
    def run(self, model):
        if model:
            self.model = model
        


    def _grouping_suitable_for_consistency_check(self, grouping, filter={}):
        if len(grouping)<2:
            return False
        
        label_count = 0
        
        for group_filter in grouping:
            df = self._pdf[_filter_data_frame(self._pdf, filter)]
            df = df[_filter_data_frame(df,group_filter)]
            if len(df) > 0:
                label_count += 1
                
        return label_count >= 2


    def _filter_words_to_slice(self, filter:FILTER)->IDXSELECT:
        s = [ #loop over dimensions
                        slice(None,None) if d not in filter else
                        [ #loop over keywords in dimesion
                            self._keywords_indices[d][k]
                            for k in filter[d]
                        ]
                        for d in self._field_names
                      ]
        #one-hot econding, otherwise the shapes won't broadcast together...
        n = len(self._field_names)
        result = [[slice(None,None,None) for _ in range(n)] for _ in range(n)]
        for i in range(n):
            result[i][i]=s[i]
        return result


    def _create_default_grouping(self, group_field):
        default_grouping_df = pd.DataFrame(self._dimensions[group_field].items(), columns=['name', 'score'])
        grouping = [{group_field: group_df['name'].tolist()} for group_score, group_df in default_grouping_df.groupby('score')]
        # print(f'No grouping stated, using default grouping on {group_field}:')
        # pprint(grouping)
        return grouping
        
    
    def _pd_values_sort_key(self, terms:pd.Series)->pd.Series:
        f = terms.name
        result = terms.apply(lambda x: self._dimensions[f][x])
        return result


    def get_filter_for_postive_keywords(self, ignore_set=None):
        ignore_set = {self._scale} if ignore_set is None else ignore_set
        result:FILTER={}
        for dim in self._keywords:
            result[dim]=[]
            for k in self._keywords[dim]:
                if self._dimensions[dim][k] > 0 or dim in ignore_set:
                    result[dim].append(k)
        return result

    
    def __add__(self,other):
        result = copy.deepcopy(self)
        result._p += other._p
        result._p_raw += other._p_raw
        return result

    
    def __sub__(self,other):
        result = copy.deepcopy(self)
        result._p -= other._p
        result._p_raw -= other._p_raw
        return result

    
    def __mul__(self,other):
        result = copy.deepcopy(self)
        result._p *= other._p
        result._p_raw *= other._p_raw
        return result

    
    def __truediv__(self,other):
        result = copy.deepcopy(self)
        result._p /= other._p
        result._p_raw /= other._p_raw
        return result

    
    def __call__(self, model=None):
        return self.run(model)


    # # def __str__(self):
    # #   return f"<{self.__class__.__name__} object> {self._descriptor}"
    # def __repr__(self):
    #   return f"<{self.__class__.__name__} object> {self._descriptor}"

    
    def __hash__(self):
        """
        Questions having the same are descriptor should represent the same question.
        """
        return hash(frozenset(self._descriptor.items()))
    
    
    def __eq__(self,other):
        """
        Questions having the same are descriptor represent the same question.
        """
        return self._descriptor == other._descriptor

    
    def set_model(self, model):
        self.model = model
       
          
    def softmax(self, dim=0, temperature = 1):
        """
        params:
        dim: int or tuple. The dimensions to normalize with softmax. Dimesions will be normalized one by one in the given order.
        """
        if not fixed_check_type(dim, List):
            dim = [dim]
        if fixed_check_type(dim, List[str]):
            dim = [self._field_names.index(d) for d in dim]
        elif fixed_check_type(dim, List[int]):
            pass
        else:
            raise TypeError(dim)

        result = copy.deepcopy(self,memo={id(self.model):self.model})
        # del result.model
        # result.model = self.model
        coo = self._keywords_grid_idx.T
        p = torch.Tensor(self._pdf["P"])
        p_dense = torch.sparse_coo_tensor(coo, p, self._dimshape).to_dense()

        p_sparse = p_dense.to_sparse()
        p_vals = p_sparse.values()
        p_coos = p_sparse.indices()
        assert torch.all(torch.eq(p, p_vals))
        assert torch.all(torch.eq(coo, p_coos))
                
        for d in dim:
            p_log = torch.log(p_dense)
            p_log = p_log / temperature
            p_dense = torch.nn.functional.softmax(p_log, dim=d)
            

        p_vals = p_dense.to_sparse().values()

        result._pdf["P"] = p_vals
        return result    


    def minmax(self, dim=Union[str,int,List[str],List[int]]):
        """
        params:
        dim: int or tuple. The dimensions to normalize with softmax. Dimesions will be normalized one by one in the given order.
        """
        if not fixed_check_type(dim, List):
            dim = [dim]
        if fixed_check_type(dim, List[str]):
            dim = [self._field_names.index(d) for d in dim]
        elif fixed_check_type(dim, List[int]):
            pass
        else:
            raise TypeError(dim)

        result = copy.deepcopy(self,memo={id(self.model):self.model})
        coo = self._keywords_grid_idx.T
        p = torch.Tensor(self._pdf["P"])
        p_dense = torch.sparse_coo_tensor(coo, p, self._dimshape).to_dense()

        p_sparse = p_dense.to_sparse()
        p_vals = p_sparse.values()
        p_coos = p_sparse.indices()
        assert torch.all(torch.eq(p, p_vals))
        assert torch.all(torch.eq(coo, p_coos))

        print(p_dense.shape)
        for d in dim:
            print(d)
#             p_dense = torch.nn.functional.softmax(p_dense, dim=d)
            p_dense = p_dense / torch.max(p_dense, dim=d).to_sparse().repeat(*list(set(dim) - set([d])))
            print(p_dense.shape)

        p_vals = p_dense.to_sparse().values()

        result._pdf["P"] = torch.Tensor(p_vals)
        return result
        
    
    def to_dataframe(self, scale:Union[str,int], index:List[str], filter:FILTER={}, categories:Dict[str,Dict[str,str]]={}):
        # try:
        #     check_type(scale, int)
        #     scale = self._field_names[scale]
        # except TypeCheckError:
        #     check_type(scale, str) #redundant check for clarity
        
        if fixed_check_type(scale, int):
          scale = self._field_names[scale]
        elif fixed_check_type(scale, str):
            pass
           #redundant check for clarity
        else:
            raise TypeError(scale)


        select = _filter_data_frame(self._pdf, filter)
        df = self._pdf[select]
        print(f"index = {index}")
        print(set(self._field_names), set([scale]), set(index) )
        aggregated_fields = set(self._field_names) - set([scale]) - set(index)
        aggregated_fields = list(aggregated_fields)
        print(aggregated_fields)
        if len(aggregated_fields)>0:
            W = self._weights_grid[select]
            W = W[aggregated_fields]
            W = W.prod(axis=1)
            df["P"] = df["P"]*W


        for f in categories.items(): #replace individual keywords with categories
            df[f] = df[f].apply(lambda x: categories[f][x])

        #group by index categories
        df = pd.pivot_table(df, values='P', index=index, columns=[scale], aggfunc=np.mean)
        df = df.sort_index(axis=1, key=self._pd_index_sort_key)
        df = df.sort_index(axis=0, key=self._pd_index_sort_key)

        return df

        
    def _pd_index_sort_key(self, terms:pd.Index)->pd.Index:
        f = terms.name
        result = pd.Index([self._dimensions[f][x] for x in terms])
        return result


    def mean_score(self, filter:FILTER={}):
        select = _filter_data_frame(self._pdf,filter)
        P = self._pdf[select]
        score = P["P"]*P["W"]
        score = score.mean()
        return score


    def internal_consistency(self, measure="silhouette_score", metric="correlation", grouping:List[FILTER]=[], scale:Union[str,int] = None, index:List[str] = None, filter:FILTER={}):
        """
        Before the internal consistency check the data is diced according to the filter -- e.g. only entries matching the filter will be considered.
        This operation first splits the data into scale long vectors. The vectors are grouped according to the grouping argument.
        The vectors within each group are expected to be similar (high correlation, low distance) while vectors in different groups are expected to be different.
        This expectation is validated by Scikit's unsupervised cluster quality measures.

        Parameters:
        -----------
        @filter
            Specifies a set of keywords to keep along chosen dimensions. Unspecified dimensions are kept (not filtered out).
        @scale
            Dimension specifying the vector for which distances are computed. This is the dimension which is used for likert scale values.
        @grouping
            Specifies multiple dices (each one is structured like the filter) within which likert scale values are expected to be the same.
            Likert scale values are expected to be different for different dices. Dices are not required to have the same shape.
        @measure: string
            An unsupervised cluster quality measure from scikit. Must be one of 'calinski_harabasz_score', 'davies_bouldin_score', 'silhouette_score', 'silhouette_samples'
        @metric: string
            One of distance metrics from 'sklearn.metrics.pairwise_distances' used for computing silhouette
        """
#         try:
#           scale = self._scale if scale is None else scale
#           check_type(scale, int)
#           scale = self._field_names[scale]
#         except TypeCheckError:
#           check_type(scale, str) #redundant check for clarity

        scale = self._scale if scale is None else scale
        if fixed_check_type(scale, int):
            scale = self._field_names[scale]
        elif fixed_check_type(scale, str):
            pass
           #redundant check for clarity
#         else:
#             raise TypeError(scale)

        # Add default grouping for the index field
        index = self._index if index is None else index
        if grouping is not None and len(grouping) == 0:
            group_field = index[0]
            grouping = self._create_default_grouping(group_field)

        df = self._pdf[_filter_data_frame(self._pdf, filter)]

        X = []
        L = []
        for label,group_filter in enumerate(grouping):
            dice = df[_filter_data_frame(df, group_filter)]
            dice = dice.pivot_table(index=index,columns=scale, values="P", aggfunc='mean')
            X += [dice]
            L += [label]*len(dice)
        X = pd.concat(X, axis=0)

        f = getattr(sk.metrics, measure)
        if 'metric' in f.__code__.co_varnames[:f.__code__.co_argcount + f.__code__.co_kwonlyargcount]:
            result = f(X=X, labels=L, metric=metric)
        else:
            result = f(X=X, labels=L)

        return result

    
    def print_gradient(self, raw_probabilities=False):
        df = self.to_dataframe(self, raw_probabilities=raw_probabilities)
        cm = sns.light_palette("green", as_cmap=True)
        return df.style.background_gradient(cmap=cm, axis=None)

    
    def inner_alpha(self, filter:FILTER={}):
        df = self.to_dataframe(scale=self._scale, index=self._index, filter=filter)
        if bool(self._filter):
            indecies_list = self._filter[self._index[0]]
            dict_items = self._dimensions[self._index[0]].items()
            matches = [(key, value) for key, value in dict_items if key in indecies_list]
            default_grouping_df = pd.DataFrame(matches, columns=['name', 'score'])
        else:
            default_grouping_df = pd.DataFrame(self._dimensions[self._index[0]].items(), columns=['name', 'score'])
        grouping = [{self._index[0]: group_df['name'].tolist()} for group_score, group_df in default_grouping_df.groupby('score')]
        group_scores = pd.DataFrame(self._dimensions[self._index[0]].items(), columns=['name', 'score'])['score']
        for group in grouping:
            vals = group[self._index[0]]
            print(vals, 'Alpha:',pg.cronbach_alpha(data=df.T[vals])[0])
        if not bool(self._filter):
            a3 = pg.cronbach_alpha(pd.DataFrame((df.T.to_numpy() * group_scores.to_numpy())))
            print('Global alpha:', a3[0])
            return a3[0]
        return pg.cronbach_alpha(data=df.T[vals])[0]
        

    def report(self,scale:Union[str,int]=None, index:List[str]=None, filters:Dict[str,FILTER]={"unfiltered":{}}, grouping:List[FILTER]=[],):
        scale = self._scale if scale is None else scale
        if not fixed_check_type(scale, str):
            scale = self._field_names.index(scale)
        if index is None:
            if self._index:
                index = self._index
            else:
                index = list(set(self._field_names) - set([scale, "P", "W"]))

        print(f"Query time: {self._T}")
        for label, filter_dict in filters.items():
            print(f"Mean score {label} [{self._weights_flat.min()}..{self._weights_flat.max()}]: {self.mean_score(filter=filter_dict)}")

         # Add default grouping for the index field
        if grouping is not None and len(grouping) == 0:
            group_field = index[0]
            grouping = self._create_default_grouping(group_field)
        
        for label, filter_dict in filters.items():
            if self._grouping_suitable_for_consistency_check(grouping, filter=filter_dict):
                partial_internal_consistency = partial(self.internal_consistency, grouping=grouping,filter=filter_dict, index=index , scale=scale)
                print(f"Internal consistency (silhouette, correlation) for {label}: {partial_internal_consistency(measure='silhouette_score', metric='correlation')}")
                print(f"Internal consistency (Calinski&Harabasz)  for {label}: {partial_internal_consistency(measure='calinski_harabasz_score')}")
                print(f"Internal consistency (Davies&Bouldin) for {label}: {partial_internal_consistency(measure='davies_bouldin_score')}")
            else:
                print("At least two groups with at least two vectors in each group should be specified to check for internal consistency.")

        for label, filter_dict in filters.items():
            print("\n")
            display(
              print_gradient(
                  self.to_dataframe(
                    scale=scale,
                    index=index,
                    filter=filter_dict,
            )))
            self.inner_alpha(filter=filter_dict)
        


class QDELEGATOR(QABSTRACT):

    
    def __init__(self, srcobj):
        """
        Shallow copy the state of the source object.
        Both objects will share the same current state.
        The run method of QDELEGATOR delegates the execution to srcobj.run and
        copies the state after each execution. Any modifications made to the state
        of srcobj prior to execution of run(..) are copied as well.
        Further modifications to the shallow state of either one of the objects will
        not affect the other one.
        @srcobj
          initialized object of a question.
        """
        super().__init__()
        self.__dict__.update(srcobj.__dict__)
        self._run = srcobj.run
        
    
    @overrides
    def run(self, model=None, **kwargs):
        #print(f"{self.__class__.__name__} delgates execution of run(..) to {self._run.__self__.__class__.__name__}" )
        result = self._run(model, **kwargs)
        d = dict(result.__dict__)
        #retain self._descriptor and self._run
        if "_descriptor" in d: del d['_descriptor']
        if "_run" in d: del d['_run']
        self.__dict__.update(d)
        # #restore self._run to maintain full delegation path
        # self._run = result.run
        return self


class QCACHE(QDELEGATOR):

    
    def __init__(self, srcobj, **kwargs):
        super().__init__(srcobj, **kwargs)
        #subclasses instantiated with questions that were already run
        #may set the cached flag to False to force rerunning the question
        self._cached = False
        self._last_model_identifier = None

    
    @overrides
    def run(self,model=None, **kwargs):
        # print(f"Execute cachable question with model {model}.")
        # print(f"Cached: {self._cached}, same model:{model.model_identifier == self._last_model_identifier}.")
        if self._cached and model.model_identifier == self._last_model_identifier:
            # print(f"Skipping model execution. Use cached results.")
            return self
        result = super().run(model=model,**kwargs)
        self._cached = True
        self._last_model_identifier = model.model_identifier
        assert result==self
        return self



class QPASS(QDELEGATOR):

    
    def __init__(self, qobject, descupdate={}):
        super().__init__(qobject)
        #copy descriptor from srcobj to avoid propagation of changes
        self._descriptor = dict(self._descriptor, **descupdate)


class QSOFTMAX(QPASS):

    
    def __init__(self, qobject, dim="intensifier", temperature=1):
        super().__init__(qobject, {'softmax':str(dim)})
        self._dim = dim
        self._temperature = temperature

    
    @overrides
    def run(self,model=None, **kwargs):
        result = super().run(model, **kwargs)
        return result.softmax(dim=self._dim, temperature=self._temperature)


class QMINMAX(QPASS):

    
    def __init__(self, qobject, dim="intensifier"):
        super().__init__(qobject, {'minmax':str(dim)})
        self._dim = dim

    
    @overrides
    def run(self,model=None, **kwargs):
        result = super().run(model, **kwargs)
        return result.minmax(dim=self._dim)


class QFILTER(QPASS):

    
    def __init__(self, qobject, filter:FILTER = {}, filtername=""):
        super().__init__(qobject, {'filter':filtername})
        self._filter = filter

    
    @overrides
    def run(self,model=None, **kwargs):
        result = super().run(model, **kwargs)
        assert result == self
        select = _filter_data_frame(self._pdf, self._filter)
        result._pdf = self._pdf[select]
        result._weights_grid = self._weights_grid[select]
        return result
