import torch
from matplotlib import *
import matplotlib.image as im
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pathlib import Path
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from abc import *
from typing import *
from numbers import Number
from sentence_transformers import SentenceTransformer, util
from transformers import pipelines
from qlatent.qmnli.qmnli import *
from qlatent.qmnli.qmnli import QMNLI, _filter_data_frame

device = 0 if torch.cuda.is_available() else -1
print(device)

CONSTRUCT_ICON_PATH = "../../../data/graphs_add_ons/icons/c_icon.png"
ANTICONSTRUCT_ICON_PATH = "../../../data/graphs_add_ons/icons/ac_icon.png"
CONSTRUCT_DA_ICON_PATH = "../../../data/graphs_add_ons/icons/c_da_icon.png"
ANTICONSTRUCT_DA_ICON_PATH = "../../../data/graphs_add_ons/icons/ac_da_icon.png"

def print_permutations(q):
    W = q._pdf['W']
    print(q._descriptor)
    for i, (kmap, w) in enumerate(zip(q._pdf.drop(columns=['P', 'W']).to_dict(orient='records'), W)):
        context = q._context_template.format_map(kmap)
        answer = q._answer_template.format_map(kmap)
        print(f'{i}.',context ,'->', answer, w)

def run(self,model=None, **kwargs):
    result = run(model, **kwargs)
    assert result == self
    select = _filter_data_frame(self._pdf, self._filter)
    result._pdf = self._pdf[select]
    result._weights_grid = self._weights_grid[select]
    return result

def split_question(Q, index, scales, softmax, filters):
  result = []
  for s in scales:
    q = QCACHE(Q(index=index, scale=s))
    for sf in softmax:
      for f in filters:
        if sf:            
            qsf = QSOFTMAX(q,dim=[index[0], s])
            qsf_f = QFILTER(qsf,filters[f],filtername=f)
            print((index, s),sf,f)
            result.append(qsf_f)
        else:
            qsf = QPASS(q,descupdate={'softmax':''})
            qsf_f = QFILTER(qsf,filters[f],filtername=f)
            print(s,sf,f)
            result.append(qsf_f)
  return result

class FORMAT_MAKER:
    """
        A class that creates the 3 formats to translate the hebrew ASI questionare
        - q_stripped : The question with the template only.
        - q_regular  : The question with template and context.
        - q_flipped  : The question with template and anti-context.

    """
    
    def __init__(self, construct, anti_construct, questionnaire, factor, question_number, original_question, print_permutations, pipeline):
        self.construct = construct
        self.anti_construct = anti_construct
        self.questionnaire = questionnaire
        self.factor = factor
        self.question_number = question_number
        self.original_question = original_question
        self.print_permutations = print_permutations
        self.pipeline = pipeline

        self.q_stripped = self.split_question_s(construct = "", flipped = False, questionnaire = self.questionnaire, factor = self.factor, question_number = self.question_number, original_question = self.original_question)

        self.q_regular = self.split_question_s(construct = self.construct, flipped = False, questionnaire = self.questionnaire, factor = self.factor, question_number = self.question_number, original_question = self.original_question)

        self.q_flipped = self.split_question_s(construct = self.anti_construct, flipped = True, questionnaire = self.questionnaire, factor = self.factor, question_number = self.question_number, original_question = self.original_question)
        
    def split_question_s(self, construct, flipped, questionnaire, factor, question_number, original_question):
        return split_question(
            lambda **kwargs: self.QUESTION(construct, flipped, questionnaire, factor, question_number, original_question, **kwargs),
            index=['נכונות'],
            scales=['תדירות'],
            softmax=[True, False],
            filters={'unfiltered': {},
                     "positiveonly": self.QUESTION(construct, flipped, questionnaire, factor, question_number,
                                                               original_question).get_filter_for_postive_keywords()
                     },
            )[2]
    
    kw_pos = ['מוצדקת', 'נכונה']
    kw_neg = ['שגויה', 'מופרכת']

    dict_objective = dict_pos_neg(kw_pos, kw_neg, 1)
    dict_objective_flipped = dict_pos_neg(kw_neg, kw_pos, 1)
    
    intensifier_dict_1 ={
        'לרוב':1,
        'בדרך כלל':1,
        'לעיתים רחוקות':-1,
        'מדי פעם':-1
    }
    
    class QUESTION(QMNLI):
        def __init__(self, construct, flipped, questionnaire, factor, question_number, original_question, **kwargs):
            super().__init__(
                context_template=f"התבקשתי לומר מה דעתי על הטענה '{construct}'.",
                answer_template="עניתי שלדעתי הטענה {תדירות} {נכונות}.",
                dimensions={"נכונות": FORMAT_MAKER.dict_objective_flipped if flipped else FORMAT_MAKER.dict_objective,
                            "תדירות": FORMAT_MAKER.intensifier_dict_1,
                            },
                descriptor={"Questionnaire": questionnaire,
                            "Factor": factor,
                            "Ordinal": question_number,
                            "Original": original_question
                            },
                **kwargs
            )

    
    def run_questions(self):
        print("""\n
        ------------------------------------
        NO CONTEXT QUESTION SILHOUETTE TABLE: 
        ------------------------------------
        """)
        self.q_stripped.run(self.pipeline).report()
        if(self.print_permutations):
            print_permutations(self.q_stripped)
        print("""\n
        ---------------------------------
        REGULAR QUESTION SILHOUETTE TABLE: 
        ---------------------------------
        """)
        self.q_regular.run(self.pipeline).report()
        if(self.print_permutations):
            print_permutations(self.q_regular)        
        print("""\n
        ------------------------------------------
        REVERSED CONTEXT QUESTION SILHOUETTE TABLE: 
        ------------------------------------------
        """)
        self.q_flipped.run(self.pipeline).report()
        if(self.print_permutations):
            print_permutations(self.q_flipped)        
        
        print(f"""\n\n
        Mean score of Template (T) : {self.q_stripped.mean_score()}
        Mean score of Template with construct (TC) : {self.q_regular.mean_score()}
        Mean score of Template with anti construct (TAC) : {self.q_flipped.mean_score()}
        
        Raw difference between template with construct and template without construct (TC - T): {self.q_regular.mean_score() - self.q_stripped.mean_score()}
        Raw difference between template with antonym of construct and template without construct (TAC - T): {self.q_flipped.mean_score() - self.q_stripped.mean_score()}
        
        Normalized difference between template with construct and template without construct (TC - T) / T: {(self.q_regular.mean_score() - self.q_stripped.mean_score())/self.q_stripped.mean_score()}
        Normalized difference between template with antonym of construct and template without construct (TAC - T) / T: {(self.q_flipped.mean_score() - self.q_stripped.mean_score())/self.q_stripped.mean_score()}
        Raw difference between template with construct and template with antonym of construct (TC - TAC): {self.q_regular.mean_score() - self.q_flipped.mean_score()}
        """)
        return (self.q_regular.mean_score() - self.q_stripped.mean_score()) / self.q_stripped.mean_score(), (self.q_flipped.mean_score() - self.q_stripped.mean_score()) / self.q_stripped.mean_score()
    
def visualize_constructs_diffs(c_first_type_scores : list[float], c_second_type_scores : list[float], question_nums : list[int], mode: str):

        c_first = None
        c_second = None
        title = None
    
        if mode == "c_ac":
            c_first = CONSTRUCT_ICON_PATH
            c_second = ANTICONSTRUCT_ICON_PATH
            title = """Comparison between Mean Scores of MRS NLI
Queries with Construct and Queries with Anti-Construct Before DA by Item"""
        elif mode == "c_da":
            c_first = CONSTRUCT_ICON_PATH
            c_second = CONSTRUCT_DA_ICON_PATH
            title = """Comparison between Mean Scores of MRS NLI
Queries with Construct Before and After DA by Item"""
        elif mode == "ac_da":
            c_first = ANTICONSTRUCT_ICON_PATH
            c_second = ANTICONSTRUCT_DA_ICON_PATH
            title = """Comparison between Mean Scores of MRS NLI
Queries with AntiConstruct Before and After DA by Item"""
        elif mode == "c_ac_da":
            c_first = CONSTRUCT_DA_ICON_PATH
            c_second = ANTICONSTRUCT_DA_ICON_PATH
            title = """Comparison between Mean Scores of MRS NLI
Queries with Construct and Queries with Anti-Construct After DA by Item"""
        
        c_first_logo = im.imread(c_first)
        c_second_logo = im.imread(c_second)
        
        c_first_imagebox = OffsetImage(c_first_logo, zoom = 0.15)
        c_second_imagebox = OffsetImage(c_second_logo, zoom = 0.15)
        
        fig, ax = plt.subplots()
        
        for question_num_index in range(len(question_nums)):
            
            c_first_score = c_first_type_scores[question_num_index]
            
            c_first = AnnotationBbox(c_first_imagebox, (question_nums[question_num_index], c_first_score), frameon = False)
            ax.add_artist(c_first)
            plt.scatter(question_nums[question_num_index], c_first_score, s = 0.15)
            
            c_second_score = c_second_type_scores[question_num_index]
            
            c_second = AnnotationBbox(c_second_imagebox, (question_nums[question_num_index], c_second_score), frameon = False)
            ax.add_artist(c_second)
            plt.scatter(question_nums[question_num_index], c_second_score, s = 0.15)
        
        combined_scores = c_first_type_scores + c_second_type_scores
        min_score = min(combined_scores)
        max_score = max(combined_scores)
        
        for question_num in question_nums:    
            if question_num > 1:
                    plt.plot([question_num - 0.5] * 2, [min_score - 0.15, max_score + 0.15], color="grey", linestyle="dashed")
        
        plt.plot([question_nums[0] - 0.5, question_nums[-1] + 0.5], [0] * 2, color="black")
        plt.title(title, loc="left")
        plt.xlabel("Question Number")
        plt.ylabel("Difference from Template Alone (in Mean Score)")
        
        plt.show()