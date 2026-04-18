from qlatent.qmnli.qmnli import *
from qlatent.qmlm.qmlm import *
from tqdm import tqdm
from qlatent.hf_model_evaluator.global_variables import *
from qlatent.hf_model_evaluator.utils.logging_utils import LoggingUtils
logging_utils = LoggingUtils()


from qlatent.questionnaire_eval.questionnaire_utils import *



# def split_question(Q, index, scales, softmax, filters):
#   result = []
#   for s in scales:
#     q = QCACHE(Q(index=index, scale=s))
# #     q.softmax_filter = softmax
# #     q.filters = filters
#     for sf in softmax:
#       for f in filters:
#         if sf:
#             qsf = QSOFTMAX(q,dim=[s])
#             qsf_f = QFILTER(qsf,filters[f],filtername=f)
#             #print(s,sf,f)
#             result.append(qsf_f)

#             qsf = QSOFTMAX(q,dim=[index[0]])
#             qsf_f = QFILTER(qsf,filters[f],filtername=f)
#             #print(index,sf,f)
#             result.append(qsf_f)

#             qsf = QSOFTMAX(q,dim=[index[0], s])
#             qsf_f = QFILTER(qsf,filters[f],filtername=f)
#             #print((index, s),sf,f)
#             result.append(qsf_f)
#         else:
#             qsf = QPASS(q,descupdate={'softmax':''})
#             qsf_f = QFILTER(qsf,filters[f],filtername=f)
#             #print(s,sf,f)
#             result.append(qsf_f)
#   return result


# class Questionnaire:
    
#     def __init__(self, name, questions=None):
#         self.name=name
#         self.questions = questions if questions else []
    
#     def add_question(self, question):
#         self.questions.append(question)
        
        
#     def __len__(self):
#         return len(self.questions)
    
#     def evaluate_questions(self, model):
#         questions_mean_score_dict = {}
#         print(f"\tEvaluating {self.name} questionnaire on {model.model_identifier}: ", flush=True)
#         for Q in tqdm(self.questions):
#             Qs = split_question(Q,
#                               index=Q.index,
#                               scales=[Q.scale],
#                               softmax=[False, True],
#                               filters={'unfiltered':{},
#                                       "positiveonly":Q().get_filter_for_postive_keywords()
#                                       },
#                               )
#             question_obj = Qs[4]
#             mean_score = question_obj.run(model).mean_score()
#             questions_mean_score_dict[question_obj] = mean_score
#             #print(f"Question {questions_obj._descriptor['Ordinal']}: {mean_score}")
        
#         return questions_mean_score_dict
    

class Questionnaires:
    def __init__(self, questionnaires_questions_lists):
        self.questionnaires_questions_lists=questionnaires_questions_lists
        if self.questionnaires_questions_lists:
            self.prepare_questionnaires()
        
        
    def get_questionnaire_num_questions(self):
        questionnaire_num_questions = {}
        #for task_type in self.questionnaires:
        for questionnaire in self.questionnaires[NLI_TYPE]:
            questionnaire_num_questions[questionnaire.name] = len(questionnaire) # uses custom __len__ method of questionaire class
        return questionnaire_num_questions
        
    def prepare_questionnaires(self):
        """
            This method prepares the questionnaires using the Questionnaire object.
            Each list in `questionnaire_questions` consists classes objects of a questionnaire.
            All questionnaires are divided into MLM and NLI type for future evalutaion.
        """
        
        questionnaires : Dict[Literal["QMLM", "QMNLI"], list] = {
            MLM_TYPE: [],
            NLI_TYPE: [],
        }
        for questionnaire_questions in self.questionnaires_questions_lists:
            questionnaire_name : str = questionnaire_questions[0]()._descriptor["Questionnair"]
            if issubclass(questionnaire_questions[0], QMLM):
                questionnaire_type: Literal["QMLM", "QMNLI"] = MLM_TYPE
            elif issubclass(questionnaire_questions[0], QMNLI):
                questionnaire_type: Literal["QMLM", "QMNLI"] = NLI_TYPE
            else:
                logging_utils.log_error(f"Unknown questionnaire type for {questionnaire_name}")
                continue

            questionnaire : Questionnaire = Questionnaire.create_questionnaire_from_questions(questionnaire_questions)
            questionnaire.questionnaire_type = questionnaire_type
            questionnaires[questionnaire_type].append(questionnaire)          
            

        self.questionnaires = questionnaires