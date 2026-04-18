import torch
import gc
import numpy as np
import pandas as pd
from matplotlib import *
import matplotlib.image as im
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Patch
from pathlib import Path
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from abc import *
from typing import *
from numbers import Number
from sentence_transformers import SentenceTransformer, util
from transformers import pipelines, AutoModel, Pipeline
import pingouin as pg
from itertools import chain
from abc import ABC, abstractmethod
from jinja2 import Template
import markdown
from datetime import datetime
import pytz
from tqdm import tqdm
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import csv
# from qlatent.qabstract.qabstract import *
# change for the path of qlatent files
from qlatent.qabstract.qabstract import *
from huggingface_hub import HfApi
hf_api = HfApi()


device = 0 if torch.cuda.is_available() else -1
print(device)

FILTER = Dict[str,Collection[str]]


class Questionnaire:
    """
       Class representing a Questionnaire.
       The class has the following parameters:
           name (str): The name of the questionnaire.
           num_of_questions (int): the number of the total questions of the questionnaire.
           factors (List(str)): the factors of the questionnaire.
           factor_grouping (Dict[str -> set(str)]): Used for grouping sub factor into one factor.
           questions (dict[int -> Qabstract]): the questions of the questionnaire are stored here, mapped via their ordinal number.
    """

    def __init__(self, name : str, num_of_questions : int, factors : List[str], factor_grouping:Dict[str,Set[str]]=None, full_name:str=None):
        # input validation
        if not isinstance(name, str):
            raise TypeError(f"Questionnaire name must be a string, got type {type(name)} instead.")
        if full_name is None:
            full_name = name
        elif not isinstance(full_name, str):
            raise TypeError(f"Questionnaire full name must be a string, got type {type(name)} instead.")
        if not isinstance(num_of_questions, int):
            raise TypeError(f"Questionnaire num_of_questions must be an int, got type {type(num_of_questions)} instead.")
            if (num_of_questions < 1):
                raise ValueError(f"Questionnaire num_of_questions must be larger than one (1), got {num_of_questions} instead.")
        if not (isinstance(factors, list) and all(isinstance(factor, str) for factor in factors)):
            raise TypeError("Questionnaire factors must be a list of strings.")
        
        self.name = name
        self.full_name = full_name
        self.num_of_questions = num_of_questions
        self.factors = {factor : set() for factor in factors}
        self.set_factor_grouping(factor_grouping)
        self.questions = {}
        
        
    def set_factor_grouping(self, factor_grouping:Dict[str,Set[str]]=None):
        if factor_grouping is None:
            self._factor_grouping = {factor : {factor,} for factor in self.factors}
        elif not (isinstance(factor_grouping, dict) and all(isinstance(k, str) and isinstance(v, set) and all(isinstance(i, str) and i in self.factors for i in v)  for k, v in factor_grouping.items())):
            raise TypeError("Questionnaire factor_split must be a dictionary mapping string (factor grouping) to set of strings (current factors).")
        else: # user provided valid factor_grouping
            self._factor_grouping = factor_grouping
            self._update_factor_grouping()
            
                
    def _update_factor_grouping(self):
        # make sure the factor grouping addresses all of the current factors in the questionnaire
        factors_covered_in_grouping = set()
        for factors in self._factor_grouping.values():
            factors_covered_in_grouping.update(factors)

        for factor in self.factors.keys(): # add missing factors to factor_grouping
            if factor not in factors_covered_in_grouping: # factor not represented in factor grouping
                    self._factor_grouping[factor] = {factor,}
        
        for factors in self._factor_grouping.values(): # remove unnecessary factors from _factor_grouping
            factors.intersection_update(self.factors.keys())
            
        # Delete factor groups that hold empty sets
        self._factor_grouping = {group_name: factors for group_name, factors in self._factor_grouping.items() if factors}
                        
        
    @classmethod
    def create_questionnaire_from_questions(cls, questions : List[QABSTRACT]):
        if not isinstance(questions, list):
            raise ValueError("questions parameter must must be of type list.")
        if len(questions) == 0:
            raise ValueError("questions parameter must contain at least one question.")
            
        # create questions instances
        questions = [q() for q in questions]
                
        # Sample questionnaire name from the first question
        questionnaire_name = questions[0]._descriptor["Questionnair"]
        
        obj = cls(name = questionnaire_name,
                   num_of_questions = len(questions),
                   factors = [],
                   factor_grouping=None,
                   full_name=None)
        
        for q in questions:
             obj.add_question(q)
        
        return obj
    
     
    def is_complete(self) -> bool:
        return len(self.questions) == self.num_of_questions

    
    def __getitem__(self, index):
        return self.questions[index]
        

    def __len__(self):
        return len(self.questions)
        
    
    def add_question(self, q: QABSTRACT):
        """ adds a question object into the Questionnaire, and updates the factors dictionary accordingly """
        question_factor = q._descriptor['Factor']
        question_ordinal = q._descriptor['Ordinal']
        self.questions[question_ordinal] = q
        if len(self) > self.num_of_questions:
            self.num_of_questions = len(self)
        if question_factor in self.factors:
            self.factors[question_factor].add(question_ordinal)
        else:
            self.factors[question_factor] = {question_ordinal,}
            self._update_factor_grouping()
            #print(f"Added a new factor <{question_factor}> to the questionnaire {self.name} based on question with ordinal <{question_ordinal}>")
            
            


    def remove_question_by_ordinal(self, question_ordinal:int):
        """ Removes a question object from the questionnaire """
        if question_ordinal in self.questions:
            q = self.questions[question_ordinal]
            question_factor = q._descriptor['Factor']
            del self.questions[question_ordinal]
            self.factors[question_factor].remove(question_ordinal)
            self._update_factor_grouping()
        else:
            raise KeyError(f"Questionnaire does not contain a question with ordinal number <{question_ordinal}>")
            
            
    def remove_question_by_object(self, q:QABSTRACT):
        """ Removes a question object from the questionnaire """
        
        # find the object in self.questions
        for ordinal, obj in self.questions.items():
            if q is obj: # exact match for the same object
                del self.questions[ordinal]
                factor = q._descriptor['Factor']
                self.factors[factor].remove(ordinal)
                self._update_factor_grouping()
                return
        
        raise ValueError(f"Questionnaire does not contain the given question object <{q}>")
        
    
    def run(self,
            pipelines : List[Pipeline],
            questions_ordinals:List[int]=None,
            result_path=Path('./results/result.csv'),
            softmax:List[str]=[],
            filters:Dict[str,dict]={},
            merge_filtered_positiveonly=False,
            local_model = False,
            ):
        """ Runs given models on specified questions with pipeline template defined inside pipeline_builder
            
            pipelines : List[Pipeline] A list of HuggingFace pipelines to evaluate on a questionnaire.
            questions_ordinals:List[int] A list of integers representing the questions to run the models on, if left empty than the models will run over all the questions of the questionnaire.
            result_path The full path to save the result csv to, defualt location is './results/result.csv'
            softmax:List[str] List of strings representing which dimentions to apply softmax on, order of item matters.
            filters:Dict[str,dict] filters to apply to a questionnaire, each filter will result in additional logging for every question.
        """
        if questions_ordinals is None:
            questions_ordinals = list(self.questions.keys())
        for pipeline in pipelines:
            self.run_pipeline_on_questions(pipe=pipeline,
                                        questions_ordinals=questions_ordinals,
                                        softmax=softmax,
                                        filters=filters,
                                        result_path=result_path,
                                        merge_filtered_positiveonly=merge_filtered_positiveonly,
                                        local_model=local_model,
                                        )


        # self.calc_correlations()
        # self.report()
        
        
            
    def run_pipeline_on_questions(self, questions_ordinals, pipe, result_path, softmax, filters: Dict[str, dict], merge_filtered_positiveonly, local_model):
        print(f"\tEvaluating {self.name} questionnaire on {pipe.model_identifier}: ", flush=True)
        if not local_model:
            commits = hf_api.list_repo_commits(repo_id=pipe.model_identifier)
            last_commit = commits[0]
            last_commit_hash = last_commit.commit_id
            last_commit_date = last_commit.created_at.strftime("%d-%m-%Y %H:%M:%S") + " UTC"
        else:
            last_commit_hash = "unknown-local-model"
            last_commit_date = "unknown-local-model"
        
        for ordinal in tqdm(questions_ordinals):
            q = self.questions[ordinal]
            q = QCACHE(q)
            
            # Apply the softmax logic
            if (softmax is None) or (isinstance(softmax, list) and len(softmax) == 0):
                qsf = QPASS(q, descupdate={'softmax': ''})
            else:
                qsf = QSOFTMAX(q, dim=softmax)
            
            # Apply the filter logic
            if (filters is None) or (isinstance(filters, dict) and len(filters) == 0):
                filters = {'unfiltered': lambda q: {}}
            for filtername, filter_function in filters.items():                
                qsf_f = QFILTER(qsf, filter=filter_function(qsf), filtername=filtername)
                # Run the pipeline on the question
                qsf_f.run(pipe)
                # Retrieve attributes after the run method has populated the necessary data
                attributes = self.question_attributes(qsf_f, last_commit_hash, last_commit_date)
                attributes = list(attributes.items())
                attributes=dict(attributes)
                # Create and save attributes as a DataFrame
                attributes_df = pd.DataFrame([attributes])  # Wrap in list

                # Convert to list of items
                self.append_df_to_csv(attributes_df, result_path)
        if merge_filtered_positiveonly:
            self.merge_unfiltered_into_positive_only(result_path)

    def question_attributes(self, q, last_commit_hash, last_commit_date):
        descriptor = q._descriptor
        now = datetime.now(pytz.UTC)
        date_of_logging = now.strftime('%d-%m-%Y %H:%M:%S %Z')
        model_id = q.model.model_identifier if q.model.model_identifier else "no model detected"
        silhouette_table_structure = self.classify_heatmap_pattern(q._pdf, frequency_weights=q._dimensions["frequency"], index_weights=q._dimensions["index"])
        attributes = {
            "questionnaire" : descriptor['Questionnair'],
            "factor" : descriptor['Factor'],
            "ordinal" : descriptor['Ordinal'],
            'model' : model_id,
            "original" : descriptor['Original'],
            "translation_method" : None, # This value is set inside the inhereting class
            'query' : descriptor['query'],
            'scale' : q._scale,
            'index' : q._index,
            'dimensions' : q._dimensions,
            'filter' : descriptor['filter'],
            'softmax' : descriptor['softmax'],            
            'mean_score' : q.mean_score(),
            'silhouette_score' : Questionnaire.silhouette_score(q),
            'silhouette_table_structure' : silhouette_table_structure,
            "last_commit_hash" : last_commit_hash,
            "commit_date" : last_commit_date,
            "model_version_id" : model_id+"_"+last_commit_hash,
            "logging_date" : date_of_logging,
        }        
        return attributes
    
    
    @staticmethod
    def silhouette_score(q):
        try:
            silhouette_score= q.internal_consistency(
                measure='silhouette_score',
                metric='correlation',
                index=q._index,
                scale=q._scale,
            )
            return silhouette_score
        except Exception as e:
            print(f"Failed to calculate silhouette score for question with ordinal {q._descriptor['Ordinal']} with filter {q._descriptor['filter']}, Error: {e}")
            return None



    def merge_unfiltered_into_positive_only(self, df_path: str) -> pd.DataFrame:
        
        df = pd.read_csv(df_path, encoding='utf-8-sig')
        
        # Split data
        positive_only = df[df["filter"] == "positive_only"].copy()
        unfiltered = df[df["filter"] == "unfiltered"]

        # Merge on identifiers
        merge_keys = ["model_version_id", "ordinal"]

        merged = pd.merge(
            positive_only,
            unfiltered[merge_keys + ["silhouette_score"]],
            on=merge_keys,
            how="left",
            suffixes=("", "_from_unfiltered")
        )
        
        # Fill silhouette_score
        merged["silhouette_score"] = merged["silhouette_score"].fillna(merged["silhouette_score_from_unfiltered"])

        # Drop the temporary column
        merged = merged.drop(columns=["silhouette_score_from_unfiltered"])
        merged.to_csv(df_path, encoding="utf-8-sig", index=False)

        return merged


    def classify_heatmap_pattern(
        self,
        df: pd.DataFrame,
        score_col: str = 'P',
        frequency_weights: dict = None,
        index_weights: dict = None,
        dominance_ratio: float = 1.4,
        support_threshold: float = 0.3
    ) -> str:
        """
        Classifies a frequency × index heatmap using:
        - Original sorted order (ascending weights)
        - Semantic quadrant splits (based on polarity of weights)

        Returns:
            str: One of the pattern types:
                'diagonal', 'anti-diagonal', 'horizontal-dominant',
                'vertical-dominant', 'high_uniform', 'low_uniform', 'quarter-mixed'
        """
        matrix = df.pivot(index='index', columns='frequency', values=score_col)

        # ✅ Original sorting: ascending weight order
        if frequency_weights:
            freq_order = [k for k, _ in sorted(frequency_weights.items(), key=lambda x: x[1])]
            matrix = matrix[[f for f in freq_order if f in matrix.columns]]

        if index_weights:
            idx_order = [k for k, _ in sorted(index_weights.items(), key=lambda x: x[1])]
            matrix = matrix.loc[[i for i in idx_order if i in matrix.index]]

        matrix = matrix.dropna(axis=0, how='all').dropna(axis=1, how='all')
        vals = matrix.values
        total_mean = np.mean(vals)
        total_sum = np.sum(vals)

        # ✅ Split rows/cols by sign of weights
        neg_rows = [i for i in matrix.index if index_weights.get(i, 0) < 0]
        pos_rows = [i for i in matrix.index if index_weights.get(i, 0) >= 0]
        neg_cols = [f for f in matrix.columns if frequency_weights.get(f, 0) < 0]
        pos_cols = [f for f in matrix.columns if frequency_weights.get(f, 0) >= 0]

        # Quadrants (if any empty, use empty array)
        q1 = matrix.loc[neg_rows, neg_cols].values if neg_rows and neg_cols else np.array([])
        q2 = matrix.loc[neg_rows, pos_cols].values if neg_rows and pos_cols else np.array([])
        q3 = matrix.loc[pos_rows, neg_cols].values if pos_rows and neg_cols else np.array([])
        q4 = matrix.loc[pos_rows, pos_cols].values if pos_rows and pos_cols else np.array([])

        q1_sum, q2_sum, q3_sum, q4_sum = map(np.sum, [q1, q2, q3, q4])
        main_diag_sum = q1_sum + q4_sum
        anti_diag_sum = q2_sum + q3_sum

        q1_vals, q2_vals, q3_vals, q4_vals = map(lambda q: q.flatten() if q.size else np.array([]), [q1, q2, q3, q4])
        main_diag_vals = np.concatenate([q1_vals, q4_vals])
        anti_diag_vals = np.concatenate([q2_vals, q3_vals])
        main_diag_support = np.mean(main_diag_vals > total_mean) if main_diag_vals.size else 0
        anti_diag_support = np.mean(anti_diag_vals > total_mean) if anti_diag_vals.size else 0

        if anti_diag_sum and main_diag_sum / anti_diag_sum > dominance_ratio and main_diag_support >= support_threshold:
            return 'diagonal'
        if main_diag_sum and anti_diag_sum / main_diag_sum > dominance_ratio and anti_diag_support >= support_threshold:
            return 'anti-diagonal'

        top_vals = np.concatenate([q1_vals, q2_vals])
        bottom_vals = np.concatenate([q3_vals, q4_vals])
        top_sum, bottom_sum = q1_sum + q2_sum, q3_sum + q4_sum
        top_support = np.mean(top_vals > total_mean) if top_vals.size else 0
        bottom_support = np.mean(bottom_vals > total_mean) if bottom_vals.size else 0

        if bottom_sum and top_sum / bottom_sum > dominance_ratio and top_support >= support_threshold:
            return 'horizontal-dominant'
        if top_sum and bottom_sum / top_sum > dominance_ratio and bottom_support >= support_threshold:
            return 'horizontal-dominant'

        left_vals = np.concatenate([q1_vals, q3_vals])
        right_vals = np.concatenate([q2_vals, q4_vals])
        left_sum, right_sum = q1_sum + q3_sum, q2_sum + q4_sum
        left_support = np.mean(left_vals > total_mean) if left_vals.size else 0
        right_support = np.mean(right_vals > total_mean) if right_vals.size else 0

        if right_sum and left_sum / right_sum > dominance_ratio and left_support >= support_threshold:
            return 'vertical-dominant'
        if left_sum and right_sum / left_sum > dominance_ratio and right_support >= support_threshold:
            return 'vertical-dominant'

        quadrant_sums = [q1_sum, q2_sum, q3_sum, q4_sum]
        q_sum_range = max(quadrant_sums) - min(quadrant_sums)

        if q_sum_range < 0.1 * total_sum:
            return 'high_uniform' if total_mean >= 0.2 else 'low_uniform'

        return 'quarter-mixed'

        
    def write_df_to_csv(self, df, csv_path):
        csv_path = Path(csv_path)
        # Ensure the parent directory exists
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, mode='w', header=True, index=False, encoding='utf-8-sig')
        
            
    def append_df_to_csv(self, df, csv_path):
        csv_path = Path(csv_path)
        # Ensure the parent directory exists
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not csv_path.exists():  # Check if the file itself exists
            df.to_csv(csv_path, mode='w', header=True, index=False, encoding='utf-8-sig')
        else:
            df.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')


    def calc_silhouette(self, run_path=None, df=None):
        if df is not None:
            run_df = df
        elif run_path is not None:
            run_df = pd.read_csv(run_path, encoding='utf-8-sig')
        else:
            raise ValueError("Either run_path or df must be provided")

        # Calculate the mean and standard deviation of silhouette scores
        mean = run_df['silhouette_score'].mean()
        std = run_df['silhouette_score'].std()

        # Group by 'model' and calculate the count of negative silhouette scores per model
        negative_counts = run_df[run_df['silhouette_score'] < 0].groupby('model').size()

        # Calculate the average number of questions with negative silhouette scores per model
        avg_negative_per_model = negative_counts.mean()

        return mean, std, avg_negative_per_model
    
    
    def calc_alpha(self, run_path=None, df=None):
        if df is not None:
            run_df = df
        elif run_path is not None:
            run_df = pd.read_csv(run_path, encoding='utf-8-sig')
        else:
            raise ValueError("Either run_path or df must be provided")

        # Drop irrelevant columns
        relevant_columns = ['questionnaire', 'factor', 'ordinal', 'model', 'mean_score']
        run_df = run_df[relevant_columns]

        # Add a column combining questionnaire, factor and ordinal
        run_df['Q'] = (
                    run_df['questionnaire']
                    .str.cat(run_df['factor'], sep=':')
                    .str.cat(run_df['ordinal'].astype(str), sep=':')
            )

        # Calculate alpha for the entire questionnaire
        questionnaire_df = run_df[run_df['questionnaire'] == self.name]
        pivot = pd.pivot_table(questionnaire_df, values='mean_score', index='model', columns='Q', aggfunc='mean')
        total_alpha = pg.cronbach_alpha(data=pivot)[0]

        # Calculate alpha per factor
        factor_alphas = {}
        for group_name, factors in self._factor_grouping.items():

            factor_group_df = questionnaire_df[questionnaire_df['factor'].isin(factors)]
            pivot = pd.pivot_table(factor_group_df, values='mean_score', index='model', columns='Q', aggfunc='mean')
            # Check the number of columns
            if pivot.shape[1] < 2:
                raise ValueError(f"Factor {group_name} -> {factors} must have at least 2 questions to calculate cronbach Alpha.")
            # Check the number of rows
            if pivot.shape[0] < 1:
                raise ValueError(f"Factor {group_name} -> {factors} must have at least 1 model observation to calculate cronbach Alpha.")

            factor_alphas[group_name] = pg.cronbach_alpha(data=pivot)[0]

        return total_alpha, factor_alphas
    
    
    def calc_correlations(self, run_path=None, correlations_path=None, df=None):
        if df is not None:
            run_df = df
        elif run_path is not None:
            run_df = pd.read_csv(run_path, encoding='utf-8-sig')
        else:
            raise ValueError("Either run_path or df must be provided")

        # Drop irrelevant columns
        relevant_columns = ['questionnaire', 'factor', 'ordinal', 'model', 'mean_score']
        run_df = run_df[relevant_columns]
        questionnaire_df = run_df[run_df['questionnaire'] == self.name]

        # Iterate over factor_grouping and duplicate rows for each group
        for group_name, factors in self._factor_grouping.items():
            # If no grouping is done for a factor then skip it
            if factors == {group_name,}: continue
            # Find rows that match the current group
            group_rows = questionnaire_df[questionnaire_df['factor'].isin(factors)].copy()
            # Update the factor to the group name
            group_rows['factor'] = group_name
            # Append these rows to the questionnaire_df
            questionnaire_df = pd.concat([questionnaire_df, group_rows], ignore_index=True)

        pivot = pd.pivot_table(questionnaire_df, values='mean_score', index='model', columns='factor', aggfunc='mean')
        correlations = pivot.rcorr(method='spearman')

        if correlations_path is not None:
            self.write_df_to_csv(correlations, correlations_path)

        return correlations
    
    
    def calc_model_info(self, run_path):
        """
        Calculates statistics about the models run on the questionnaire.
        """
        run_df = pd.read_csv(run_path, encoding='utf-8-sig')
        model_groups = run_df.groupby('model')
        parameter_counts = {model: self.get_model_parameters(model) for model in model_groups.groups.keys()}
        valid_parameter_counts = [p for p in parameter_counts.values() if p is not None]
        num_models = len(parameter_counts)
        avg_parameters = np.mean(valid_parameter_counts) if valid_parameter_counts else 0
        std_parameters = np.std(valid_parameter_counts) if valid_parameter_counts else 0
        return num_models, avg_parameters, std_parameters
    
    
    def get_model_parameters(self, model_path):
        """
        Fetches the number of parameters for a Hugging Face model.
        """
        try:
            model = AutoModel.from_pretrained(model_path)
            return model.num_parameters()
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            return None        
        

    ##### DASHBOARD #####
    def report(self, run_path, output_path, template_path="./report_template.md"):
        """
        Generates a report for the questionnaire analysis and saves it as a Markdown file.
    
        Args:
            run_path (str): Path to the run directory containing required data for calculations.
            output_path (str): Path where the rendered Markdown file will be saved.
            template_path (str): Path to the Markdown template file.
    
        The Markdown template file should use placeholders and loops compatible with the Jinja2 templating engine.
        Example placeholders:
            - {{ questionnaire_name }}
            - {{ avg_silhouette }}
        Example loops:
            - {% for factor in factors %} ... {% endfor %}
        """
    
        # Step 1: Read the Markdown template from a file
        with open(template_path, "r", encoding="utf-8") as file:
            markdown_template = file.read()
    
        # Step 2: Define the dynamic data
        avg_silhouette, std_silhouette, num_negative_silhouette = self.calc_silhouette(run_path)
        cronbach_alpha_total, cronbach_alpha_factors = self.calc_alpha(run_path)
        num_models, avg_parameters, std_parameters = self.calc_model_info(run_path)
        data = {
            "questionnaire_name": f"{self.full_name} ({self.name})" if self.full_name != self.name else self.name,
            "num_items": self.num_of_questions,
            "factors": self.factors.keys(),
            "avg_silhouette": f"{avg_silhouette:.4f}",
            "std_silhouette": f"{std_silhouette:.4f}",
            "num_negative_silhouette": f"{num_negative_silhouette:.4f}",
            "cronbach_alpha_total": f"{cronbach_alpha_total:.4f}",
            "cronbach_alpha_factors": {factor: f"{alpha:.4f}" for factor, alpha in cronbach_alpha_factors.items()},
            "num_models": num_models,
            "avg_parameters": f"{avg_parameters:.4f}",
            "std_parameters": f"{std_parameters:.4f}",
        }
    
        # Step 3: Render the Markdown content
        template = Template(markdown_template)
        rendered_markdown = template.render(data)
        
        # Save Markdown content to a file
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(rendered_markdown)
        
        return rendered_markdown


    
    def create_dashboard(self, run_path, filename):
        """Creates the dashboard with a widescreen 16:9 aspect ratio."""
        
        # Define dimensions for 16:9 aspect ratio
        width = 16 * inch  # Width (16 units)
        height = 9 * inch  # Height (9 units)

        # Initialize canvas with custom dimensions
        c = canvas.Canvas(filename, pagesize=(width, height))

        # Dimensions
        panel_width = width * 0.25  # 25% for the left panel

        # Draw the left panel
        questionnaire_name = f"{self.full_name} ({self.name})" if self.full_name != self.name else self.name
        num_items = self.num_of_questions
        factors = self.factors.keys()
        self.draw_left_panel(c, width, height, panel_width, questionnaire_name, num_items, factors)

        # Draw the top-left silhouette section
        avg_silhouette, std_dev, num_negative_silhouette = self.calc_silhouette(run_path)
        self.draw_silhouette_section(c, panel_width, width, height, avg_silhouette, std_dev, num_negative_silhouette)

        # Draw the bottom-left alpha section
        cronbach_alpha_total, cronbach_alpha_factors = self.calc_alpha(run_path)
        self.draw_alpha_section(c, panel_width, width, height, cronbach_alpha_total, cronbach_alpha_factors)

        # Draw the top-right domain adaptation section
        self.draw_domain_adaptation_section(c, panel_width, width, height)

        # Calculate model information
        num_models, avg_parameters, std_parameters = self.calc_model_info(run_path)
        # Draw the bottom-right model information section
        self.draw_model_section(c, panel_width, width, height, num_models, avg_parameters, std_parameters)

        # Save the PDF
        c.save()

    def draw_wrapped_text(self, c, x, y, width, text):
        """Utility function to draw wrapped text with increased line spacing."""
        styles = getSampleStyleSheet()
        style = styles["BodyText"]
        style.fontSize = 14  # Increased font size for content text
        style.leading = 18  # Line spacing (increased for better readability)
        para = Paragraph(text, style)
        _, h = para.wrap(width, 200)  # Wrap text within the specified width
        para.drawOn(c, x, y - h)  # Adjust y to prevent overlap
        return h  # Return the height of the wrapped text


    def draw_left_panel(self, c, width, height, panel_width, questionnaire_name, num_items, factors):
        """Draws the left panel with proper spacing for wrapped text."""
        margin = 20
        y_position = height - margin  # Start from the top

        # Draw panel background
        c.setFillColor(colors.lightgrey)
        c.rect(0, 0, panel_width, height, fill=True)

        # Add panel title
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 18)  # Increased header font size
        c.drawCentredString(panel_width / 2, height - margin - 20, "Questionnaire Summary")
        y_position -= 50

        # Add questionnaire name
        y_position -= self.draw_wrapped_text(c, margin, y_position, panel_width - 2 * margin, f"Name: {questionnaire_name}") + 15

        # Add number of items
        y_position -= self.draw_wrapped_text(
            c,
            margin,
            y_position,
            panel_width - 2 * margin,
            f"Number of Items: {num_items:.4f}" if isinstance(num_items, float) else f"Number of Items: {num_items}",
        ) + 15

        # Add factors
        factors_text = "Factors:<br/>" + "<br/>".join(f"- {factor}" for factor in factors)
        y_position -= self.draw_wrapped_text(c, margin, y_position, panel_width - 2 * margin, factors_text)

        # Draw panel border
        c.setStrokeColor(colors.black)
        c.setLineWidth(1)
        c.rect(0, 0, panel_width, height, fill=False)

    def draw_silhouette_section(self, c, panel_width, width, height, avg_silhouette, std_dev, num_negative_silhouette):
        """Draws the silhouette section with proper spacing."""
        section_width = (width - panel_width) / 2
        section_height = height / 2
        margin = 20
        y_position = height - margin

        # Add section title
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 18)  # Increased header font size
        c.drawCentredString(panel_width + section_width / 2, height - margin - 20, "Silhouette Information")
        y_position -= 50

        # Add silhouette details
        silhouette_text = (
            f"Average Silhouette: {avg_silhouette:.4f}<br/>"
            f"Standard Deviation: {std_dev:.4f}<br/>"
            f"Number of Questions with Negative Silhouette(Avg per model): {num_negative_silhouette:.4f}"
        )
        self.draw_wrapped_text(c, panel_width + margin, y_position, section_width - 2 * margin, silhouette_text)

        # Draw section border
        c.setStrokeColor(colors.black)
        c.setLineWidth(1)
        c.rect(panel_width, height - section_height, section_width, section_height, fill=False)

    def draw_alpha_section(self, c, panel_width, width, height, cronbach_alpha_total, cronbach_alpha_factors):
        """Draws the Cronbach's Alpha Information section with proper spacing."""
        section_width = (width - panel_width) / 2
        section_height = height / 2
        margin = 20
        y_position = section_height - margin

        # Add section title
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 18)  # Increased header font size
        c.drawCentredString(panel_width + section_width / 2, section_height - margin - 20, "Cronbach's Alpha Information")
        y_position -= 50

        # Add Cronbach's alpha details
        alpha_text = (
            f"Total Alpha: {cronbach_alpha_total:.4f}<br/><br/>"
            + "Factors:<br/>"
            + "<br/>".join(f"- {factor}: {alpha:.4f}" for factor, alpha in cronbach_alpha_factors.items())
        )
        self.draw_wrapped_text(c, panel_width + margin, y_position, section_width - 2 * margin, alpha_text)

        # Draw section border
        c.setStrokeColor(colors.black)
        c.setLineWidth(1)
        c.rect(panel_width, 0, section_width, section_height, fill=False)

    def draw_model_section(self, c, panel_width, width, height, num_models, avg_parameters, std_parameters):
        """Draws the model information section."""
        section_width = (width - panel_width) / 2
        section_height = height / 2
        margin = 20
        y_position = section_height - margin

        # Add section title
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 18)  # Increased header font size
        c.drawCentredString(panel_width + section_width + section_width / 2, section_height - margin - 20, "Model Information")
        y_position -= 50

        # Add model statistics
        model_text = (
            f"Number of Models: {num_models}<br/>"
            f"Average Parameters per Model: {avg_parameters:.4f}<br/>"
            f"Standard Deviation of Parameters: {std_parameters:.4f}"
        )
        self.draw_wrapped_text(c, panel_width + section_width + margin, y_position, section_width - 2 * margin, model_text)

        # Draw section border
        c.setStrokeColor(colors.black)
        c.setLineWidth(1)
        c.rect(panel_width + section_width, 0, section_width, section_height, fill=False)

    def draw_domain_adaptation_section(self, c, panel_width, width, height):
        """
        Draws the domain adaptation section in the top-right of the dashboard.
        """
        section_width = (width - panel_width) / 2
        section_height = height / 2
        margin = 20
        y_position = height - margin

        # Add section title
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 18)  # Increased header font size
        c.drawCentredString(panel_width + section_width + section_width / 2, height - margin - 20, "Domain Adaptation Information")
        y_position -= 50

        # Placeholder values are currently shown.
        domain_text = (
            "Learning Rate: [To be implemented]<br/>"
            "Number of Epochs: [To be implemented]<br/>"
            "Average Change in Mean Score per Epoch: [To be implemented]<br/>"
            "Standard Deviation of Change in Mean Score per Epoch: [To be implemented]"
        )
        self.draw_wrapped_text(c, panel_width + section_width + margin, y_position, section_width - 2 * margin, domain_text)

        # Draw section border
        c.setStrokeColor(colors.black)
        c.setLineWidth(1)
        c.rect(panel_width + section_width, height - section_height, section_width, section_height, fill=False)

    


##################
#   FUNCTIONS   #
##################


def apply_filters(Q, filters_dict):
    return {key: func(Q) for key, func in filters_dict.items()}

def print_permutations(q):
    W = q._pdf['W']
    print(q._descriptor)
    for i, (kmap, w) in enumerate(zip(q._pdf.drop(columns=['P', 'W']).to_dict(orient='records'), W)):
        context = q._context_template.format_map(kmap)
        answer = q._answer_template.format_map(kmap)
        print(f'{i}.',context ,'->', answer, w)

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
    
class QMNLIQuestionnaire(Questionnaire):
    
    def run(self, models:List[str]=None,
            questions_ordinals:List[int]=None, 
            result_path=Path('./results/result.csv'), 
            softmax:List[str]=[], 
            filters:Dict[str,dict]={}):
                
        super().run(models = models,
            pipeline_builder = lambda model: pipeline("zero-shot-classification",device=device, model=model),
            questions_ordinals = questions_ordinals,
            result_path = result_path,
            softmax = softmax,
            filters = filters)
        
        
    def question_attributes(self, q):
        attributes = super().question_attributes(q)
        attributes.update({"translation_method" : 'NLI'})        
        return attributes
        
        
class QMLMQuestionnaire(Questionnaire):
    
    def run(self, models:List[str]=None,
            questions_ordinals:List[int]=None, 
            result_path=Path('./results/result.csv'), 
            softmax:List[str]=[], 
            filters:Dict[str,dict]={}):
        
        super().run(models = models,
            pipeline_builder = lambda model: pipeline('fill-mask',device=device, model=model),
            questions_ordinals = questions_ordinals,
            result_path = result_path,
            softmax = softmax,
            filters = filters)
    
    
    def question_attributes(self, q):
        attributes = super().question_attributes(q)
        attributes.update({"translation_method" : 'MLM'})        
        return attributes


from qlatent.qnsp.qnsp import NextSentencePredictionPipeline
class QNSPQuestionnaire(Questionnaire):
    
    def run(self, models:List[str]=None,
            questions_ordinals:List[int]=None, 
            result_path=Path('./results/result.csv'), 
            softmax:List[str]=[], 
            filters:Dict[str,dict]={}):
        
        super().run(models = models,
            pipeline_builder = lambda model: NextSentencePredictionPipeline(model_name=model, device=device),
            questions_ordinals = questions_ordinals,
            result_path = result_path,
            softmax = softmax,
            filters = filters)
        
    
    def question_attributes(self, q):
        attributes = super().question_attributes(q)
        attributes.update({"translation_method" : 'NSP'})        
        return attributes
    

class QCOLAQuestionnaire(Questionnaire):
    
    def run(self, models:List[str]=None,
            questions_ordinals:List[int]=None, 
            result_path=Path('./results/result.csv'), 
            softmax:List[str]=[], 
            filters:Dict[str,dict]={}):
        
        super().run(models = models,
            pipeline_builder = lambda model: pipeline("text-classification",device=device, model=model),
            questions_ordinals = questions_ordinals,
            result_path = result_path,
            softmax = softmax,
            filters = filters)
        
        
    def question_attributes(self, q):
        attributes = super().question_attributes(q)
        attributes.update({"translation_method" : 'COLA'})        
        return attributes
        
##################
# VISUALIZATION  #
##################


def run_models_on_question_and_extract_mean_scores(q, models):
    mean_scores_by_checkpoints={}
    for model in models:
        mean_score=q.run(model).mean_score()
        mean_scores_by_checkpoints[model.checkpoint] = mean_score
    return mean_scores_by_checkpoints

def visualize_checkpoint_scores(q, models):
    checkpoint_scores = run_models_on_question_and_extract_mean_scores(q, models)
    
    constant=checkpoint_scores[0]
    # Sort the dictionary by checkpoint (keys)
    checkpoints = sorted(checkpoint_scores.keys())
    mean_scores = [checkpoint_scores[cp] for cp in checkpoints]
    
    # Calculate y-axis padding based on the range of mean_scores and constant
    all_scores = mean_scores + [constant]
    y_min, y_max = min(all_scores), max(all_scores)
    y_range = y_max - y_min
    
    # Add padding (10% of the range) to y_min and y_max to stretch the plot
    padding = 0.1 * y_range if y_range != 0 else 0.1  # If range is 0, use a small constant padding
    y_min_padded = y_min - padding
    y_max_padded = y_max + padding
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot the checkpoint scores (green line connecting points)
    plt.plot(checkpoints, mean_scores, 'o-', color='green')
    
    # Plot the constant line (red dashed)
    plt.axhline(y=constant, color='red', linewidth=0.8, linestyle='--', label=f'Unbiased mean score (Score={constant})')
    
    # Set x-axis and y-axis ticks to match the dictionary keys and values
    plt.xticks(checkpoints)  # Exact checkpoint values on x-axis
    
    # Make sure the x-axis starts from 0 and is aligned with the y-axis
    plt.xlim(left=0)
    
    # Apply the padded y-axis limits
    plt.ylim(y_min_padded, y_max_padded)
    
    # Add titles and labels
    plt.title('Mean Scores vs Checkpoints')
    plt.xlabel('Checkpoint')
    plt.ylabel('Mean Score')
    
    # Add legend with dynamic labels
    handles, labels = plt.gca().get_legend_handles_labels()
    for cp, score in zip(checkpoints, mean_scores):
        handles.append(plt.Line2D([0], [0], marker='o', color='green', markersize=10, linestyle='-', label=f'Checkpoint {cp} (Score: {checkpoint_scores[cp]})'))
    
    # Display the plot
    plt.grid(True)
    plt.show()



def model_accuracy_on_dataset_per_da_epoch(accuracies : List[float], dataset_name : str) -> None:

    epochs_num = len(accuracies) - 1
    
    title = f"""Accuracy Scores for {dataset_name} in Every DA Epoch"""

    acc_color = "purple"
    
    legend_elements = [
        Patch(facecolor=acc_color, edgecolor=acc_color, label='Accuracy'),
    ]

    fig, ax = plt.subplots()

    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.subplots_adjust(right=0.75)
    
    for epoch_num in range(epochs_num):
        
        acc_xs, acc_ys = [epoch_num, epoch_num + 1], [accuracies[epoch_num], accuracies[epoch_num + 1]]

        plt.plot(acc_xs, acc_ys, color=acc_color)

    all_xs = [i for i in range(epochs_num + 1)]
    plt.scatter(all_xs, accuracies, color=acc_color)
    
    for epoch_num in range(0, epochs_num + 1): 
        plt.plot([epoch_num] * 2, [-0.15, 100.15], color="grey", linestyle="dashed")
    
    plt.plot([-0.5, epochs_num + 0.5], [0] * 2, color="black")
    plt.title(title, loc="left")
    plt.xlabel("DA Epoch Number")
    plt.ylabel("Model Accuracy when Tested on the Mentioned Dataset (%)")
    
    plt.show()
    
    
    

        