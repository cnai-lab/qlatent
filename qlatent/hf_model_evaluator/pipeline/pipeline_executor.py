import os
from pathlib import Path
from typing import List, Literal
import sys
import traceback

from qlatent.hf_model_evaluator.utils.file_utils import *
from qlatent.hf_model_evaluator.utils.logging_utils import LoggingUtils
from qlatent.hf_model_evaluator.utils.model_utils import ModelUtils
from qlatent.qmnli.qmnli import *
from qlatent.qmlm.qmlm import *
from .questionnaire import *
from huggingface_hub import HfApi
from qlatent.hf_model_evaluator.utils.model_acc_evaluator import *

file_utils = FileUtils()
hf_api = HfApi()


class InitiatePipeline:
    def __init__(
        self,
        models_info: List[str],
        questionnaires_questions_lists: List[List["Question"]],
        merge_filtered_positiveonly: bool = True,
        override_results: bool = False,
        base_dir: str = "./model_logs",
        cache_dir: str = str(Path.home() / ".cache" / "huggingface" / "hub"),
    ):
        self.cache_dir = cache_dir
        self.override_results = override_results
        self.base_dir = base_dir
        self.models_info = models_info
        self.questionnaires_questions_lists = questionnaires_questions_lists
        self.merge_filtered_positiveonly = merge_filtered_positiveonly
        self.models_evaluator = ModelsEvaluator(self.base_dir + "/models_acc.csv")
        self.questionnaires_obj = Questionnaires(questionnaires_questions_lists)
        self.prepare_pipelines()

    def prepare_pipelines(self):
        import torch

        questionnaire_num_questions = self.questionnaires_obj.get_questionnaire_num_questions()
        FileUtils.remove_uncompleted_models_evals(questionnaire_num_questions, self.merge_filtered_positiveonly, self.base_dir)
        self.model_evaluated_questionnaires = file_utils.set_logged_questionnaires_each_model(self.base_dir)

        models_not_evaluated = 0
        cached_models = 0

        for model_info in self.models_info:
            model_info = hf_api.model_info(model_info.id)
            model_name = model_info.id
            print(f"📊 Evaluating model: {model_name}", flush=True)

            try:
                try:
                    last_commit = hf_api.list_repo_commits(repo_id=model_name)[0]
                    model_info.last_commit_hash = last_commit.commit_id
                    model_info.last_commit_date = last_commit.created_at.strftime("%d-%m-%Y %H:%M:%S") + " UTC"
                    model_info.model_version_id = model_name + "_" + model_info.last_commit_hash
                except Exception as e:
                    file_utils.log_model_errors(model_name, "Gated repo:" + str(e), self.base_dir)
                    logging_utils.log_error(f"Failed to initialize model {model_name}: {e}")
                    continue

                if ModelUtils.exist_in_error_logs(model_info.model_version_id, self.base_dir):
                    continue

                if model_info.pipeline_tag in ("zero-shot-classification", "text-classification"):
                    questionnaire_type = NLI_TYPE
                elif model_info.pipeline_tag == "fill-mask":
                    questionnaire_type = MLM_TYPE
                else:
                    raise ValueError(f"Unsupported pipeline_tag: {model_info.pipeline_tag}")

                file_utils.log_current_model(model_info.model_version_id, self.base_dir)

                if model_info.model_version_id in self.model_evaluated_questionnaires:
                    model_type = self.model_evaluated_questionnaires[model_info.model_version_id][0]
                    model_evaluated_questionnaires = self.model_evaluated_questionnaires[model_info.model_version_id][1]
                    if len(model_evaluated_questionnaires) == len(self.questionnaires_obj.questionnaires[model_type]):
                        print(f"  ✓ All questionnaires already evaluated: {', '.join(model_evaluated_questionnaires)}", flush=True)
                        file_exists, model_already_logged = FileUtils.model_exist_meta_data(model_info.model_version_id, self.override_results, self.base_dir)
                        if file_exists and model_already_logged:
                            continue
                        pipeline = ModelUtils.load_pipeline_safely(model_info, self.base_dir)
                        file_utils.log_model_meta_data(model_info, pipeline, self.override_results, self.base_dir)
                        continue

                pipeline = ModelUtils.load_pipeline_safely(model_info, self.base_dir)
                if not pipeline:
                    continue

                self.execute_pipeline_save_results(model_info, pipeline, questionnaire_type, self.merge_filtered_positiveonly)

                if questionnaire_type == NLI_TYPE:
                    self.models_evaluator.get_models_accuracies(
                        models=[model_info.model_version_id],
                        has_model_version_id=True,
                        clear_file_errors=False
                    )

            finally:
                cached_models += 1
                if 'pipeline' in locals():
                    ModelUtils.cleanup_pipeline_safely(pipeline)
                ModelUtils.cleanup_cache(self.cache_dir)

        print(f"📈 Final stats: {models_not_evaluated} models not evaluated out of {len(self.models_info)} total models", flush=True)

    def execute_pipeline_save_results(self, model_info, pipeline, questionnaire_type, merge_filtered_positiveonly):
        try:
            model_id = model_info.id
            questionnaires_evaluated = (
                self.model_evaluated_questionnaires[model_info.model_version_id][1]
                if self.model_evaluated_questionnaires[model_info.model_version_id]
                else []
            )

            for questionnaire in self.questionnaires_obj.questionnaires[questionnaire_type]:
                if questionnaire.name in questionnaires_evaluated:
                    print(f"  ✓ {questionnaire.name} already evaluated for {model_id}", flush=True)
                    continue

                print(f"  📝 Evaluating {questionnaire.name} for {model_id}", flush=True)
                questionnaire.run(
                    pipelines=[pipeline],
                    softmax=['index', 'frequency'],
                    filters={
                        "unfiltered": lambda q: {},
                        "positive_only": (lambda q: q.get_filter_for_postive_keywords(['frequency'])),
                    },
                    result_path=f"{self.base_dir}/{questionnaire.questionnaire_type}/{questionnaire.name}.csv",
                    merge_filtered_positiveonly=merge_filtered_positiveonly,
                )

            file_utils.log_model_meta_data(model_info, pipeline, self.override_results, self.base_dir)

        except Exception as e:
            file_utils.log_model_errors(model_info.model_version_id, str(e), self.base_dir)
            error_trace = traceback.format_exc()
            logging_utils.log_error(
                f"{model_info.model_version_id} failed on {questionnaire.name}:\n{error_trace}"
            )
