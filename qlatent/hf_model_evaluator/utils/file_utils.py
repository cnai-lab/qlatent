import pandas as pd
import sys
from .logging_utils import LoggingUtils
import os
import shutil
from datetime import datetime, timezone
import time
from collections import defaultdict
from typing import List, Literal
from pathlib import Path, PurePosixPath
import csv
import pytz
from qlatent.hf_model_evaluator.global_variables import *
from .logging_utils import LoggingUtils
logging_utils = LoggingUtils()
from huggingface_hub import HfApi
hf_api = HfApi()

from dateutil import parser
class FileUtils():
    def __init__(self):
        pass
    
    @staticmethod
    def count_directories(path):
        """Counts the number of non-hidden directories in the given path."""
        directory_count = 0
        try:
            with os.scandir(path) as entries:
                for entry in entries:
                    # Check if it's a directory and does not start with a period
                    if entry.is_dir() and not entry.name.startswith('.'):
                        directory_count += 1
        except Exception as e:
            print(f"Error in counting directories: {e}")
        return directory_count
    
    @staticmethod
    def safe_rmtree(path, retries=3, delay=5):
        if not os.path.isdir(path):
            return
        if FileUtils.count_directories(path) > 200:
            logging_utils.log_error(f"Too many models not deleted, stopping process!")
            sys.exit(1)
        if FileUtils.count_directories(path) >= 5:
            for i in range(retries):
                try:
                    shutil.rmtree(path)
                    return
                except Exception as e:
                    if e.errno == 16:
                        logging_utils.log_error(f"Can not proceed with deletion: {e}\n\n\n\n")
                        return
                    else:
                        logging_utils.log_error(f"Can not proceed with deletion: {e}\n\n\n\n")
                    if i < retries - 1:
                        time.sleep(delay)
                    
                    
    @staticmethod
    def check_disk_quota_error(e):
        if isinstance(e, OSError) and ("quota exceeded" in str(e) or "No space left on device" in str(e)):
            # Log the error and stop execution
            logging_utils.log_error(f"Disk quota error: {e}")
            sys.exit(1)  # Exit the program with a non-zero status to indicate an error
     
            
    @staticmethod
    def set_logged_questionnaires_each_model(base_dir):
        model_evaluated_questionnaires = defaultdict(list)
        base_dir = Path(base_dir)

        if not base_dir.exists():
            return model_evaluated_questionnaires

        # List model directories (e.g., 'typeform_distilbert-base-uncased-mnli')
        files_base_dir = os.listdir(base_dir)
        folders_base_dir = [base_dir / folder for folder in files_base_dir if os.path.isdir(base_dir / folder) and (folder == MLM_TYPE or folder == NLI_TYPE)]


        for folder_path in folders_base_dir:

            model_evaluated_questionnaires
            # List files inside the subdirectory (e.g., 'ASI_evaluation.csv')
            questionnaires_files = os.listdir(folder_path)
            questionnaires_type = str(PurePosixPath(folder_path)).split("/")[-1]
            this_type_models = []
            for questionnaire in questionnaires_files:

                questionnaire_path = folder_path / questionnaire
                questionnaire_df = pd.read_csv(questionnaire_path, encoding='utf-8-sig')
                all_logged_models = questionnaire_df[MODEL_IDENTIFIER].to_list()
                for model in set(all_logged_models):
                    questionnaire_name = questionnaire.split(".")[0]
                    model_evaluated_questionnaires[model].append(questionnaire_name)
                    this_type_models.append(model)
            
            for model in set(this_type_models):
                questionnaires_in_this_model = model_evaluated_questionnaires[model]
                model_evaluated_questionnaires[model] = (questionnaires_type, questionnaires_in_this_model)

        return model_evaluated_questionnaires
    
    
    def remove_uncompleted_models_evals(questionnaire_num_questions, merge_filtered_positiveonly, base_dir):
        base_dir = Path(base_dir)

        if not base_dir.exists():
            return

        # List model directories (e.g., 'typeform_distilbert-base-uncased-mnli')
        files_base_dir = os.listdir(base_dir)
        folders_base_dir = [base_dir / folder for folder in files_base_dir if os.path.isdir(base_dir / folder)]

        for folder_path in folders_base_dir:
            # List files inside the subdirectory (e.g., 'ASI_evaluation.csv')
            questionnaires_files = os.listdir(folder_path)
            for questionnaire in questionnaires_files:
                questionnaire_name = questionnaire.split(".")[0]

                if questionnaire_name not in list(questionnaire_num_questions.keys()):
                    continue

                questionnaire_path = folder_path / questionnaire
                questionnaire_df = pd.read_csv(questionnaire_path, encoding='utf-8-sig')
                logged_models = questionnaire_df[MODEL_IDENTIFIER].to_list()

                # Check if there are any models logged - skip if empty
                if len(logged_models) == 0:
                    continue

                last_logged_model = logged_models[-1]

                num_of_questions = questionnaire_num_questions[questionnaire_name]
                num_of_questions = num_of_questions if merge_filtered_positiveonly else num_of_questions*2 # multiplied by 2 since each questions is logged twice for unfiltered and positiveonly
                if logged_models.count(last_logged_model)!=num_of_questions:
                    questionnaire_df = questionnaire_df[questionnaire_df[MODEL_IDENTIFIER] != last_logged_model] # delete all model instances in the file
                    questionnaire_df.reset_index(inplace=True, drop=True)
                    questionnaire_df.to_csv(questionnaire_path, encoding='utf-8-sig', index=False)
                    logging_utils.log_warning(f"Model {last_logged_model} wasn't evaluated with all questions in {questionnaire_name}, model deleted from file for re-evaluation!")

        
    
    
    
    
    @staticmethod
    def model_exists_in_logs(csv_log_file, model_id, model_name_column="model_version_id"):
        # If the file doesn't exist, return False immediately.
        # Though this check might be redundant if you perform it before calling this function.
        if not csv_log_file.is_file():
            return False

        with open(csv_log_file, mode='r', newline='', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Iterate through all rows in the CSV
            for row in reader:
                if str(row.get(model_name_column)) == str(model_id):
                    return True
        
        return False

    @staticmethod
    def _get_scale(number):
        if number < 1_000:
            return f"{number}"
        elif number < 1_000_000:
            return f"{number / 1_000:.2f}_K (Thousands)"
        elif number < 1_000_000_000:
            return f"{number / 1_000_000:.2f}_M (Millions)"
        elif number < 1_000_000_000_000:
            return f"{number / 1_000_000_000:.2f}_B (Billions)"
        else:
            return f"{number / 1_000_000_000_000:.2f}_T (Trillions)"


    @staticmethod
    def model_exist_meta_data(model_version_id, override_results, base_dir):
        base_directory = Path(base_dir)
        if not base_directory.is_dir():
            # if one of parents doesn't exist raise an error, if the directory already exists also raise an error.
            base_directory.mkdir(parents=True, exist_ok=False)
            print(f"Logs Directory '{str(base_directory)}' has been created.", flush=True)

        csv_log_file = base_directory / META_DATA_PATH
        file_exists = csv_log_file.is_file()
        model_already_logged = FileUtils.model_exists_in_logs(csv_log_file, model_version_id, "model_version_id")
        if file_exists and model_already_logged and not override_results:
            print(f"\t================================================================\n\tMeta data has been already logged for {model_version_id}!\n\t================================================================", flush=True)
            return file_exists, model_already_logged
        return file_exists, model_already_logged

    @staticmethod
    def log_model_meta_data(model_info, pipeline, override_results, base_dir):
        
        file_exists, model_already_logged = FileUtils.model_exist_meta_data(model_info.model_version_id, override_results, base_dir)
        if file_exists and model_already_logged:
            return
        
        model_id = model_info.id
        model_info.last_commit_hash = model_info.last_commit_hash
        model_info.last_commit_date = model_info.last_commit_date
        model_info.model_version_id = model_info.model_version_id
        
        
        now = datetime.now(pytz.UTC)
        date_of_logging = now.strftime('%d-%m-%Y %H:%M:%S %Z')
    
        last_commit_hash = model_info.last_commit_hash
        commit_date = model_info.last_commit_date
        
        
        id = model_info.id if model_info.id else "unknown"
        author = model_info.author if model_info.author else "unknown"        
        
        if hasattr(model_info, "created_at"):
            dt = model_info.created_at
        else:
            dt = parser.parse(model_info.createdAt)
            
        created_at = dt.strftime("%d-%m-%Y %H:%M:%S") + " UTC"
        
        downloads = model_info.downloads if model_info.downloads else "unknown"
        #downloads_all_time = model_info.downloads_all_time if model_info.downloads_all_time else "unknown"
        likes = model_info.likes if model_info.likes else "unknown"
        library_name = model_info.library_name if hasattr(model_info, "library_name") else "unknown"

        if hasattr(model_info, "original_pipeline_tag"):
            pipeline_tag = model_info.original_pipeline_tag
        else:
            pipeline_tag = model_info.pipeline_tag if model_info.pipeline_tag else "unknown"

        
        architectures = model_info.config['architectures'] if model_info.config and "architectures" in model_info.config and model_info.config['architectures'] else "unknown"
        model_type = model_info.config['model_type'] if model_info.config and "model_type" in model_info.config and model_info.config['model_type'] else "unknown"
        base_model = model_info.cardData['base_model'] if model_info.cardData and "base_model" in model_info.cardData and model_info.cardData['base_model'] else "unknown"
        fine_tune_base_model = [tag.split('base_model:finetune:')[1] for tag in model_info.tags if 'base_model:finetune:' in tag] or "unknown"
        datasets = model_info.cardData['datasets'] if model_info.cardData and "datasets" in model_info.cardData and model_info.cardData['datasets'] else [tag.split('dataset:')[1] for tag in model_info.tags if 'dataset:' in tag] or "unknown"
        language = model_info.cardData['language'] if model_info.cardData and "language" in model_info.cardData and model_info.cardData['language'] else "unknown"
        #trainable_params = FileUtils._get_scale(model_info.safetensors['total']) if model_info.safetensors else pipeline.trainable_params if pipeline.trainable_params else "unknown"
        if not pipeline:
            pass
        # trainable_params = model_info.safetensors['total'] if hasattr(model_info,"safetensors") else pipeline.trainable_params if pipeline.trainable_params else "unknown"
        trainable_params = pipeline.trainable_params if pipeline.trainable_params else "unknown"
 
        
        
        vocab_size = pipeline.vocab_size if pipeline.vocab_size else "unknown"
        region = next((tag.split('region:')[1] for tag in model_info.tags if 'region:' in tag), "unknown")

        model_info_dict = {
            "id": id,
            "author": author,
            "last_commit_hash": last_commit_hash,
            "last_commit_date": commit_date,
           # "last_modified": last_modified,
            MODEL_IDENTIFIER: model_info.model_version_id,
            "created_at": created_at,
            "downloads": downloads,
            #"downloads_all_time": downloads_all_time,
            "likes": likes,
            "library_name": library_name,
            "pipeline_tag": pipeline_tag,
            "architectures" : architectures,
            "model_type" : model_type,
            "base_model" : base_model,
            "fine_tune_base_model" : fine_tune_base_model,
            "datasets" : datasets,
            "language" : language,
            "trainable_params" : trainable_params,
            "vocab_size": vocab_size,
            "region": region,
            "date_of_logging": date_of_logging,
        }

        # Convert the single-row dictionary to a DataFrame
        new_df = pd.DataFrame([model_info_dict])
        base_directory = Path(base_dir)
        csv_log_file = base_directory / META_DATA_PATH
        if file_exists:
            # If file exists, read it into a DataFrame
            df = pd.read_csv(csv_log_file, encoding='utf-8-sig')

            if model_already_logged and override_results:
                # Remove old rows for this model and replace with new entry
                df = df[df["id"] != model_info.id]
                df = pd.concat([df, new_df], ignore_index=True)
                #print(f"Overridden the meta data for {model_info.id}.")
            else:
                # Model not logged or override not needed, just append new row
                df = pd.concat([df, new_df], ignore_index=True)
                #print(f"Logged meta data for {model_info.id}.")
        else:
            # File does not exist, create a new DataFrame from the single row
            df = new_df
            #print(f"Created new logs and logged meta data for {model_info.id}.")

        # Write the DataFrame back to CSV
        df.to_csv(csv_log_file, index=False, encoding='utf-8-sig')
        
    @staticmethod
    def log_model_errors(model_name, error, base_dir):
        base_directory = Path(base_dir)
        if not base_directory.is_dir():
            base_directory.mkdir(parents=True, exist_ok=False)
            print(f"\tLogs Directory '{str(base_directory)}' has been created.", flush=True)   
        
        csv_log_file = base_directory / MODELS_ERROR_PATH
        file_exists = csv_log_file.is_file()
        
        model_error = {
            MODEL_IDENTIFIER: model_name,
            "error" : error,
        }
        
        new_df = pd.DataFrame([model_error])
        
        if file_exists:
            # If file exists, read it into a DataFrame
            df = pd.read_csv(csv_log_file, encoding='utf-8-sig')
            df = pd.concat([df, new_df], ignore_index=True)
            #print(f"Logged meta data for {model_info.id}.")
        else:
            # File does not exist, create a new DataFrame from the single row
            df = new_df

        # Write the DataFrame back to CSV
        df.to_csv(csv_log_file, index=False, encoding='utf-8-sig')
        
        
    @staticmethod
    def log_current_model(model_name, base_dir):
        base_directory = Path(base_dir)
        if not base_directory.is_dir():
            base_directory.mkdir(parents=True, exist_ok=False)
            print(f"\tLogs Directory '{str(base_directory)}' has been created.", flush=True)   
        
        csv_log_file = base_directory / "current_model.csv"
        file_exists = csv_log_file.is_file()
        
        model_info = {
            MODEL_IDENTIFIER: model_name,
            "timestamp": pd.Timestamp.now(tz='UTC').strftime('%d-%m-%Y %H:%M:%S'),
            "status": "processing"
        }
        
        new_df = pd.DataFrame([model_info])
        
        if file_exists:
            df = pd.read_csv(csv_log_file, encoding='utf-8-sig')
            if model_name in list(df[MODEL_IDENTIFIER]):
                return # model already logged
            df = pd.concat([df, new_df], ignore_index=True)
        else:
            df = new_df

        df.to_csv(csv_log_file, index=False, encoding='utf-8-sig')
        return True
