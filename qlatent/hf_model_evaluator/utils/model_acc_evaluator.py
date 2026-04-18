import os
import gc
import json
import traceback
from pathlib import Path
from typing import List, Callable

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

hf_api = HfApi()
mnli_val_dataset = load_dataset("multi_nli")["validation_matched"]


class ModelsEvaluator:
    def __init__(self, acc_csv_output_path: str):
        self.acc_csv_output_path = acc_csv_output_path
        self.mnli_val = mnli_val_dataset

    def create_predict_function(self, model_name: str, has_model_version_id: bool) -> Callable[[str, str], int]:
        torch.cuda.empty_cache()
        gc.collect()

        model_name = model_name.rsplit("_", 1)[0]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        if model.config.num_labels != 3:
            raise ValueError(f"Model {model_name} has {model.config.num_labels} labels, expected 3 for NLI")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        id2label = model.config.id2label
        model_labels_lower = [str(l).lower().strip() for l in id2label.values()]
        if sorted(model_labels_lower) != sorted(["entailment", "neutral", "contradiction"]):
            raise ValueError(f"Model {model_name} labels {list(id2label.values())} do not match expected NLI labels")

        print(f"Model label mapping: {id2label}")

        mnli_label_to_id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        model_to_mnli = {
            model_id: mnli_label_to_id[label_name.lower()]
            for model_id, label_name in id2label.items()
        }

        def predict_one(premise: str, hypothesis: str) -> int:
            inputs = tokenizer(premise, hypothesis, truncation=True, padding=True, return_tensors='pt')
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            model_pred_id = np.argmax(probs, axis=1)[0]
            return model_to_mnli[model_pred_id], probs[0]

        return predict_one

    def test_model(self, dataset_split, mnli):
        print(f"  🔄 Testing model accuracy on {len(dataset_split)} examples...", flush=True)
        true_labels, predictions = [], []

        for example in tqdm(dataset_split, desc="  🧪 Accuracy test", unit="ex"):
            prediction, _ = mnli(example['premise'], example['hypothesis'])
            true_labels.append(example['label'])
            predictions.append(prediction)

        accuracy = accuracy_score(true_labels, predictions)
        print(f"  📊 Accuracy: {accuracy:.4f}", flush=True)
        return accuracy

    def safe_cleanup_memory(self):
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except RuntimeError as e:
            print(f"CUDA cleanup warning: {e}")
        gc.collect()

    def reset_cuda_context(self):
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.set_device(torch.cuda.current_device())
                torch.cuda.synchronize()
                return True
            except Exception as e:
                print(f"Failed to reset CUDA context: {e}")
                return False
        return False

    def get_models_accuracies(self, models, has_model_version_id=False, clear_file_errors=False):
        models_accuracies = {"model_version_id": [], "acc": [], "error": []}
        acc_output_csv_path = Path(self.acc_csv_output_path)

        if acc_output_csv_path.exists():
            existing_df = pd.read_csv(acc_output_csv_path)
            if clear_file_errors:
                existing_df = existing_df[existing_df["error"] == "no error"]
                existing_df.to_csv(acc_output_csv_path, index=False)
            all_evaluated_models = list(existing_df["model_version_id"])
            original_count = len(models)
            if isinstance(models[0], Path):
                models = [m for m in models if m.name not in all_evaluated_models]
            else:
                models = [m for m in models if m not in all_evaluated_models]
            if original_count > len(models):
                print(f"  ✓ Accuracy already tested for this model", flush=True)

        for model_path in models:
            model_name = model_path.name if Path(model_path).is_dir() else model_path
            models_accuracies["model_version_id"].append(model_name)

            try:
                print(f"  🔬 Starting accuracy test for {model_name}...", flush=True)
                mnli = self.create_predict_function(str(model_path), has_model_version_id)
                acc = self.test_model(self.mnli_val, mnli)
                models_accuracies["acc"].append(acc)
                models_accuracies["error"].append("no error")
            except Exception as e:
                print(f"Error on {str(model_path)}, skipping!")
                models_accuracies["acc"].append(None)
                models_accuracies["error"].append(str(e))

            current_results = {
                "model_version_id": models_accuracies["model_version_id"][-1:],
                "acc": models_accuracies["acc"][-1:],
                "error": models_accuracies["error"][-1:],
            }

            if not acc_output_csv_path.exists():
                pd.DataFrame(current_results).to_csv(str(acc_output_csv_path), index=False)
            else:
                existing_df = pd.read_csv(acc_output_csv_path)
                combined_df = pd.concat([existing_df, pd.DataFrame(current_results)], ignore_index=True)
                combined_df.to_csv(str(acc_output_csv_path), index=False)

            self.safe_cleanup_memory()

        return models_accuracies

    def get_models_from_folder(self, folder_path):
        return [Path(a) for a in Path(folder_path).glob('*_mnli')]
