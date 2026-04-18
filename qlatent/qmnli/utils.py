
######################BuildModelLabels######################
import torch
from transformers import pipeline
import pandas as pd
import numpy as np
from typing import Callable, List, Dict
import os
import warnings
import gc
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings('ignore')
from collections import Counter
######################BuildModelLabels######################



class BuildModelLabels:
    def __init__(self, model_name : str,
                 label_2_dataset_id : dict = {"entailment": 0, "neutral" : 1, "contradiction" : 2}):
        
        self.model_name=model_name.replace("/","_",1)
        self.label_2_dataset_id = label_2_dataset_id
        self.data_set_path = os.path.join(os.path.dirname(__file__), 'mnli_label_detection_dataset')
        self._build_predictions_dict()
    
    def _build_predictions_dict(self):
        self.predictions_dict = {key:[] for key in self.label_2_dataset_id}
        
    def _get_names(self, directory_path : str, ending : str) -> List[str]:
        """
            Return a list of the names of all files with a specific ending that are inside a given directory.
        """

        names_list = []
    
        for filename in os.listdir(directory_path):
            if filename.endswith(ending):
                names_list.append(filename[:-(len(ending)+1)]) # Dont include ending

        return names_list
    
    
    def _get_split_length(self,split_name : str) -> int: # works
        """
        Returns the number of rows of the specified split.
        """

        df = pd.read_csv(os.path.join(self.data_set_path, f"{split_name}.csv"), encoding = "utf-8-sig")
        row_count = len(df)
        return row_count
    
    
    def _load_k_rows(self, split_name : str, k : int,total_predictions) -> pd.DataFrame:
        """
        Returns a dataframe that contains k new rows of the split $split_name.
        """

        header_names = ['premise', 'hypothesis', 'genre', 'label']
        k_rows_df = pd.read_csv(os.path.join(self.data_set_path, f"{split_name}.csv"),
                    encoding = "utf-8-sig",
                    header = None,
                    names = header_names,
                    skiprows = 1 + total_predictions,
                    nrows=k)

        return k_rows_df # THE BATCH TO BE CLASSIFED

    
    
    def _predict_k_rows(self, split_name : str, predict_batch : Callable[[List[str]], List[int]], k : int, total_predictions) -> None:
        """
        Predicts the label of $k rows (premise hypothesis pairs)
        And increases the split_index and correct_predictions of the $model csv file.
        """

        rows_df = self._load_k_rows(split_name, k, total_predictions)
        premises, hypotheses, true_labels = [], [], []
        for row in rows_df.itertuples():
            premises.append(row.premise)
            hypotheses.append(row.hypothesis)
            true_labels.append(row.label)

        predicted_labels = predict_batch(premises, hypotheses)    
        self.predictions_dict[split_name]=self.predictions_dict[split_name]+predicted_labels
        
        #correct_predictions = sum([predicted_labels[i] == true_labels[i] for i in range(k)])
        total_predictions += k
        return total_predictions
                    

    
    def _predict_function(self):
        
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()
        mnli = pipeline("zero-shot-classification", device=torch.device(0 if torch.cuda.is_available() else 1), model=self.model_name.replace('_', '/',1))
        if hasattr(mnli.model.config, 'id2label'):
            print(f"{self.model_name} ORIGINAL CONFIG:\n {mnli.model.config.id2label}")
        else:
            print(f"{self.model_name} original config is unknown.")
        def predict_batch(premises: List[str], hypotheses: List[str]) -> List[int]:
            """
                Uses model given create_predict_function to predict a batch of premise&hypothesis pairs.
            """        
            # Initialize a list to store the predicted labels
            predicted_ids = []
            # Tokenize the batch of premises and hypotheses
            inputs = mnli.tokenizer(premises, hypotheses, truncation=True, max_length=1024, padding=True, return_tensors='pt')
            # Move inputs to CUDA if available
            model_inputs = {k: v.to('cuda') for k, v in inputs.items()}
            # Forward pass through the model
            with torch.no_grad():
                outputs = mnli.model(**model_inputs)
                # Calculate probabilities and predict labels
                probs = torch.softmax(outputs.logits, dim=1).cpu().detach().numpy()
                batch_predicted_ids = np.argmax(probs, axis=1).tolist()
                # Append batch predictions to the list of predicted labels
                predicted_ids.extend(batch_predicted_ids)

            return predicted_ids
        return predict_batch
    
    
    def _perform_predictions(self):
        splits_names = self._get_names(self.data_set_path,'csv')
        predict_function = self._predict_function()
        batch_size = 64
        for split_name in splits_names:
            total_predictions = 0
            split_length = self._get_split_length(split_name)
            k = min(split_length - total_predictions, batch_size) 
            while k > 0:
                total_predictions = self._predict_k_rows(split_name, predict_function, k, total_predictions)
                k = min(split_length - total_predictions, batch_size)
    
    def return_id2label(self):
        self._perform_predictions()
        splits_names = self._get_names(self.data_set_path,'csv')
        id2_label={}
        for split_name in splits_names:
            
            numbers = [int(x) for x in self.predictions_dict[split_name]]
            # Use Counter to count occurrences of each number
            counter = Counter(numbers)

            # Use max() function with key argument to find the most common number
            most_common_number = max(counter, key=counter.get)
            id2_label[most_common_number]=split_name
        print("============NEW MODEL CONFIG===========")
        print(id2_label)
        print("========================================")
        return id2_label


