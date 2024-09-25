
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


######################ModelTrainer######################
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification,pipeline,\
DataCollatorForLanguageModeling, DataCollatorWithPadding, Trainer, TrainingArguments, AutoModel, EvalPrediction, AutoConfig
from datasets import load_dataset, Dataset, Features, load_metric, DatasetDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import numpy as np
import csv
from typing import Union
from transformers import TrainerCallback
import pandas as pd
import os
os.environ["WANDB_MODE"] = "disabled"
######################ModelTrainer######################







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
                

class DataLoader:
    def __init__(self, label2_id=None):
        self.label2_id = label2_id # {'entailment': 0, 'neutral': 1, 'contradiction': 2}

    def _print_dataset_status(self, dataset, num_samples_train, num_samples_validation, validate, mnli_dataset):
        def get_sample_count(dataset, mnli_dataset):
            if mnli_dataset:
                return len(dataset['premise'])
            else:
                return len(dataset)

        # Print status for training set
        if num_samples_train:
            print(f"Sampled {get_sample_count(dataset['train'], mnli_dataset)} training samples!")
        else:
            print(f"num_samples_train was not provided, used whole {get_sample_count(dataset['train'], mnli_dataset)} training samples!")

        # Print status for validation set (if applicable)
        if validate:
            if num_samples_validation:
                print(f"Sampled {get_sample_count(dataset['validation'], mnli_dataset)} validation samples!")
            else:
                print(f"num_samples_validation was not provided, used whole {get_sample_count(dataset['validation'], mnli_dataset)} validation samples!")

        
        
    def _read_csv(self, file_path, mnli_dataset):
        data = {'premise': [], 'hypothesis': [], 'label': []} if mnli_dataset else {'sentence': []}
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader, None) if mnli_dataset else None # Skip the header row if it exists
            for row in csv_reader:
                if any(cell == "" for cell in row):
                    raise ValueError("There is an empty row in the dataset!")
                if mnli_dataset:
                    data['premise'].append(row[0])
                    data['hypothesis'].append(row[1])
                    if row[2].isalpha():
                        data['label'].append(self.label2_id[row[2]])
                    else:
                        data['label'].append(int(row[2]))
                else:
                    data['sentence'].append(row[0])
        return data

    def _load_csv_data(self, dataset_path, mnli_dataset, num_samples_train=None, num_samples_validation=None, validate=False, val_dataset=None):
        train_data = self._read_csv(dataset_path, mnli_dataset)
        val_data = self._read_csv(val_dataset, mnli_dataset) if val_dataset else {'premise': [], 'hypothesis': [], 'label': []} if mnli_dataset else {'sentence': []}


        if validate:
            if val_dataset:
                val_set = {k: v[:num_samples_validation] if num_samples_validation else v for k, v in val_data.items()}
            else:
                val_set = {k: v[num_samples_train:num_samples_train + num_samples_validation] if num_samples_validation else [] for k, v in train_data.items()}
            train_set = {k: v[:num_samples_train] if num_samples_train else v for k, v in train_data.items()}
        else:
            val_set = {'premise': [], 'hypothesis': [], 'label': []} if mnli_dataset else {'sentence': []}
            train_set = {k: v[:num_samples_train] if num_samples_train else v for k, v in train_data.items()}


        if mnli_dataset:
            return train_set, val_set
        else:
            return train_set['sentence'], val_set['sentence']

    def _prepare_dict_dataset(self, dataset, mnli_dataset, num_samples_train=None, num_samples_validation=None, validate=False):
        """
        Prepares the dataset by slicing and constructing a Hugging Face DatasetDict for training and validation.

        Args:
            dataset (dict): The input dataset, either for MNLI (with 'premise', 'hypothesis', 'label') or text (with 'text').
            mnli_dataset (bool): Whether the dataset is an MNLI dataset.
            num_samples_train (int, optional): Number of samples for the training set. Defaults to None.
            num_samples_validation (int, optional): Number of samples for the validation set. Defaults to None.
            validate (bool, optional): Whether to include validation set. Defaults to False.

        Returns:
            DatasetDict: Hugging Face DatasetDict for training (and validation if `validate` is True).
        """
        if mnli_dataset:
            keys = ['premise', 'hypothesis', 'label']

        # Slicing the dataset if num_samples_train or num_samples_validation is provided
        if num_samples_train:
            dataset['train'] = {key: value[:num_samples_train] for key, value in dataset['train'].items()} if mnli_dataset else dataset['train'][:num_samples_train]

        if num_samples_validation and validate:
            dataset['validation'] = {key: value[:num_samples_validation] for key, value in dataset['validation'].items()} if mnli_dataset else dataset['validation'][:num_samples_validation]

        # Constructing the Hugging Face DatasetDict
        if validate and 'validation' in dataset.keys():
            train_dataset = Dataset.from_dict({key: dataset['train'][key] for key in keys}) if mnli_dataset else Dataset.from_dict({"text": dataset['train']})
            validation_dataset = Dataset.from_dict({key: dataset['validation'][key] for key in keys}) if mnli_dataset else Dataset.from_dict({"text": dataset['validation']})
            dataset = DatasetDict({"train": train_dataset, "validation": validation_dataset})
        else:
            train_dataset = Dataset.from_dict({key: dataset['train'][key] for key in keys}) if mnli_dataset else Dataset.from_dict({"text": dataset['train']})
            dataset = DatasetDict({"train": train_dataset})

        return dataset
    
        
    
class SaveCheckpointByEpochCallback(TrainerCallback):
    """
    Callback to save the model and tokenizer at the end of each epoch during training.

    This callback saves the model and tokenizer state to a specified directory after each training epoch,
    allowing for periodic checkpoints of the training process.

    """

    def __init__(self, output_dir: str, tokenizer, save_checkpoint : bool, epochs_to_save : list[int], head_to_save):
        """
        Initialize the SaveCheckpointByEpochCallback.

        Args:
            output_dir (str): The directory where the checkpoints will be saved.
            tokenizer: The tokenizer associated with the model being trained.
        """
        self.output_dir = output_dir  # Set the directory to save the checkpoints
        self.tokenizer = tokenizer  # Set the tokenizer to be saved with the model
        self.head_to_save=head_to_save
        self.save_checkpoint=save_checkpoint
        self.epochs_to_save = epochs_to_save
        
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """
        Save the model and tokenizer at the end of each epoch.

        This method is called automatically by the Trainer at the end of each epoch.
        It saves the model and tokenizer to a subdirectory named after the current epoch.

        Args:
            args: The training arguments.
            state: The current state of the Trainer.
            control: The current control object.
            model: The model being trained.
            **kwargs: Additional keyword arguments.
        """
        epoch = state.epoch  # Get the current epoch number
        
        if not self.output_dir:
            checkpoint_dir = f"AutoCheckpoint_{model.name_or_path}/checkpoint-epoch-{int(epoch)}"
        else:
            checkpoint_dir = f"{self.output_dir}/checkpoint-epoch-{int(epoch)}"
        
        if self.head_to_save:
            model=self.head_to_save
            
        if self.save_checkpoint:
            if not self.epochs_to_save or epoch in self.epochs_to_save:
                model.save_pretrained(checkpoint_dir)
                self.tokenizer.save_pretrained(checkpoint_dir)


class ModelTrainer:
        
    def __init__(self):
        pass
    
    def _set_nested_attribute(self, obj, attribute_string: str, value):
        """
        Set the value of a nested attribute in an object.

        This method sets the value of a nested attribute (e.g., "layer1.layer2.weight") in an object.

        Args:
            obj: The object containing the nested attribute.
            attribute_string (str): A string representing the nested attribute (e.g., "layer1.layer2.weight").
            value: The value to set for the specified nested attribute.
        """
        attrs = attribute_string.split('.')  # Split the attribute string into individual attributes
        current_obj = obj
        # Traverse the attribute hierarchy except for the last attribute
        for attr in attrs[:-1]:
            current_obj = getattr(current_obj, attr)  # Get the nested object
        setattr(current_obj, attrs[-1], value)  # Set the final attribute value

    def _get_nested_attribute(self, obj, attribute_string: str):
        """
        Get the value of a nested attribute from an object.

        This method retrieves the value of a nested attribute (e.g., "layer1.layer2.weight") from an object.

        Args:
            obj: The object containing the nested attribute.
            attribute_string (str): A string representing the nested attribute (e.g., "layer1.layer2.weight").

        Returns:
            The value of the specified nested attribute.
        """
        attributes = attribute_string.split(".")  # Split the attribute string into individual attributes
        layer_obj = obj
        # Traverse the attribute hierarchy
        for attribute_name in attributes:
            layer_obj = getattr(layer_obj, attribute_name)  # Get the nested object
        return layer_obj  # Return the final attribute value    

    


    def _build_training_args(self, validate, per_device_train_batch_size, num_train_epochs, learning_rate, logging_dir, output_dir, overwrite_output_dir, save_strategy, per_device_eval_batch_size=None, evaluation_strategy='no'):
        training_args_dict = {
            "per_device_train_batch_size": per_device_train_batch_size,
            "num_train_epochs": num_train_epochs,
            "learning_rate": learning_rate,
            "logging_dir": logging_dir,
            "output_dir": output_dir,
            "overwrite_output_dir": overwrite_output_dir,
            "save_strategy": save_strategy,
        }

        if validate:
            training_args_dict["per_device_eval_batch_size"] = per_device_eval_batch_size
            training_args_dict["evaluation_strategy"] = evaluation_strategy

        return TrainingArguments(**training_args_dict)


    
    def _build_trainer(self, validate, model, args, train_dataset, data_collator, compute_metrics, callbacks, mnli_dataset, eval_dataset=None, preprocess_logits_for_metrics=None):
        params_for_val = ['eval_dataset']
        params_to_remove_for_nli = ['preprocess_logits_for_metrics']
        trainer_args = {
            "model": model,
            "args": args,
            "train_dataset": train_dataset,
            "data_collator": data_collator,
            "compute_metrics": compute_metrics,
            "callbacks": callbacks,
            "eval_dataset": eval_dataset,
            "preprocess_logits_for_metrics": preprocess_logits_for_metrics
        }
        
        if not validate:
            trainer_args = {key: value for key, value in trainer_args.items() if key not in params_for_val}
        if mnli_dataset:
            trainer_args = {key: value for key, value in trainer_args.items() if key not in params_to_remove_for_nli}
        return Trainer(**trainer_args)


    
    def init_head(self, uninitialized_head : AutoModelForMaskedLM, initialized_head : AutoModelForMaskedLM, layers_to_init : list[str]):
        model_name = uninitialized_head.base_model.config._name_or_path   
        print(f"===================================Copying layers weights and biases to {model_name} model===========")
        # this is done to copy the whole layer and not just an attribute of it, for example, at first we get: "vocab_transform.weight", and I want to access the whole layer "vocab_transform"
        layers_to_init = list(set([".".join(layer.split(".")[:-1]) for layer in layers_to_init]))
        for init_layer_name in layers_to_init:
            if "." in init_layer_name: # if there are iterative nested attributes, for example: lm_head.decoder
                
                layer_obj = self._get_nested_attribute(initialized_head, init_layer_name) 
                self._set_nested_attribute(uninitialized_head, init_layer_name, layer_obj)
                
            else:           
                setattr(uninitialized_head, init_layer_name, getattr(initialized_head, init_layer_name))
            print(f"The {init_layer_name} layer was copied from the initialized head!")            
        print("===================================Done copying layers weights and biases===================================")
    
    
    
    
    def _preprocess_logits_for_metrics_mlm(self, logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)


    def _compute_metrics_mlm(self, eval_pred):
        predictions, labels = eval_pred
        #predictions = logits.argmax(-1)
        metric = load_metric("accuracy")

        predictions = predictions.reshape(-1)
        labels = labels.reshape(-1)
        # Convert predictions and labels to lists
        mask = labels != -100       
        labels = labels[mask]
        predictions = predictions[mask]

        return metric.compute(predictions=predictions, references=labels)
    
    
    def _compute_metrics_nli(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        metric = load_metric("accuracy")
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result
        
        
    def _freeze_base_model(self, model, freeze_base):
        for param in model.base_model.parameters():
            param.requires_grad = not freeze_base      
        
    
    def _get_min_sequence_length(self, tokenizer, dataset, mnli_dataset):
        model_max_tokens = tokenizer.model_max_length                                        
        def find_longest_sequence(dataset, tokenizer):
            max_length = 0
            for sample in dataset:
                if mnli_dataset:
                    inputs = tokenizer(sample['premise'], sample['hypothesis'], truncation=False)
                else:
                    inputs = tokenizer(sample['text'], truncation=False)
                seq_length = len(inputs['input_ids'])
                if seq_length > max_length:
                    max_length = seq_length
            return max_length

        train_max_length = find_longest_sequence(dataset['train'], tokenizer)
        if 'validation' in dataset.keys():
            validation_max_length = find_longest_sequence(dataset['validation'], tokenizer)
            longest_sequence = max(train_max_length, validation_max_length)
        else:
            longest_sequence = train_max_length
            
        training_model_max_tokens = min(model_max_tokens, longest_sequence)
        return training_model_max_tokens    
    
    
    def _train_mlm(self, model, tokenizer, dataset: Union[str, DatasetDict, dict], num_samples_train, num_samples_validation, val_dataset, validate, batch_size, num_epochs, learning_rate, save_checkpoint, checkpoint_path, epochs_to_save, head_to_save, freeze_base):
        
        self.data_loader = DataLoader()
        mnli_dataset=False
        
        def preprocess_function(dataset):
            return tokenizer(dataset['text'], truncation=True, padding=True, max_length=training_model_max_tokens)

        if val_dataset is not None and not validate:
            raise ValueError("If a validation dataset is provided, then validate must be True!")

        if isinstance(dataset, str):
            if not dataset.endswith(".csv"):
                raise ValueError("The dataset must be a path to a CSV file.")

            # Load training and validation sets
            #training_set, validation_set = self.load_mlm_csv_data(dataset, num_samples_train, num_samples_validation, validate, val_dataset)
            training_set, validation_set = self.data_loader._load_csv_data(dataset_path=dataset, mnli_dataset=False, num_samples_train=num_samples_train, num_samples_validation=num_samples_validation, validate=validate, val_dataset=val_dataset)

            if validate:
                train_dataset = Dataset.from_dict({"text": training_set})
                validation_dataset = Dataset.from_dict({"text": validation_set})
                dataset = DatasetDict({"train": train_dataset, "validation": validation_dataset})
            else:
                train_dataset = Dataset.from_dict({"text": training_set})
                dataset = DatasetDict({"train": train_dataset})
            
                
        elif isinstance(dataset, DatasetDict):
            if num_samples_train:
                dataset['train'] = dataset['train'].select(range(num_samples_train))
            
            if num_samples_validation and validate:
                dataset['validation'] = dataset['validation'].select(range(num_samples_validation))

        elif isinstance(dataset, dict):
            if  not (all(isinstance(item, str) for item in dataset['train']) and (all(isinstance(item, str) for item in dataset['validation']) if 'validation' in dataset.keys() else True)):
                raise ValueError("The data must be strings contained in a list!")
            
            dataset = self.data_loader._prepare_dict_dataset(dataset=dataset, mnli_dataset=mnli_dataset, num_samples_train=num_samples_train, num_samples_validation=num_samples_validation, validate=validate)
                
        else:
            raise TypeError("Unsupported dataset type. Please provide a path to a CSV file, a DatasetDict, or a list of data.")
         
        self.data_loader._print_dataset_status(dataset, num_samples_train, num_samples_validation, validate=validate, mnli_dataset=mnli_dataset)
                
            
        training_model_max_tokens = self._get_min_sequence_length(tokenizer=tokenizer, dataset=dataset, mnli_dataset=mnli_dataset)    
                    
                
        tokenized_dataset = dataset.map(preprocess_function, batched=True)

        train_sampled_dataset=tokenized_dataset["train"]

        validation_sampled_dataset=tokenized_dataset["validation"] if 'validation' in tokenized_dataset.keys() and validate else None
        
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

        if freeze_base:
            self._freeze_base_model(model=model, freeze_base=freeze_base)

        training_args = self._build_training_args(validate=validate, per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size, num_train_epochs=num_epochs, learning_rate=learning_rate, evaluation_strategy="epoch", logging_dir="./mlm_training/logs/logging_mlm", output_dir="./mlm_training/output", overwrite_output_dir = True, save_strategy="no")
        trainer = self._build_trainer(validate=validate, model=model, args=training_args, train_dataset=train_sampled_dataset, eval_dataset=validation_sampled_dataset, data_collator=data_collator,mnli_dataset=mnli_dataset, compute_metrics=self._compute_metrics_mlm,preprocess_logits_for_metrics=self._preprocess_logits_for_metrics_mlm, callbacks=[SaveCheckpointByEpochCallback(checkpoint_path, tokenizer, save_checkpoint, epochs_to_save, head_to_save=head_to_save)])

        # Train the model
        trainer.train()
        return model


    
    def _train_nli(self, model, tokenizer, dataset : Union[str, DatasetDict, dict], num_samples_train, num_samples_validation, val_dataset, validate, label2_id, batch_size, num_epochs, learning_rate,save_checkpoint , checkpoint_path, epochs_to_save, head_to_save, freeze_base):
                  
        self.data_loader = DataLoader(label2_id=label2_id)
        mnli_dataset=True      
        # Tokenize the combined dataset
        def preprocess_function(dataset):
            return tokenizer(dataset['premise'], dataset['hypothesis'], padding=True, truncation=True, max_length=training_model_max_tokens)  
            
        if val_dataset is not None and not validate:
            raise ValueError("If a validation dataset is provided then validate must be True!")
        
        if isinstance(dataset, str):                
            if not dataset.endswith(".csv"):
                raise ValueError("The dataset must be a path to a csv file.")
        
            training_set, validation_set = self.data_loader._load_csv_data(dataset_path=dataset, mnli_dataset=mnli_dataset, num_samples_train=num_samples_train, num_samples_validation=num_samples_validation, validate=validate, val_dataset=val_dataset)

            if validation_set and validate:
                train_dataset = Dataset.from_dict(training_set)
                validation_dataset = Dataset.from_dict(validation_set)
                dataset = DatasetDict({"train": train_dataset, "validation": validation_dataset})
            else:
                train_dataset = Dataset.from_dict(training_set)
                dataset = DatasetDict({"train": train_dataset})
                
                
        elif isinstance(dataset, DatasetDict):
            if num_samples_train:
                dataset['train'] = dataset['train'].select(range(num_samples_train))
            
            if num_samples_validation and validate:
                dataset['validation'] = dataset['validation'].select(range(num_samples_validation))
                                                           
        elif isinstance(dataset, dict):
            # Validate that the 'train' and 'validation' datasets are structured correctly for MNLI
            required_keys = ['premise', 'hypothesis', 'label']

            dataset['train']['label'] = [label2_id[label] for label in dataset['train']['label']]
            if 'validation' in dataset.keys():
                dataset['validation']['label'] = [label2_id[label] for label in dataset['validation']['label']]


            def validate_mnli_data(data):
                return all(isinstance(data[key], list) for key in required_keys) and \
                       all(isinstance(item, str) for item in data['premise']) and \
                       all(isinstance(item, str) for item in data['hypothesis']) and \
                       all(isinstance(item, int) for item in data['label'])

            if not validate_mnli_data(dataset['train']) or \
               ('validation' in dataset.keys() and not validate_mnli_data(dataset['validation']) and validate):
                raise ValueError("The data must be a dictionary with lists for 'premise', 'hypothesis', and 'label'!")

            dataset = self.data_loader._prepare_dict_dataset(dataset=dataset, mnli_dataset=mnli_dataset, num_samples_train=num_samples_train, num_samples_validation=num_samples_validation, validate=validate)

        else:
            raise TypeError("Unsupported dataset type. Please provide a path to a CSV file, a DatasetDict, or a list of data.")                                                           
                                                           
                                                           
        self.data_loader._print_dataset_status(dataset, num_samples_train, num_samples_validation, validate=validate, mnli_dataset=True)
      
        training_model_max_tokens = self._get_min_sequence_length(tokenizer=tokenizer, dataset=dataset, mnli_dataset=mnli_dataset)    

        tokenized_dataset = dataset.map(preprocess_function, batched=True)

        train_sampled_dataset=tokenized_dataset["train"]
        validation_sampled_dataset=tokenized_dataset["validation"] if 'validation' in tokenized_dataset.keys() and validate else None


        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        if freeze_base:
            self._freeze_base_model(model=model, freeze_base=freeze_base)        
        
        training_args = self._build_training_args(validate=validate, per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size, num_train_epochs=num_epochs, learning_rate=learning_rate, evaluation_strategy="epoch", logging_dir="./mlm_training/logs/logging_mlm", output_dir="./mlm_training/output", overwrite_output_dir = True, save_strategy="no")
        trainer = self._build_trainer(validate=validate, model=model, args=training_args, train_dataset=train_sampled_dataset, eval_dataset=validation_sampled_dataset, data_collator=data_collator, mnli_dataset=mnli_dataset, compute_metrics=self._compute_metrics_nli, callbacks=[SaveCheckpointByEpochCallback(checkpoint_path, tokenizer, save_checkpoint, epochs_to_save, head_to_save=head_to_save)])
        
    
        # Train the model
        trainer.train()
        return model
    
    
    def get_non_base_layers(self, model):
        
        all_layers = list(model.state_dict().keys())
        base_layers = list(model.base_model.state_dict().keys())
        head_layers=[]
        for layer in all_layers:
            if ".".join(layer.split(".")[1:]) not in base_layers: # when looping over the layers of the base model we want to remove the prefix of the layer which is the name of the model, hence the ".".join(layer.split(".")[1:])
                head_layers.append(layer)
                
        return head_layers
    
    
    def attach_head_to_model(self, head1, head2, model_identifier : str):       
        setattr(head1, model_identifier, getattr(head2 ,model_identifier))
    
        

    def train_head(self, model, tokenizer, dataset, nli_head=False, mlm_head=False, 
                   model_to_copy_weights_from=None, num_samples_train=None, num_samples_validation=None,
                   val_dataset=None,validate=True, label2_id={'entailment': 0, 'neutral': 1, 'contradiction': 2}, 
                   batch_size=16, num_epochs=10, learning_rate=2e-5, freeze_base = False, copy_weights=False, 
                   save_checkpoint=False, checkpoint_path=None, epochs_to_save=None, head_to_save=None):
        
        model_name = model.base_model.config._name_or_path
                
        if  (not nli_head and not mlm_head) or (nli_head and mlm_head): # if both false or both true
            raise ValueError("You must have one head (nli_head or mlm_head) set to True at a time.")
        
        if (validate and not val_dataset and not num_samples_train):
            raise TypeError("`num_samples_train` is required when using validation!")

        if copy_weights:
            
            if not model_to_copy_weights_from:
                raise ValueError("Please pass in a model (model_to_copy_weights_from=?) to load the initialized layers from!")
                
            
            get_initialized_layers = self.get_non_base_layers(model_to_copy_weights_from)
            get_uninitialized_layers = self.get_non_base_layers(model)
            if sorted(get_uninitialized_layers)!=sorted(get_initialized_layers):
                raise ValueError(f"Models architecture are not equal, make sure that {model_to_copy_weights_from.base_model.config._name_or_path} head layers are the same as {model_name}'s")
            self.init_head(model, model_to_copy_weights_from, get_uninitialized_layers)

        
        if nli_head:
            print(f"Detected {model_name} with an NLI head...")
            self._train_nli(model=model, tokenizer=tokenizer, dataset=dataset, num_samples_train=num_samples_train, num_samples_validation=num_samples_validation, val_dataset=val_dataset, validate=validate,label2_id=label2_id, batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate,save_checkpoint=save_checkpoint, checkpoint_path=checkpoint_path, epochs_to_save=epochs_to_save, head_to_save=head_to_save, freeze_base=freeze_base)
        
        elif mlm_head:
            print(f"Detected {model_name} with an MLM head...")
            self._train_mlm(model=model, tokenizer=tokenizer, dataset=dataset, num_samples_train=num_samples_train, num_samples_validation=num_samples_validation, val_dataset=val_dataset, validate=validate, batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate,save_checkpoint=save_checkpoint, checkpoint_path=checkpoint_path, epochs_to_save=epochs_to_save, head_to_save=head_to_save, freeze_base=freeze_base)
            

