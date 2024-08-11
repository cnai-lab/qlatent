
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
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification,\
DataCollatorForLanguageModeling, DataCollatorWithPadding, Trainer, TrainingArguments, AutoModel, EvalPrediction, AutoConfig
from datasets import load_dataset, Dataset, Features, load_metric, DatasetDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import random
import numpy as np
import csv
from typing import Union
from transformers import TrainerCallback
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
                



class SaveCheckpointByEpochCallback(TrainerCallback):
    """
    Callback to save the model and tokenizer at the end of each epoch during training.

    This callback saves the model and tokenizer state to a specified directory after each training epoch,
    allowing for periodic checkpoints of the training process.

    """

    def __init__(self, output_dir: str, tokenizer, head_to_save):
        """
        Initialize the SaveCheckpointByEpochCallback.

        Args:
            output_dir (str): The directory where the checkpoints will be saved.
            tokenizer: The tokenizer associated with the model being trained.
        """
        self.output_dir = output_dir  # Set the directory to save the checkpoints
        self.tokenizer = tokenizer  # Set the tokenizer to be saved with the model
        self.head_to_save=head_to_save
        
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
        checkpoint_dir = f"{self.output_dir}/checkpoint-epoch-{int(epoch)}"  # Define the checkpoint directory for the current epoch
        if self.head_to_save:
            model=self.head_to_save
        model.save_pretrained(checkpoint_dir)  # Save the model to the checkpoint directory
        self.tokenizer.save_pretrained(checkpoint_dir)  # Save the tokenizer to the checkpoint directory    


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

    
    def _train_mlm(self, model, tokenizer, dataset : Union[str, DatasetDict], num_samples_train, num_samples_validation, val_dataset, validate, batch_size, num_epochs, learning_rate, checkpoint_path, head_to_save, freeze_base, training_model_max_tokens):       

        # Tokenize the combined dataset
        def preprocess_function(dataset):
            return tokenizer(dataset['text'], truncation=True, padding=True, max_length=training_model_max_tokens)  
        
        if not isinstance(dataset, str) and not isinstance(dataset, DatasetDict):
            raise TypeError("dataset must be of type 'str' or 'Dataset'")
        
        if val_dataset is not None and not validate:
            raise ValueError("If a validation dataset is provided then validate must be True!")
        
        
        if isinstance(dataset, str):
            if dataset[-4:] != ".csv":
                raise ValueError("The dataset must be a path to a csv file.")
        
        
            sentences = []
            with open(dataset, newline='', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                for row in csv_reader:
                    if row[0]=="":
                        raise ValueError("There is an empty row at the dataset!")
                    # Assuming each row contains only one value
                    sentences.append(row[0])
                    
            #random.shuffle(sentences)
            if num_samples_train:
                training_set = sentences[:num_samples_train]
            else:
                training_set = sentences
                
            if val_dataset and validate:     
                validation_set=[]
                with open(val_dataset, newline='', encoding='utf-8') as csvfile:
                    csv_reader = csv.reader(csvfile)
                    for row in csv_reader:
                        if row[0]=="":
                            raise ValueError("There is an empty row at the dataset!")
                        # Assuming each row contains only one value
                        validation_set.append(row[0])
                            # Create Dataset objects for each split
                            
                train_dataset = Dataset.from_dict({"text": training_set})
                validation_dataset = Dataset.from_dict({"text": validation_set})

                dataset = DatasetDict({"train": train_dataset, "validation": validation_dataset})
            elif validate:
                # Split samples into training and validation sets
                if num_samples_train:
                    validation_set = sentences[num_samples_train:]
                
                else:
                    raise TypeError("Since num_samples_train is not provided, the validation dataset would include samples from training, so please specify num_samples_train")
                    
                # Create Dataset objects for each split
                train_dataset = Dataset.from_dict({"text": training_set})
                validation_dataset = Dataset.from_dict({"text": validation_set})

                dataset = DatasetDict({"train": train_dataset, "validation": validation_dataset})
                
            else:
                # Create Dataset objects for each split
                train_dataset = Dataset.from_dict({"text": training_set})
                dataset = DatasetDict({"train": train_dataset})
        
                      
        tokenized_dataset = dataset.map(preprocess_function, batched=True)

        if num_samples_train:
            print(f"Sampling {num_samples_train} training samples!")
            train_sampled_dataset = tokenized_dataset['train'].select(range(num_samples_train))
        else:
            print(f"num_samples_train was not provided, using whole {len(tokenized_dataset['train'])} training samples!")
            train_sampled_dataset = tokenized_dataset['train']
                    
        if num_samples_validation and validate:
            print(f"Sampling {num_samples_validation} validation samples!")
            validation_sampled_dataset = tokenized_dataset['validation'].select(range(num_samples_validation))
            
        elif validate:
            print(f"num_samples_validation was not provided, using whole {len(tokenized_dataset['validation'])} validation samples!")
            validation_sampled_dataset = tokenized_dataset['validation']
                
 
#         # Sample the indices of the items
#         train_sampled_indices = random.sample(range(len(tokenized_dataset['train'])), num_samples_train)
#         # Create a new dataset with the sampled items
#         train_sampled_dataset=tokenized_dataset['train'].select(train_sampled_indices)
        
          
#         validation_sampled_indices = random.sample(range(len(tokenized_dataset['validation'])), num_samples_validation)
#         validation_sampled_dataset=tokenized_dataset['validation'].select(validation_sampled_indices)



        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

        # Freeze/unfreeze base model
        for param in model.base_model.parameters():
            param.requires_grad = not freeze_base

        if validate:
            # Define training arguments
            training_args = TrainingArguments(
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                evaluation_strategy="epoch",  # Log metrics at the end of each epoch
                logging_dir="./mlm_training/logs/logging_mlm",
                output_dir="./mlm_training/output", 
                overwrite_output_dir = True,
                save_strategy="no",
                #save_strategy="epoch",  # Save checkpoint at the end of each epoch
            )

            # Define Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_sampled_dataset,
                eval_dataset=validation_sampled_dataset,
                data_collator=data_collator,
                compute_metrics=self._compute_metrics_mlm,
                preprocess_logits_for_metrics=self._preprocess_logits_for_metrics_mlm,
                callbacks=[SaveCheckpointByEpochCallback(checkpoint_path, tokenizer,head_to_save=head_to_save)],
            )
        else:
            # Define training arguments
            training_args = TrainingArguments(
                per_device_train_batch_size=batch_size,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                logging_dir="./mlm_training/logs/logging_mlm",  
                output_dir="./mlm_training/output", 
                overwrite_output_dir = True,
                save_strategy="no",
                #save_strategy="epoch",  # Save checkpoint at the end of each epoch
            )

            # Define Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_sampled_dataset,
                data_collator=data_collator,
                compute_metrics=self._compute_metrics_mlm,
                preprocess_logits_for_metrics=self._preprocess_logits_for_metrics_mlm,
                callbacks=[SaveCheckpointByEpochCallback(checkpoint_path, tokenizer,head_to_save=head_to_save)],

            )

        # Train the model
        trainer.train()
        return model

    
    
    def _train_nli(self, model, tokenizer, dataset : Union[str, DatasetDict], num_samples_train, num_samples_validation, val_dataset, validate, batch_size, num_epochs, learning_rate, checkpoint_path, head_to_save, freeze_base, training_model_max_tokens):
                  
                  
        # Tokenize the combined dataset
        def preprocess_function(dataset):
            return tokenizer(dataset['premise'], dataset['hypothesis'], padding=True, truncation=True, max_length=training_model_max_tokens)  
        
        
        if not isinstance(dataset, str) and not isinstance(dataset, DatasetDict):
            raise TypeError("dataset must be of type 'str' or 'Dataset'")
            
        if val_dataset is not None and not validate:
            raise ValueError("If a validation dataset is provided then validate must be True!")
        
        if isinstance(dataset, str):                
            if dataset[-4:] != ".csv":
                raise ValueError("The dataset must be a path to a csv file.")
        
            training_premise=[]
            training_hypothesis=[]
            training_label=[]
            label2_id = {'entailment': 0, 'neutral': 1, 'contradiction':2}
            with open(dataset, newline='', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip the header row
                for row in csv_reader:
                    training_premise.append(row[0])
                    training_hypothesis.append(row[1])
                    training_label.append(label2_id[row[2]])
                    
            if num_samples_train:
                training_premise=training_premise[:num_samples_train]
                training_hypothesis=training_hypothesis[:num_samples_train]
                training_label=training_label[:num_samples_train]
                    
            if val_dataset and validate:     
                validation_premise=[]
                validation_hypothesis=[]
                validation_label=[]
                with open(val_dataset, newline='', encoding='utf-8') as csvfile:
                    csv_reader = csv.reader(csvfile)
                    next(csv_reader)  # Skip the header row
                    for row in csv_reader:
                        validation_premise.append(row[0])
                        validation_hypothesis.append(row[1])
                        validation_label.append(label2_id[row[2]])

                # Create a dictionary with the data
                training_set = {
                    'premise': training_premise,
                    'hypothesis': training_hypothesis,
                    'label': training_label
                }
                
                # Create a dictionary with the data
                validation_set = {
                    'premise': validation_premise,
                    'hypothesis': validation_hypothesis,
                    'label': validation_label
                }

                # Create Dataset objects for each split
                train_dataset = Dataset.from_dict(training_set)
                validation_dataset = Dataset.from_dict(validation_set)

                dataset = DatasetDict({"train": train_dataset, "validation_matched": validation_dataset})
                
                
                
            elif validate:  
                if num_samples_train:                             
                    validation_premise = training_premise[num_samples_train:]
                    validation_hypothesis = training_hypothesis[num_samples_train:]
                    validation_label = training_label[num_samples_train:]
                
                else:
                    raise TypeError("Since num_samples_train is not provided, the validation dataset would include samples from training, so please specify num_samples_train")
                # Create a dictionary with the data
                training_set = {
                    'premise': training_premise,
                    'hypothesis': training_hypothesis,
                    'label': training_label
                }
                
                # Create a dictionary with the data
                validation_set = {
                    'premise': validation_premise,
                    'hypothesis': validation_hypothesis,
                    'label': validation_label
                }

                # Create Dataset objects for each split
                train_dataset = Dataset.from_dict(training_set)
                validation_dataset = Dataset.from_dict(validation_set)

                dataset = DatasetDict({"train": train_dataset, "validation_matched": validation_dataset})
                
            else:
                training_set = {
                    'premise': training_premise,
                    'hypothesis': training_hypothesis,
                    'label': training_label
                }
                # Create Dataset objects for each split
                train_dataset = Dataset.from_dict({"features": training_set})
                dataset = DatasetDict({"train": train_dataset})
        

            
        tokenized_dataset = dataset.map(preprocess_function, batched=True)
        if num_samples_train:
            print(f"Sampling {num_samples_train} training samples!")
            train_sampled_dataset = tokenized_dataset['train'].select(range(num_samples_train))
        else:
            print(f"num_samples_train was not provided, using whole {len(tokenized_dataset['train'])} training samples!")
            train_sampled_dataset = tokenized_dataset['train']

        if num_samples_validation and validate:
            print(f"Sampling {num_samples_validation} validation samples!")
            validation_sampled_dataset = tokenized_dataset['validation_matched'].select(range(num_samples_validation))

        elif validate:
            print(f"num_samples_validation was not provided, using whole {len(tokenized_dataset['validation_matched'])} validation samples!")
            validation_sampled_dataset = tokenized_dataset['validation_matched']            

        
#         train_random_indices = random.sample(range(len(tokenized_dataset['train'])), num_samples_train)
#         train_sampled_dataset = tokenized_dataset['train'].select(train_random_indices)        
        
#         validation_random_indices = random.sample(range(len(tokenized_dataset['validation_matched'])), num_samples_validation)
#         validation_sampled_dataset = tokenized_dataset['validation_matched'].select(validation_random_indices)
        
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
        # Freeze/unfreeze base model
        for param in model.base_model.parameters():
            param.requires_grad = not freeze_base
        
        
        if validate:
            # Define training arguments
            training_args = TrainingArguments(
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                evaluation_strategy="epoch",  # Log metrics at the end of each epoch
                logging_dir="./nli_training/logs/logging_nli",  
                output_dir="./nli_training/output_benevolent/",
                overwrite_output_dir = True,
                save_strategy="no",

            )

            # Define Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_sampled_dataset,
                eval_dataset=validation_sampled_dataset,
                data_collator=data_collator,
                compute_metrics=self._compute_metrics_mlm,
                preprocess_logits_for_metrics=self._preprocess_logits_for_metrics_mlm,
                callbacks=[SaveCheckpointByEpochCallback(checkpoint_path, tokenizer,head_to_save=head_to_save)],
            )
        else:
            # Define training arguments
            training_args = TrainingArguments(
                per_device_train_batch_size=batch_size,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                logging_dir="./nli_training/logs/logging_nli",  
                output_dir="./nli_training/output",
                overwrite_output_dir = True,
                save_strategy="no",

            )
            

            # Define Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_sampled_dataset,
                data_collator=data_collator,
                compute_metrics=self._compute_metrics_mlm,
                preprocess_logits_for_metrics=self._preprocess_logits_for_metrics_mlm,
                callbacks=[SaveCheckpointByEpochCallback(checkpoint_path, tokenizer,head_to_save=head_to_save)],
            )
    
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
    
        

    def train_head(self, model, tokenizer, dataset, nli_head=False, mlm_head=False, model_to_copy_weights_from=None, num_samples_train=None, num_samples_validation=None,val_dataset=None,validate=True,training_model_max_tokens=512, batch_size=16, num_epochs=10, learning_rate=2e-5, freeze_base = False, copy_weights=False, checkpoint_path=None, head_to_save=None):
        model_name = model.base_model.config._name_or_path         
        
        if  (not nli_head and not mlm_head) or (nli_head and mlm_head): # if both false or both true
            raise ValueError("You must have one head (nli_head or mlm_head) set to True at a time.")
            

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
            if not checkpoint_path:
                checkpoint_path = "./nli_training_checkpoint"
            self._train_nli(model=model, tokenizer=tokenizer, dataset=dataset, num_samples_train=num_samples_train, num_samples_validation=num_samples_validation, val_dataset=val_dataset, validate=validate, batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate, checkpoint_path=checkpoint_path, head_to_save=head_to_save, freeze_base=freeze_base, training_model_max_tokens=training_model_max_tokens)
        elif mlm_head:
            print(f"Detected {model_name} with an MLM head...")
            if not checkpoint_path:
                checkpoint_path = "./mlm_training_checkpoint"
            self._train_mlm(model=model, tokenizer=tokenizer, dataset=dataset, num_samples_train=num_samples_train, num_samples_validation=num_samples_validation, val_dataset=val_dataset, validate=validate, batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate, checkpoint_path=checkpoint_path, head_to_save=head_to_save, freeze_base=freeze_base, training_model_max_tokens=training_model_max_tokens)



