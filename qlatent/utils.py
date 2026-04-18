import os
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification,pipeline,\
DataCollatorForLanguageModeling, DataCollatorWithPadding, Trainer, TrainingArguments, AutoModel, EvalPrediction, AutoConfig
from datasets import load_dataset, Dataset, Features, load_metric, DatasetDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Union
from transformers import TrainerCallback
import pandas as pd
from huggingface_hub import HfApi
hf_api = HfApi()
from datasets import Dataset, DatasetDict
from datasets import load_dataset
from datasets import load_dataset, Dataset
import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, label2_id):
        self.label2_id = label2_id # {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    
    def _print_dataset_status(self, dataset, task_type):
        def get_sample_count(dataset, task_type):
            if task_type == 'nli':
                return len(dataset['premise'])
            else:
                return len(dataset)

        print(f"Sampled {get_sample_count(dataset['train'], task_type)} training samples!")
        # Print status for validation set (if applicable)
        if "validation" in dataset:
            print(f"Sampled {get_sample_count(dataset['validation'], task_type)} validation samples!")

    def _convert_labels_to_numeric(self, dataset, task_type):
        """Converts non-numeric labels to numeric labels using `self.label2_id`."""
        if task_type == 'nli' and self.label2_id:
            def convert_labels(sentence):
                sentence['label'] = self.label2_id[sentence['label']] if sentence['label'] in self.label2_id else sentence['label']
                return sentence
            dataset = dataset.map(convert_labels)
        return dataset


    def _load_csv_data(self, dataset_path, task_type, num_percentage_validation, val_dataset):
        # Prepare data files for loading
        data_files = {'train': dataset_path}
        if val_dataset:
            data_files['validation'] = val_dataset

        if task_type=="mlm":
            dataset = load_dataset('csv', data_files=data_files, encoding="utf-8-sig")
        # Load dataset with appropriate columns
        elif task_type=='nli':
            dataset = load_dataset('csv', data_files=data_files, encoding="utf-8-sig")
        if num_percentage_validation and not val_dataset:
            if not (0 < num_percentage_validation < 1):
                raise ValueError("num_percentage_validation must be between 0 and 1")
            

            dataset = dataset['train'].train_test_split(test_size=num_percentage_validation)
            dataset['validation'] = dataset['test']
            del dataset['test']
                
        if "validation" in dataset:
            dataset['validation'] = self._convert_labels_to_numeric(dataset['validation'], task_type)
        
        # Convert labels to numeric if needed
        dataset['train'] = self._convert_labels_to_numeric(dataset['train'], task_type)

        return dataset


    def _prepare_dict_dataset(self, data_dict, task_type, num_percentage_validation, val_dataset):
        """
        Load a dataset from a dictionary for NLI or MLM, with optional validation set splitting.
        
        Args:
            data_dict (dict): A dictionary containing the data. For NLI, it should have 'premise', 'hypothesis', and 'label'.
                              For MLM, it should have 'text'.
            task_type (str): Either 'nli' for Natural Language Inference or 'mlm' for Masked Language Modeling.
            num_percentage_validation (float): The percentage of data to use for validation if no validation set is provided.
            validate (bool): Whether to create a validation set if it doesn't exist in the data.
            val_dataset (dict): A validation dataset, if it exists separately.
        
        Returns:
            DatasetDict: A DatasetDict object containing 'train' and optionally 'validation' splits.
        """

        if 'train' not in data_dict:
            raise ValueError("The dictionary must contain a 'train' key with the data.")

        if task_type == 'nli':
            # Ensure the NLI format has 'premise', 'hypothesis', and 'label' keys
            if not all(key in data_dict['train'] for key in ['premise', 'hypothesis', 'label']):
                raise ValueError("For NLI, the dictionary must contain 'premise', 'hypothesis', and 'label' keys.")

            if "validation" in data_dict:
                train_dataset = Dataset.from_dict(data_dict['train'])
                validation_dataset = Dataset.from_dict(data_dict['validation'])
                dataset_dict = {'train': train_dataset, 'validation': validation_dataset}
            # Convert the train split into a Dataset object
            else:
                # Create Dataset objects for NLI
                train_dataset = Dataset.from_dict(data_dict['train'])
                dataset_dict = {'train': train_dataset}


            if num_percentage_validation and "validation" not in data_dict:
                if not 0 < num_percentage_validation < 1:
                    raise ValueError("num_percentage_validation should be between 0 and 1")
                dataset = train_dataset.train_test_split(test_size=num_percentage_validation)
                dataset_dict = {'train': dataset['train'], 'validation': dataset['test']}

        elif task_type == 'mlm':
            # Ensure the MLM format has 'text' key
            if not isinstance(data_dict['train'], list):
                raise ValueError("For MLM, the dictionary must contain a list of 'text' under the 'train' key.")

            if "validation" in data_dict:
                train_dataset = Dataset.from_dict({"text": data_dict['train']})
                validation_dataset = Dataset.from_dict({"text": data_dict['validation']})
                dataset_dict = {'train': train_dataset, 'validation': validation_dataset}
            else:
                train_dataset = Dataset.from_dict({"text": data_dict['train']})
                dataset_dict = {'train': train_dataset}
                

            if num_percentage_validation and not "validation" in data_dict:
                if not 0 < num_percentage_validation < 1:
                    raise ValueError("num_percentage_validation should be between 0 and 1")
                dataset = train_dataset.train_test_split(test_size=num_percentage_validation)
                dataset_dict = {'train': dataset['train'], 'validation': dataset['test']}

        else:
            raise ValueError(f"Unsupported task_type '{task_type}'. Supported types: 'nli', 'mlm'.")

        # Convert labels to numeric if needed
        dataset_dict['train'] = self._convert_labels_to_numeric(dataset_dict['train'], task_type)
        if 'validation' in dataset_dict:
            dataset_dict['validation'] = self._convert_labels_to_numeric(dataset_dict['validation'], task_type)

        return DatasetDict(dataset_dict)

 
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
 
    def fix_model_embedding_layer(self, nli_model):
        # Print the original type_vocab_size
        print("Old type_vocab_size:", nli_model.config.type_vocab_size)
        print("hello")
        nli_model.config.type_vocab_size = 2

        # Locate the token type embedding layer
        old_token_type_embeddings = nli_model.base_model.embeddings.token_type_embeddings
        hidden_size = old_token_type_embeddings.embedding_dim

        # Create a new embedding layer with 2 rows (for two segment types) and the same hidden size
        new_token_type_embeddings = nn.Embedding(2, hidden_size)

        # Initialize the new embeddings:
        # Copy the existing embedding (row 0) for both token type 0 and token type 1.
        with torch.no_grad():
            new_token_type_embeddings.weight.data[0] = old_token_type_embeddings.weight.data[0].clone()
            new_token_type_embeddings.weight.data[1] = old_token_type_embeddings.weight.data[0].clone()

        nli_model.base_model.embeddings.token_type_embeddings = new_token_type_embeddings

        print("New type_vocab_size:", nli_model.config.type_vocab_size)

 
 
    def _build_training_args(self, validate, per_device_train_batch_size, num_train_epochs, learning_rate, logging_dir, output_dir, overwrite_output_dir, save_strategy, per_device_eval_batch_size=None, evaluation_strategy='no'):
        training_args_dict = {
            "per_device_train_batch_size": per_device_train_batch_size,
            "num_train_epochs": num_train_epochs,
            "learning_rate": learning_rate,
            "logging_dir": logging_dir,
            "output_dir": output_dir,
            "overwrite_output_dir": overwrite_output_dir,
            "save_strategy": save_strategy,
            "logging_strategy":"epoch",
            "max_grad_norm": 1.0,  # Gradient Clipping
            "fp16": True  # Mixed Precision Training (FP16)
        }
 
        if validate:
            training_args_dict["per_device_eval_batch_size"] = per_device_eval_batch_size
            training_args_dict["evaluation_strategy"] = evaluation_strategy
 
        return TrainingArguments(**training_args_dict)
 
 
 
    def _build_trainer(self, model, args, train_dataset, data_collator, compute_metrics, callbacks, task_type, eval_dataset=None, preprocess_logits_for_metrics=None):
        trainer_args = {
            "model": model,
            "args": args,
            "train_dataset": train_dataset,
            "data_collator": data_collator,
            "compute_metrics": compute_metrics,
            "callbacks": callbacks,
        }
        if eval_dataset:
            trainer_args['eval_dataset']=eval_dataset
        if task_type=="mlm":
            trainer_args['preprocess_logits_for_metrics']=preprocess_logits_for_metrics
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
 
 
    def _get_min_sequence_length(self, tokenizer, dataset, task_type):
        model_max_tokens = tokenizer.model_max_length                                        
        def find_longest_sequence(dataset, tokenizer):
            max_length = 0
            for sample in dataset:
                if task_type=="nli":
                    inputs = tokenizer(sample['premise'], sample['hypothesis'], truncation=False)
                elif task_type=="mlm":
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
 
 
    def _train_mlm(self, model, tokenizer, dataset: Union[str, DatasetDict, dict], val_dataset,shuffle_dataset, batch_size,num_percentage_validation,num_samples_train, num_samples_validation, num_epochs, learning_rate, save_checkpoint, checkpoint_path, epochs_to_save, training_model_max_tokens, head_to_save, freeze_base):
        task_type="mlm"
        self.data_loader = DataLoader()
 
        def preprocess_function(dataset):
            return tokenizer(dataset['text'], truncation=True, padding="max_length", max_length=training_model_max_tokens)

 
        if isinstance(dataset, str):
            if not dataset.endswith(".csv"):
                raise ValueError("The dataset must be a path to a CSV file.")
            dataset = self.data_loader._load_csv_data(dataset_path=dataset, task_type="mlm", num_percentage_validation=num_percentage_validation, val_dataset=val_dataset)
 
        elif isinstance(dataset, DatasetDict):
            pass
        #     if val_dataset:
        #         raise ValueError("Include the validation set in the `DatasetDict`!")
 
        elif type(dataset) is dict:
            if val_dataset:
                raise ValueError("Include the validation set in the dictionary!")
            if  not (all(isinstance(item, str) for item in dataset['train']) and (all(isinstance(item, str) for item in dataset['validation']) if 'validation' in dataset.keys() else True)):
                raise ValueError("The data must be strings contained in a list!")
 
            dataset = self.data_loader._prepare_dict_dataset(data_dict=dataset, task_type=task_type, num_percentage_validation=num_percentage_validation, val_dataset=val_dataset)
 
        else:
            raise TypeError("Unsupported dataset type. Please provide a path to a CSV file, a DatasetDict, or a dictionary of data.")
 
        if shuffle_dataset:
            for key in dataset.keys():
                dataset[key] = dataset[key].shuffle(seed=42)
                
        if num_samples_train:
            dataset['train'] = dataset['train'].select(range(num_samples_train))

        if num_samples_validation and "validation" in dataset:
            dataset['validation'] = dataset['validation'].select(range(num_samples_validation)) 
 
        self.data_loader._print_dataset_status(dataset, task_type=task_type)
 
        if not training_model_max_tokens:
            training_model_max_tokens = self._get_min_sequence_length(tokenizer=tokenizer, dataset=dataset, task_type=task_type)    
 
 
        tokenized_dataset = dataset.map(preprocess_function, batched=True)
 
        train_sampled_dataset=tokenized_dataset["train"]
 
        validate = False
        validation_sampled_dataset=None
        if "validation" in tokenized_dataset:
            validation_sampled_dataset=tokenized_dataset["validation"]
            validate=True
 
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
 
        if freeze_base:
            self._freeze_base_model(model=model, freeze_base=freeze_base)
        
        training_args = self._build_training_args(validate=validate,per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size, num_train_epochs=num_epochs, learning_rate=learning_rate, evaluation_strategy="epoch", logging_dir="./mlm_training/logs/logging_mlm", output_dir="./mlm_training/output", overwrite_output_dir = True, save_strategy="no")
        trainer = self._build_trainer(model=model, args=training_args, train_dataset=train_sampled_dataset, eval_dataset=validation_sampled_dataset, data_collator=data_collator,task_type=task_type, compute_metrics=self._compute_metrics_mlm,preprocess_logits_for_metrics=self._preprocess_logits_for_metrics_mlm, callbacks=[SaveCheckpointByEpochCallback(checkpoint_path, tokenizer, save_checkpoint, epochs_to_save, head_to_save=head_to_save)])
 
        train_result = trainer.train()

        history = trainer.state.log_history
        checkpoint_path
        
        df = pd.DataFrame(history)
        df.to_csv(f"{checkpoint_path}/training_metrics_log.csv", index=False)
        
        return model
 
 
 
    def _train_nli(self, model, tokenizer, dataset : Union[str, DatasetDict, dict], val_dataset, label2_id, num_samples_train, num_samples_validation, num_percentage_validation, shuffle_dataset, batch_size, num_epochs, learning_rate,save_checkpoint , checkpoint_path, epochs_to_save, training_model_max_tokens, head_to_save, freeze_base, fix_model_embedding_layer):

        if fix_model_embedding_layer:        
            if hasattr(model.config, 'type_vocab_size') and model.config.type_vocab_size < 2:
                self.fix_model_embedding_layer(model)
            else:
                raise ValueError("model.config has no att `type_vocab_size` or model.config.type_vocab_size >= 2")
        
        task_type="nli"
        self.data_loader = DataLoader(label2_id=label2_id)
        # Tokenize the combined dataset
        def preprocess_function(dataset):
            return tokenizer(dataset['premise'], dataset['hypothesis'], padding="max_length", truncation=True, max_length=training_model_max_tokens)  
 
        if isinstance(dataset, str):
            if not dataset.endswith(".csv"):
                raise ValueError("The dataset must be a path to a CSV file.")
            dataset = self.data_loader._load_csv_data(dataset_path=dataset, task_type="nli", num_percentage_validation=num_percentage_validation, val_dataset=val_dataset)
 
        elif isinstance(dataset, DatasetDict):
            dataset=self.data_loader._convert_labels_to_numeric(dataset, "nli")
 
        elif type(dataset) is dict:
            if val_dataset:
                raise ValueError("Include the validation set in the dictionary!")
            if not (all(isinstance(item, str) for item in dataset['train']) and (all(isinstance(item, str) for item in dataset['validation']) if 'validation' in dataset.keys() else True)):
                raise ValueError("The data must be strings contained in a list!")
 
            dataset = self.data_loader._prepare_dict_dataset(data_dict=dataset, task_type=task_type, num_percentage_validation=num_percentage_validation, val_dataset=val_dataset)
 
        else:
            raise TypeError("Unsupported dataset type. Please provide a path to a CSV file, a DatasetDict, or a dictionary of data.")

        if shuffle_dataset:
            for key in dataset.keys():
                dataset[key] = dataset[key].shuffle(seed=42)
                
        if num_samples_train:
            dataset['train'] = dataset['train'].select(range(num_samples_train))

        if num_samples_validation and "validation" in dataset:
            dataset['validation'] = dataset['validation'].select(range(num_samples_validation))        
 
        self.data_loader._print_dataset_status(dataset, task_type=task_type)
 
        if not training_model_max_tokens:
            training_model_max_tokens = self._get_min_sequence_length(tokenizer=tokenizer, dataset=dataset, task_type=task_type)    
 
        print(f"training_model_max_tokens: {training_model_max_tokens}")
        tokenized_dataset = dataset.map(preprocess_function, batched=True)
 
        train_sampled_dataset=tokenized_dataset["train"]
        validate = False
        validation_sampled_dataset=None
        if "validation" in tokenized_dataset:
            validation_sampled_dataset=tokenized_dataset["validation"]
            validate = True
 
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        if freeze_base:
            self._freeze_base_model(model=model, freeze_base=freeze_base)        
 
        training_args = self._build_training_args(validate=validate,per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size, num_train_epochs=num_epochs, learning_rate=learning_rate, evaluation_strategy="epoch", logging_dir="./nli_training/logs/logging_nli", output_dir="./nli_training/output", overwrite_output_dir = True, save_strategy="no")
        trainer = self._build_trainer(model=model, args=training_args, train_dataset=train_sampled_dataset, eval_dataset=validation_sampled_dataset, data_collator=data_collator, task_type=task_type, compute_metrics=self._compute_metrics_nli, callbacks=[SaveCheckpointByEpochCallback(checkpoint_path, tokenizer, save_checkpoint, epochs_to_save, head_to_save=head_to_save)])
        
        train_result = trainer.train()

        history = trainer.state.log_history
        checkpoint_path
        
        df = pd.DataFrame(history)
        df.to_csv(f"{checkpoint_path}/training_metrics_log.csv", index=False)
        
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
 
 
 
    def train_head(self, model, tokenizer, dataset, label2_id=None, nli_head=False, mlm_head=False, 
                   model_to_copy_weights_from=None,num_samples_train=None, num_samples_validation=None, num_percentage_validation=None,
                   shuffle_dataset=False, val_dataset=None, batch_size=16, num_epochs=10, learning_rate=2e-5,training_model_max_tokens = None, freeze_base = False, copy_weights=False, 
                   save_checkpoint=False, checkpoint_path=None, epochs_to_save=None, head_to_save=None, fix_model_embedding_layer=False):
 
        model_name = model.base_model.config._name_or_path
        if (num_percentage_validation and val_dataset):
            raise ValueError("Cannot specify `num_percentage_validation` when validation set is available!")

        if (num_samples_validation and not val_dataset and not num_percentage_validation):
            raise ValueError("`val_dataset` or `num_percentage_validation` must be specified to use `num_samples_validation`!")
        
        if (nli_head and not label2_id):
            raise ValueError("`label2_id` must be specified for an NLI task!")

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
 
 
        try:
            last_commit = hf_api.list_repo_commits(repo_id=model_name)[0]
            last_commit_date = last_commit.created_at.strftime("%d-%m-%Y %H:%M:%S") + " UTC"
            last_commit_id = last_commit.commit_id
        except Exception as e:
            last_commit_date = "unknown"
            last_commit_id = "unknown"
 
 
        if head_to_save:
            head_to_save.config.last_commit_hash = last_commit_id
            head_to_save.config.last_commit_date = last_commit_date
            head_to_save.config.model_version_id = model_name + "_" + head_to_save.config.last_commit_hash
            
        else:
            model.config.last_commit_hash = last_commit_id
            model.config.last_commit_date = last_commit_date
            model.config.model_version_id = model_name + "_" + model.config.last_commit_hash       
                 
        if nli_head:
            print(f"Detected {model_name} with an NLI head...")
            self._train_nli(model=model, tokenizer=tokenizer, dataset=dataset, num_percentage_validation=num_percentage_validation, num_samples_train=num_samples_train, num_samples_validation=num_samples_validation, shuffle_dataset=shuffle_dataset,val_dataset=val_dataset,label2_id=label2_id, batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate,save_checkpoint=save_checkpoint, checkpoint_path=checkpoint_path, epochs_to_save=epochs_to_save, training_model_max_tokens=training_model_max_tokens, head_to_save=head_to_save, freeze_base=freeze_base, fix_model_embedding_layer=fix_model_embedding_layer)
 
        elif mlm_head:
            print(f"Detected {model_name} with an MLM head...")
            self._train_mlm(model=model, tokenizer=tokenizer, dataset=dataset, num_percentage_validation=num_percentage_validation, num_samples_train=num_samples_train, num_samples_validation=num_samples_validation, shuffle_dataset=shuffle_dataset,val_dataset=val_dataset, batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate,save_checkpoint=save_checkpoint, checkpoint_path=checkpoint_path, epochs_to_save=epochs_to_save, training_model_max_tokens=training_model_max_tokens, head_to_save=head_to_save, freeze_base=freeze_base)