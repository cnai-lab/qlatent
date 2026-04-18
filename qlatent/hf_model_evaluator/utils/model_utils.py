import re
import gc
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

from .file_utils import FileUtils
from .logging_utils import LoggingUtils
from qlatent.hf_model_evaluator.global_variables import *

logging_utils = LoggingUtils()
file_utils = FileUtils()


class ModelUtils:
    def __init__():
        pass

    @staticmethod
    def exist_in_error_logs(model_name, base_dir):
        if file_utils.model_exists_in_logs(Path(base_dir) / MODELS_ERROR_PATH, model_name, model_name_column="model_version_id"):
            logging_utils.log_warning(f"{model_name} exists in error logs, skipping...")
            return True

    @staticmethod
    def cleanup_cache(cache_dir):
        file_utils.safe_rmtree(cache_dir)

    @staticmethod
    def reset_cuda_context():
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                logging_utils.log_info("CUDA context reset completed")
        except Exception as e:
            logging_utils.log_error(f"Failed to reset CUDA context: {e}")
            raise RuntimeError(f"Cannot recover CUDA context: {e}")

    @staticmethod
    def cleanup_pipeline_safely(pipeline):
        try:
            if pipeline is None:
                return
            if hasattr(pipeline, 'model') and pipeline.model is not None:
                try:
                    pipeline.model.cpu()
                except Exception as e:
                    logging_utils.log_warning(f"Failed to move model to CPU: {e}")
            if hasattr(pipeline, 'model'):
                del pipeline.model
            if hasattr(pipeline, 'tokenizer'):
                del pipeline.tokenizer
            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
        except Exception as e:
            logging_utils.log_warning(f"Error during pipeline cleanup: {e}")
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            except Exception:
                pass

    @staticmethod
    def get_scale(number):
        if number < 1_000:
            return f"{number} (Less than 1 Thousand)"
        elif number < 1_000_000:
            return f"{number / 1_000:.2f}K (Thousands)"
        elif number < 1_000_000_000:
            return f"{number / 1_000_000:.2f}M (Millions)"
        elif number < 1_000_000_000_000:
            return f"{number / 1_000_000_000:.2f}B (Billions)"
        else:
            return f"{number / 1_000_000_000_000:.2f}T (Trillions)"

    @staticmethod
    def load_pipeline_safely(model_info, base_dir):
        model_name = model_info.id
        max_retries = 1

        for attempt in range(max_retries):
            try:
                device = 0 if torch.cuda.is_available() else -1

                if torch.cuda.is_available():
                    try:
                        torch.cuda.current_device()
                        torch.cuda.empty_cache()
                    except RuntimeError as cuda_error:
                        if "CUDA" in str(cuda_error):
                            logging_utils.log_warning(f"CUDA context error detected, attempting recovery: {cuda_error}")
                            ModelUtils.reset_cuda_context()

                print(f"  🔄 Loading model: {model_name}...", flush=True)

                if model_info.pipeline_tag == "text-classification":
                    model_info.original_pipeline_tag = "text-classification"
                    model_info.pipeline_tag = "zero-shot-classification"

                config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

                if model_info.pipeline_tag == "zero-shot-classification":
                    if not any(label.lower().startswith("entail") for label in config.label2id.keys()):
                        raise ValueError(f"Model '{model_name}' does not have an entailment label for NLI")

                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

                pipe = pipeline(
                    task=model_info.pipeline_tag,
                    tokenizer=tokenizer,
                    model=model_name,
                    device=device,
                    trust_remote_code=True,
                )

                print(f"  ✅ Model loaded successfully", flush=True)

                pipe.model_identifier = model_name
                pipe.trainable_params = sum(p.numel() for p in pipe.model.parameters() if p.requires_grad)
                pipe.vocab_size = ModelUtils.get_vocab_size(pipe, model_info, base_dir)
                return pipe

            except RuntimeError as e:
                if "CUDA" in str(e) and attempt < max_retries - 1:
                    logging_utils.log_warning(f"CUDA error on attempt {attempt + 1}, retrying: {e}")
                    ModelUtils.reset_cuda_context()
                    continue
                else:
                    file_utils.log_model_errors(model_info.model_version_id, f"CUDA ERROR: {str(e)}", base_dir)
                    logging_utils.log_error(f"CUDA failure for model {model_name}: {e}")
                    return None

            except Exception as e:
                if attempt < max_retries - 1:
                    logging_utils.log_warning(f"Error on attempt {attempt + 1}, retrying: {e}")
                    continue
                else:
                    file_utils.log_model_errors(model_info.model_version_id, str(e), base_dir)
                    file_utils.check_disk_quota_error(e)
                    logging_utils.log_error(f"Failed to initialize model {model_name}: {e}")
                    return None

        return None

    @staticmethod
    def get_vocab_size(pipe, model_info, base_dir):
        try:
            return pipe.model.config.vocab_size
        except Exception as e:
            file_utils.log_model_errors(model_info.model_version_id, str(e), base_dir)
            logging_utils.log_error(f"Failed to get vocab size for {model_info.model_version_id}: {e}")
            return None
