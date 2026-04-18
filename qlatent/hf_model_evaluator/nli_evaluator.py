import os
from pathlib import Path

from qlatent.hf_model_evaluator.single_model_qmnli import evaluate_single_model
from qlatent.hf_model_evaluator.per_model_controller_nli import (
    get_all_nli_models,
    load_models_from_chunk_file,
    get_processed_models,
    process_single_model,
)


class NLIEvaluator:

    def __init__(self, base_dir='./results'):
        """
        Parameters
        ----------
        base_dir : str
            Directory where results and logs are saved.
            Relative paths are resolved to absolute at instantiation time,
            so subprocesses always receive an unambiguous path.
            Default: './results'
        """
        self.base_dir = str(Path(base_dir).resolve())
        Path(self.base_dir).mkdir(parents=True, exist_ok=True)

    def evaluate(self, model_id):
        """
        Evaluate a single NLI model on all QMNLI questionnaires.

        Parameters
        ----------
        model_id : str
            HuggingFace model ID (e.g. 'typeform/distilbert-base-uncased-mnli').
            Must have pipeline_tag 'zero-shot-classification' or 'text-classification'.

        Returns
        -------
        int
            0 on success, 1 on failure.
        """
        return evaluate_single_model(model_id, self.base_dir)

    def run(self, chunk_file=None, only_zero_shot=False, hf_token=None):
        """
        Run the NLI evaluation pipeline over multiple models.

        Parameters
        ----------
        chunk_file : str, optional
            Path to a plain-text file with one HuggingFace model ID per line.
            If omitted, all NLI models are fetched from HuggingFace automatically.
        only_zero_shot : bool
            If True, restrict to zero-shot-classification models only.
            Ignored when chunk_file is provided. Default: False.
        hf_token : str, optional
            HuggingFace API token. Reduces rate-limiting when fetching models.
            If omitted, uses the HF_TOKEN environment variable if set.
        """
        if hf_token:
            os.environ['HF_TOKEN'] = hf_token

        processed = get_processed_models(self.base_dir)

        if chunk_file:
            model_ids = load_models_from_chunk_file(chunk_file)
            remaining = sorted(m for m in model_ids if m not in processed)
        else:
            all_models = get_all_nli_models(only_zero_shot=only_zero_shot)
            remaining = sorted(
                [m.id for m in all_models if m.id not in processed]
            )

        print(f"Total remaining: {len(remaining)} models")
        successful, failed = 0, 0
        for i, model_id in enumerate(remaining):
            print(f"\nProgress: {i+1}/{len(remaining)} | ✅ {successful} ❌ {failed}")
            if process_single_model(model_id, self.base_dir):
                successful += 1
            else:
                failed += 1

        print(f"\nFinal: ✅ {successful} successful, ❌ {failed} failed")