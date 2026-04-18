import os
import sys
import argparse
from qlatent.hf_model_evaluator.pipeline.pipeline_executor import InitiatePipeline
from qlatent.hf_model_evaluator.global_variables import *
from huggingface_hub import HfApi
from qpsychometric import *


def evaluate_single_model(model_id, base_dir):
    try:
        hf_api = HfApi()

        try:
            model_info = hf_api.model_info(model_id)
        except Exception as e:
            print(f"✗ Error fetching model info for {model_id}: {e}")
            return 1

        if model_info.pipeline_tag != "fill-mask":
            print(f"✗ Skipping {model_id}: unsupported pipeline tag ({model_info.pipeline_tag})")
            return 1

        all_qmlm_questionnaires = all_psychometrics['QMLM'].get_questions()
        all_qmnli_questionnaires = all_psychometrics['QMNLI'][['ASI', 'BIG5', 'CS', 'GAD7', 'PHQ9', 'SOC']].get_questions()
        all_questionnaires = all_qmlm_questionnaires + all_qmnli_questionnaires

        InitiatePipeline(
            models_info=[model_info],
            questionnaires_questions_lists=all_questionnaires,
            base_dir=base_dir,
            override_results=False
        )

        print(f"✓ Completed evaluation for {model_id}")
        return 0

    except Exception as e:
        print(f"✗ Error evaluating {model_id}: {str(e)}")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a single fill-mask model on psychometric questionnaires.")
    parser.add_argument('--model_id', required=True, help='HuggingFace model ID to evaluate')
    parser.add_argument('--base_dir', default='./model_logs', help='Directory to save results (default: ./model_logs)')
    args = parser.parse_args()

    sys.exit(evaluate_single_model(args.model_id, args.base_dir))
