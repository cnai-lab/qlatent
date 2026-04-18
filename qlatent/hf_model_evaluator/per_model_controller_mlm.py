import os
import subprocess
import sys
import logging
import pandas as pd
from pathlib import Path
from huggingface_hub import HfApi


def get_all_fill_mask_models():
    hf_api = HfApi()
    fill_mask_models = list(hf_api.list_models(filter="fill-mask"))
    valid_models = [m for m in fill_mask_models if m.pipeline_tag == "fill-mask"]
    unique_models = {m.modelId: m for m in valid_models}.values()
    return list(unique_models)


def load_models_from_chunk_file(chunk_file_path):
    with open(chunk_file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def get_processed_models(base_dir):
    processed = set()

    asi_file = Path(base_dir) / "QMLM" / "ASI.csv"
    if asi_file.exists():
        try:
            df = pd.read_csv(asi_file, encoding='utf-8-sig')
            if 'model_version_id' in df.columns:
                model_names = {mvid.rsplit('_', 1)[0] for mvid in df['model_version_id'].dropna().unique()}
                processed.update(model_names)
                print(f"📋 Found {len(model_names)} processed models in ASI.csv")
        except Exception as e:
            logging.warning(f"Could not read ASI file: {e}")

    error_file = Path(base_dir) / "models_errors.csv"
    if error_file.exists():
        try:
            df = pd.read_csv(error_file, encoding='utf-8-sig')
            if 'model_version_id' in df.columns and 'error' in df.columns:
                error_model_names = set()
                for _, row in df.iterrows():
                    mvid = row['model_version_id']
                    error = str(row['error']) if pd.notna(row['error']) else ''
                    if pd.notna(mvid):
                        if (('Gated repo' in error and '403 Client Error' in error) or
                                'wrong pipeline tag' in error.lower() or
                                'Per-model subprocess: Subprocess error (code 1)' in error or
                                'Per-model MLM subprocess: Subprocess error (code 1)' in error):
                            error_model_names.add(mvid)
                        else:
                            error_model_names.add(mvid.rsplit('_', 1)[0])
                processed.update(error_model_names)
                print(f"⚠️  Found {len(error_model_names)} error models in models_errors.csv")
        except Exception as e:
            logging.warning(f"Could not read error file: {e}")

    return processed


def process_single_model(model_id, base_dir, timeout=3600):
    print(f"📊 Evaluating model: {model_id}")
    try:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).parent / "single_model_qmlm.py"), "--model_id", model_id, "--base_dir", base_dir],
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        output_lines = []
        model_had_error = False

        for line in iter(process.stdout.readline, ''):
            line = line.rstrip()
            if line:
                output_lines.append(line)
                if any(kw in line.lower() for kw in [
                    'evaluating', 'questionnaire', 'accuracy', 'error', 'failed',
                    'skipping', 'completed', 'loading', '🔄', '✅', '📊', '📝', '✓'
                ]):
                    print(f"  {line}")
                if 'error on' in line.lower() and 'skipping' in line.lower():
                    model_had_error = True

        return_code = process.wait(timeout=timeout)

        if return_code == 0:
            if model_had_error:
                print(f"⚠️  Skipped: {model_id} (model error)")
            else:
                print(f"✅ Completed: {model_id}")
            return True
        else:
            error_msg = '\n'.join(output_lines[-10:])
            print(f"❌ Failed: {model_id}\n   {error_msg}")
            log_failed_model(model_id, base_dir, f"Subprocess error (code {return_code}): {error_msg}")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout: {model_id} (after {timeout}s)")
        process.kill()
        log_failed_model(model_id, base_dir, f"Timeout after {timeout} seconds")
        return False

    except Exception as e:
        print(f"💥 Error processing {model_id}: {e}")
        log_failed_model(model_id, base_dir, f"Exception: {str(e)}")
        return False


def log_failed_model(model_id, base_dir, error_message):
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    csv_log_file = base_dir / "models_errors.csv"
    new_df = pd.DataFrame([{"model_version_id": model_id, "error": f"Per-model MLM subprocess: {error_message}"}])

    if csv_log_file.is_file():
        try:
            df = pd.read_csv(csv_log_file, encoding='utf-8-sig')
            df = pd.concat([df, new_df], ignore_index=True)
        except Exception:
            df = new_df
    else:
        df = new_df

    df.to_csv(csv_log_file, index=False, encoding='utf-8-sig')


