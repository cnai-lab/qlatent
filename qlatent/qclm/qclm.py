import numpy as np
import pandas as pd
import torch
import seaborn as sns
import scipy
import time
import sklearn as sk
from pprint import pprint
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import pipeline
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer
from abc import ABC, abstractmethod
import copy
from overrides import overrides
from typing import *
from numbers import Number
import itertools
from typeguard import check_type
from functools import partial
import pingouin as pg
import contextlib
import string

from qlatent.qabstract.qabstract_torch import *
from qlatent.qabstract.qabstract_torch import SCALE, DIMENSIONS, FILTER, IDXSELECT, _filter_data_frame

from typing import Dict

VALID_SCORING_METHODS = ('prod', 'geometric_mean', 'harmonic_mean')


def _score_token_probs(probs: torch.Tensor, method: str) -> torch.Tensor:
    """
    Combine per-token probabilities into a single score.

    Args:
        probs: Tensor of per-token probabilities.
        method: One of 'prod', 'geometric_mean', 'harmonic_mean'.

    Returns:
        Single-element tensor with the combined score.
    """
    if method == 'prod':
        return torch.prod(probs)

    elif method == 'geometric_mean':
        # exp(mean(log(probs))) -- length-normalized, numerically stable
        return torch.exp(torch.mean(torch.log(probs)))

    elif method == 'harmonic_mean':
        # N / sum(1/p_i) -- penalizes low-probability tokens heavily
        if torch.any(probs <= 0):
            raise ValueError("All probabilities must be positive for harmonic mean")
        return probs.numel() / torch.sum(1.0 / probs)

    else:
        raise ValueError(f"Unknown scoring method '{method}'. Must be one of {VALID_SCORING_METHODS}")


def _parse_template_last_placeholder(template: str, dimensions: dict) -> Tuple[str, str]:
    """
    Find the textually-last placeholder in `template` and validate that all
    placeholders correspond to keys in `dimensions`.

    Returns:
        (last_dim_name, template_prefix)
        - last_dim_name: name of the dimension whose placeholder is last.
        - template_prefix: the substring of `template` up to (but not including)
                           the opening brace of the last placeholder. Still
                           contains other placeholders to be format_map'd later.

    Raises:
        ValueError if the template has no placeholders, or any placeholder is
        not a declared dimension.
    """
    formatter = string.Formatter()
    placeholders = []  # list of (start_index, field_name)

    cursor = 0
    for literal_text, field_name, format_spec, conversion in formatter.parse(template):
        cursor += len(literal_text)
        if field_name is not None:
            # The placeholder starts at the current cursor in the original template.
            placeholders.append((cursor, field_name))
            # Advance cursor past the full placeholder "{field_name...}".
            # We find the matching closing brace from cursor in the template.
            close = template.index('}', cursor)
            cursor = close + 1

    if not placeholders:
        raise ValueError(f"Template has no placeholders: {template!r}")

    # Validate every placeholder is a declared dimension.
    for _, name in placeholders:
        if name not in dimensions:
            raise ValueError(
                f"Template placeholder {{{name}}} is not a declared dimension. "
                f"Declared dimensions: {list(dimensions.keys())}"
            )

    last_start, last_name = placeholders[-1]
    template_prefix = template[:last_start]
    return last_name, template_prefix


class QCLM(QABSTRACT):
    """
    Query-based Causal Language Model for measuring psychological constructs.

    Measures P(last-placeholder's tokens | preceding context) across all
    keyword combinations defined by `dimensions`.

    The "scale" measured is the dimension whose placeholder appears textually
    last in `template`. Any text after that placeholder is ignored.
    """
    def __init__(self,
                 template: str,
                 dimensions: DIMENSIONS = {},
                 model: pipeline = None,
                 probabilities=None,
                 index=None,
                 scale: str = 'intensifier',
                 descriptor: dict = {}):
        super().__init__(dimensions, model, probabilities, index, scale, descriptor=descriptor)

        self._descriptor['query'] = template
        self._template = template

        QCLM._qregister[self.__class__.__name__] = self

    def run(self, model=None, pre_text: str = None, training: bool = False,
            scoring: str = 'geometric_mean', debug: bool = False) -> 'QCLM':
        """
        Execute the evaluation or training pass across all keyword combinations.

        Args:
            model: Override model to use (optional).
            pre_text: Optional text to prepend to each formatted template
                      (followed by a newline). Becomes part of the prefix.
            training: If True, retains gradients and computation graph.
                      If False, uses inference_mode and detaches values.
            scoring: How to combine per-token probabilities.
                     'prod' / 'geometric_mean' / 'harmonic_mean'.
            debug: If True, print per-combination token info.
        """
        if scoring not in VALID_SCORING_METHODS:
            raise ValueError(f"scoring must be one of {VALID_SCORING_METHODS}, got '{scoring}'")

        super().run(model)
        start_time = time.time()

        # Parse the template once: find last placeholder and validate.
        last_dim_name, template_prefix = _parse_template_last_placeholder(
            self._template, self._dimensions
        )
        if debug:
            print(f"[DEBUG] last_dim_name = {last_dim_name!r}")
            print(f"[DEBUG] template_prefix = {template_prefix!r}")

        # Cache of tokenized prefixes within this run. Keyed by the rendered
        # prefix string; value is the prefix's token-id tensor on the model's
        # device.
        prefix_ids_cache: Dict[str, torch.Tensor] = {}

        # Storage for results.
        grid_indices = []
        probabilities_floats = []   # for DataFrame analysis (detached values)
        probabilities_tensors = []  # for autograd graph (training only)

        # Select context manager.
        ctx_manager = contextlib.nullcontext() if training else torch.inference_mode()

        with ctx_manager:
            for combo_idx, (keyword_map, keyword_grid_idx) in enumerate(
                zip(self._keywords_map, self._keywords_grid_idx)
            ):
                # 1. Build prefix_text by formatting the template-prefix.
                #    This fills in the non-scale dimension's value if it
                #    appears before the last placeholder.
                prefix_text = template_prefix.format_map(keyword_map)
                if pre_text is not None:
                    prefix_text = pre_text + "\n" + prefix_text

                # 2. Build full_text = prefix_text + scale_value.
                scale_value = keyword_map[last_dim_name]
                full_text = prefix_text + scale_value

                if debug:
                    print(f"\n[DEBUG] === Combination {combo_idx} ===")
                    print(f"[DEBUG] keyword_map = {keyword_map}")
                    print(f"[DEBUG] prefix_text = {prefix_text!r}")
                    print(f"[DEBUG] full_text   = {full_text!r}")

                # 3. Tokenize prefix (cached by prefix_text).
                if prefix_text in prefix_ids_cache:
                    prefix_ids = prefix_ids_cache[prefix_text]
                else:
                    prefix_ids = self.model.tokenizer(
                        text=[prefix_text], return_tensors="pt"
                    ).to(self.model.device)['input_ids'][0]
                    prefix_ids_cache[prefix_text] = prefix_ids

                # 4. Tokenize full_text.
                full_batch = self.model.tokenizer(
                    text=[full_text], return_tensors="pt"
                ).to(self.model.device)
                full_ids = full_batch['input_ids'][0]

                # 5. Compute answer span by finding where full_ids first
                #    diverges from prefix_ids. This is robust to byte-level BPE
                #    boundary merges (e.g. a trailing space in the prefix being
                #    absorbed into the scale word's leading-space token), which
                #    a plain len(prefix_ids) offset would get wrong.
                answer_start = 0
                min_len = min(len(prefix_ids), len(full_ids))
                while answer_start < min_len and \
                        prefix_ids[answer_start].item() == full_ids[answer_start].item():
                    answer_start += 1
                answer_len = len(full_ids) - answer_start

                if answer_len <= 0:
                    raise RuntimeError(
                        f"Combination {combo_idx}: scale value {scale_value!r} "
                        f"produced 0 tokens beyond the prefix. "
                        f"prefix_ids len={len(prefix_ids)}, full_ids len={len(full_ids)}, "
                        f"answer_start={answer_start}."
                    )

                if answer_start < 1:
                    # Need at least one preceding token to read a causal logit from.
                    raise RuntimeError(
                        f"Combination {combo_idx}: prefix is empty (no preceding "
                        f"context for a causal LM to condition on)."
                    )

                if debug:
                    full_tokens = self.model.tokenizer.convert_ids_to_tokens(full_ids)
                    span_tokens = full_tokens[answer_start:answer_start + answer_len]
                    print(f"[DEBUG] answer_start = {answer_start}, answer_len = {answer_len}")
                    print(f"[DEBUG] span tokens  = {span_tokens}")

                # 6. Forward pass.
                logits = self.model.model(**full_batch).logits
                probabilities_batch = torch.softmax(logits, dim=-1)

                # 7. Extract per-token probabilities.
                #    To predict token at position i, read logits at position i-1.
                answer_token_probs = []
                for i in range(answer_len):
                    token_position = answer_start + i
                    logit_position = token_position - 1
                    token_id = full_ids[token_position]

                    prob = probabilities_batch[0, logit_position, token_id]
                    answer_token_probs.append(prob)

                    if debug:
                        token_str = self.model.tokenizer.convert_ids_to_tokens(
                            [token_id.item()]
                        )[0]
                        print(f"  [DEBUG] tok {i}: {token_str!r} id={token_id.item()} "
                              f"prob={prob.item():.6f}")

                answer_token_probs = torch.stack(answer_token_probs)

                # 8. Combine into a single score.
                score = _score_token_probs(answer_token_probs, scoring)

                if debug:
                    print(f"[DEBUG] score ({scoring}) = {score.item():.8f}")

                # 9. Store.
                if training:
                    probabilities_tensors.append(score)
                    probabilities_floats.append(score.detach().cpu().item())
                else:
                    probabilities_floats.append(score.item())

                grid_indices.append(keyword_grid_idx)

                del answer_token_probs, logits, probabilities_batch, full_batch, full_ids

        # --- Finalize ---
        grid_indices = torch.stack(grid_indices).T
        assert torch.all(torch.eq(grid_indices.T, self._keywords_grid_idx)), \
            "Grid indices do not match expected keyword grid"

        self._pdf["P"] = probabilities_floats

        if training:
            self._t = torch.stack(probabilities_tensors)
        else:
            self._t = torch.tensor(probabilities_floats, device=self.model.device)

        del grid_indices
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._T = time.time() - start_time
        self.result = self
        return self.result


# Class-level registry for QCLM instances.
QCLM._qregister: Dict[str, QCLM] = {}