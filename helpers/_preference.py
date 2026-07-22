# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Mapping, Optional
import logging
import re

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset

from torchtune.data import ChosenRejectedToMessages, CROSS_ENTROPY_IGNORE_IDX
from torchtune.modules.transforms import Transform

from torchtune.modules.transforms.tokenizers import ModelTokenizer

log = logging.getLogger(__name__)

class PreferenceDataset(Dataset):
    """
    Primary class for fine-tuning via preference modelling techniques (e.g. training
    a preference model for RLHF, or directly optimizing a model through DPO) on a
    preference dataset sourced from Hugging Face Hub, local files, or remote files. This
    class requires the dataset to have "chosen" and "rejected" model responses. These are
    typically either full conversations between user and assistant in separate columns::

        |  chosen                                |  rejected                              |
        |----------------------------------------|----------------------------------------|
        | [{"role": "user", "content": Q1},      | [{"role": "user", "content": Q1},      |
        |  {"role": "assistant", "content": A1}] |  {"role": "assistant", "content": A2}] |

    or a user prompt column with separate chosen and rejected assistant reponses::

        |  prompt  |  chosen  |  rejected  |
        |----------|----------|------------|
        |  Q1      |  A1      |  A2        |


    In the above case when the format is prompt-chosen-rejected, only single-turn interactions are supported.

    At a high level, this class will load the data from source and apply the following pre-processing steps when a
    sample is retrieved:

    1. Dataset-specific transform. This is typically unique to each dataset and extracts
       the necessary prompt and chosen/rejected columns into torchtune's :class:`~torchtune.data.Message`
       format, a standardized API for all model tokenizers.
    2. Tokenization with optional prompt template if configured


    All datasets are formatted into a list of :class:`~torchtune.data.Message`
    because preference datasets can be considered as chosen and rejected "conversations"
    with the model, or AI assistant. Thus, we can standardize all text content as messages
    in a conversation assigned to a role:

    - ``"user"`` messages contain the input prompt into the model
    - ``"assistant"`` messages are the response of the model and what you actually want
      to train for and compute loss directly against

    The :class:`~torchtune.data.Message` forms the core data unit that all tokenizer
    APIs expect. The key component of this class that ensures any dataset is transformed
    into this format is the ``message_transform``. This is a callable class that takes
    in a sample dictionary - typically a single row from the source dataset - that
    processes the sample in any configurable way to output a list of messages::

        [
            Message(
                role=<system|user|assistant|ipython>,
                content=<message>,
            ),
            ...
        ]

    For any custom dataset, use the ``message_transform`` to contain all pre-processing to
    return the list of messages.

    Args:
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See `Hugging Face's
            <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path>`_
            ``load_dataset`` for more details.
        message_transform (Transform): callable that keys into the desired fields in the sample
            and converts text content to a list of :class:`~torchtune.data.Message`. It is expected that the final list
            of messages are stored in the ``"chosen"`` and ``"rejected"`` keys.
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
            Since PreferenceDataset only supports text data, it requires a
            :class:`~torchtune.modules.transforms.tokenizers.ModelTokenizer` instead of the ``model_transform`` in
            :class:`~torchtune.datasets.SFTDataset`.
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        packed (bool): Whether or not to pack the dataset to ``max_seq_len`` prior to training. Default is False. Packed is
            currently not supported for ``PreferenceDataset`` and a ``ValueError`` will be raised if this is set to True.
        **load_dataset_kwargs (dict[str, Any]): additional keyword arguments to pass to ``load_dataset``. See Hugging
            Face's `API ref <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset>`_
            for more details.

    Raises:
        ValueError: If ``packed`` is True, this feature is not supported for ``PreferenceDataset``.
    """

    def __init__(
        self,
        *,
        source: str,
        message_transform: Transform,
        tokenizer: ModelTokenizer,
        filter_fn: Optional[Callable] = None,
        packed: bool = False,
        num_attack_tokens: int = 10,
        attack_token_prefix: str = "<ATTACK_",
        attack_token_mode: str = "suffix",
        attack_tokens_per_sample: Optional[int] = None,
        **load_dataset_kwargs: dict[str, Any],
    ) -> None:
        if packed:
            raise ValueError(
                "Packed is currently not supported for preference datasets."
            )
        valid_attack_token_modes = {
            "suffix",
            "span_replacement",
            "random_injection_gaps",
            "random_structural_boundary_block",
        }
        if attack_token_mode not in valid_attack_token_modes:
            raise ValueError(
                f"Unsupported attack_token_mode={attack_token_mode!r}. "
                f"Expected one of {sorted(valid_attack_token_modes)}."
            )

        self._tokenizer = tokenizer
        self._message_transform = message_transform
        self._data = load_dataset(source, **load_dataset_kwargs)
        self._num_attack_tokens = num_attack_tokens
        self._attack_token_prefix = attack_token_prefix
        self._attack_token_mode = attack_token_mode
        self._attack_tokens_per_sample = attack_tokens_per_sample
        if (
            self._attack_tokens_per_sample is not None
            and self._attack_tokens_per_sample <= 0
        ):
            raise ValueError(
                "attack_tokens_per_sample must be positive when provided."
            )
        self._attack_tokens = [
            f"{attack_token_prefix}{idx}>" for idx in range(num_attack_tokens)
        ]
        self._attack_token_ids = [
            self._tokenizer.token_to_id(tok) for tok in self._attack_tokens
        ]
        if any(tok_id is None or tok_id < 0 for tok_id in self._attack_token_ids):
            raise RuntimeError(
                "Attack tokens must be registered on the tokenizer before dataset setup."
            )

        if filter_fn is not None:
            self._data = self._data.filter(filter_fn)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _find_rejected_span(
        self, prompt: str, rejected_input_whole: str
    ) -> tuple[int, int] | None:
        if rejected_input_whole in prompt:
            start = prompt.find(rejected_input_whole)
            return start, start + len(rejected_input_whole)

        # Some datasets store a whitespace-normalized rejected_input rather than the
        # exact prompt substring. Try a whitespace-tolerant match first.
        pattern = re.escape(rejected_input_whole.strip()).replace(r"\ ", r"\s+")
        match = re.search(pattern, prompt)
        if match is not None:
            return match.start(), match.end()

        return None

    def _build_suffix_attacked_prompt(
        self, prompt: str, rejected_input_whole: str
    ) -> tuple[str, list[int], list[int]]:
        span = self._find_rejected_span(prompt, rejected_input_whole)
        if span is not None:
            start, end = span
            span_token_len = len(self._tokenizer.encode(prompt[start:end]))
        else:
            span_token_len = len(self._tokenizer.encode(rejected_input_whole))

        if self._attack_tokens_per_sample is None:
            suffix_len = min(span_token_len, self._num_attack_tokens)
        else:
            suffix_len = min(self._attack_tokens_per_sample, self._num_attack_tokens)

        attack_suffix = " " + " ".join(self._attack_tokens[:suffix_len])
        attack_init_token_ids = [0] * self._num_attack_tokens
        attack_active_mask = [0] * self._num_attack_tokens
        for idx in range(suffix_len):
            attack_active_mask[idx] = 1

        if span is not None:
            _, end = span
            return (
                prompt[:end] + attack_suffix + prompt[end:],
                attack_init_token_ids,
                attack_active_mask,
            )

        # Final fallback: place the attack suffix immediately before the
        # assistant header, i.e. after the entire injected prompt content.
        assistant_header = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        assistant_idx = prompt.find(assistant_header)
        if assistant_idx == -1:
            raise ValueError(
                "Could not locate rejected_input span or assistant header in prompt, "
                "so the attack suffix cannot be inserted."
            )
        return (
            prompt[:assistant_idx] + attack_suffix + prompt[assistant_idx:],
            attack_init_token_ids,
            attack_active_mask,
        )

    def _build_span_replacement_prompt(
        self, prompt: str, rejected_input_whole: str
    ) -> tuple[list[int], list[int], list[int]]:
        span = self._find_rejected_span(prompt, rejected_input_whole)
        if span is None:
            raise ValueError(
                "attack_token_mode='span_replacement' requires locating the "
                "rejected_input span in the prompt."
            )

        start, end = span
        prefix_ids = self._tokenizer.encode(prompt[:start])
        span_ids = self._tokenizer.encode(prompt[start:end])
        suffix_ids = self._tokenizer.encode(prompt[end:])

        num_replaced = min(len(span_ids), self._num_attack_tokens)
        replacement_ids = (
            self._attack_token_ids[:num_replaced] + span_ids[num_replaced:]
        )
        prompt_tokenized = prefix_ids + replacement_ids + suffix_ids

        attack_init_token_ids = [0] * self._num_attack_tokens
        attack_active_mask = [0] * self._num_attack_tokens
        for idx in range(num_replaced):
            attack_init_token_ids[idx] = span_ids[idx]
            attack_active_mask[idx] = 1

        return prompt_tokenized, attack_init_token_ids, attack_active_mask

    def _build_random_injection_gap_prompt(
        self, prompt: str, rejected_input_whole: str
    ) -> tuple[list[int], list[int], list[int]]:
        span = self._find_rejected_span(prompt, rejected_input_whole)
        if span is None:
            raise ValueError(
                "attack_token_mode='random_injection_gaps' requires locating the "
                "rejected_input span in the prompt."
            )

        start, end = span
        prefix_ids = self._tokenizer.encode(prompt[:start])
        span_ids = self._tokenizer.encode(prompt[start:end])
        suffix_ids = self._tokenizer.encode(prompt[end:])
        span_token_len = len(span_ids)

        if self._attack_tokens_per_sample is None:
            num_inserted = min(span_token_len, self._num_attack_tokens)
        else:
            num_inserted = min(self._attack_tokens_per_sample, self._num_attack_tokens)
        num_inserted = min(num_inserted, span_token_len + 1)

        if num_inserted > 0:
            selected_gaps = set(
                int(gap)
                for gap in np.random.choice(
                    span_token_len + 1, size=num_inserted, replace=False
                )
            )
        else:
            selected_gaps = set()

        span_with_attacks = []
        next_attack_idx = 0
        for gap_idx in range(span_token_len + 1):
            if gap_idx in selected_gaps:
                span_with_attacks.append(self._attack_token_ids[next_attack_idx])
                next_attack_idx += 1
            if gap_idx < span_token_len:
                span_with_attacks.append(span_ids[gap_idx])

        prompt_tokenized = prefix_ids + span_with_attacks + suffix_ids

        attack_init_token_ids = [0] * self._num_attack_tokens
        attack_active_mask = [0] * self._num_attack_tokens
        for idx in range(num_inserted):
            attack_active_mask[idx] = 1

        return prompt_tokenized, attack_init_token_ids, attack_active_mask

    @staticmethod
    def _find_structural_boundary_offsets(text: str) -> list[int]:
        """Return character offsets that do not split lexical content."""
        offsets = {0, len(text)}

        for match in re.finditer(r"\n+", text):
            offsets.add(match.end())

        sentence_end = r'''[.!?]+["')\]]*(?=\s|$)'''
        for match in re.finditer(sentence_end, text):
            offsets.add(match.end())

        return sorted(offsets)

    def _build_random_structural_boundary_block_prompt(
        self, prompt: str, rejected_input_whole: str
    ) -> tuple[list[int], list[int], list[int]]:
        span = self._find_rejected_span(prompt, rejected_input_whole)
        if span is None:
            raise ValueError(
                "attack_token_mode='random_structural_boundary_block' requires "
                "locating the rejected_input span in the prompt."
            )

        start, end = span
        prefix_ids = self._tokenizer.encode(prompt[:start])
        rejected_span = prompt[start:end]
        span_ids = self._tokenizer.encode(rejected_span)
        suffix_ids = self._tokenizer.encode(prompt[end:])
        span_token_len = len(span_ids)

        if self._attack_tokens_per_sample is None:
            num_inserted = min(span_token_len, self._num_attack_tokens)
        else:
            num_inserted = min(
                self._attack_tokens_per_sample, self._num_attack_tokens
            )
        num_inserted = min(num_inserted, self._num_attack_tokens)

        if num_inserted > 0:
            boundary_offsets = self._find_structural_boundary_offsets(rejected_span)
            boundary_offset = int(np.random.choice(boundary_offsets))

            if boundary_offset == 0:
                token_gap = 0
            elif boundary_offset == len(rejected_span):
                token_gap = span_token_len
            else:
                token_gap = len(
                    self._tokenizer.encode(rejected_span[:boundary_offset])
                )
                token_gap = min(token_gap, span_token_len)

            attack_block = self._attack_token_ids[:num_inserted]
            span_with_attacks = (
                span_ids[:token_gap] + attack_block + span_ids[token_gap:]
            )
        else:
            span_with_attacks = span_ids

        prompt_tokenized = prefix_ids + span_with_attacks + suffix_ids

        attack_init_token_ids = [0] * self._num_attack_tokens
        attack_active_mask = [0] * self._num_attack_tokens
        for idx in range(num_inserted):
            attack_active_mask[idx] = 1

        return prompt_tokenized, attack_init_token_ids, attack_active_mask

    def _prepare_sample(self, sample: Mapping[str, Any]) -> dict[str, list[int]]:
        prompt = sample["prompt"]
        rejected_input_whole = sample.get("rejected_input_whole")
        if rejected_input_whole is None:
            rejected_input_whole = sample.get("rejected_input")

        if not isinstance(prompt, str):
            raise ValueError(f"Expected string prompt, got {type(prompt)!r}")
        if not isinstance(rejected_input_whole, str):
            raise ValueError(
                "Expected string rejected_input_whole/rejected_input, "
                f"got {type(rejected_input_whole)!r}"
            )

        attack_init_token_ids = None
        attack_active_mask = None
        if self._attack_token_mode == "suffix":
            (
                attacked_prompt,
                attack_init_token_ids,
                attack_active_mask,
            ) = self._build_suffix_attacked_prompt(
                prompt, rejected_input_whole
            )
            prompt_tokenized = self._tokenizer.encode(attacked_prompt)
        elif self._attack_token_mode == "span_replacement":
            (
                prompt_tokenized,
                attack_init_token_ids,
                attack_active_mask,
            ) = self._build_span_replacement_prompt(prompt, rejected_input_whole)
        elif self._attack_token_mode == "random_structural_boundary_block":
            (
                prompt_tokenized,
                attack_init_token_ids,
                attack_active_mask,
            ) = self._build_random_structural_boundary_block_prompt(
                prompt, rejected_input_whole
            )
        else:
            (
                prompt_tokenized,
                attack_init_token_ids,
                attack_active_mask,
            ) = self._build_random_injection_gap_prompt(prompt, rejected_input_whole)

        prompt_mask = [True] * len(prompt_tokenized)
        chosen_tokenized = self._tokenizer.encode(sample["chosen"])
        chosen_mask = [False] * (len(chosen_tokenized) - 1) + [True]
        rejected_tokenized = self._tokenizer.encode(sample["rejected"])
        rejected_mask = [False] * (len(rejected_tokenized) - 1) + [True]

        chosen_input_ids = prompt_tokenized + chosen_tokenized
        rejected_input_ids = prompt_tokenized + rejected_tokenized
        chosen_masks = prompt_mask + chosen_mask
        rejected_masks = prompt_mask + rejected_mask

        # TODO: Truncation differs from original DPO repo
        # in DPO: first truncate prompts, then responses
        chosen_labels = list(
            np.where(chosen_masks, CROSS_ENTROPY_IGNORE_IDX, chosen_input_ids)
        )

        rejected_labels = list(
            np.where(rejected_masks, CROSS_ENTROPY_IGNORE_IDX, rejected_input_ids)
        )

        assert len(chosen_input_ids) == len(chosen_labels)
        assert len(rejected_input_ids) == len(rejected_labels)

        tokenized_dict = dict(
            chosen_input_ids=chosen_input_ids,
            chosen_labels=chosen_labels,
            rejected_input_ids=rejected_input_ids,
            rejected_labels=rejected_labels,
        )
        if attack_init_token_ids is not None and attack_active_mask is not None:
            tokenized_dict["attack_init_token_ids"] = attack_init_token_ids
            tokenized_dict["attack_active_mask"] = attack_active_mask

        return tokenized_dict


def preference_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str,
    column_map: Optional[dict[str, str]] = None,
    train_on_input: bool = False,
    new_system_prompt: Optional[str] = None,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    num_attack_tokens: int = 10,
    attack_token_prefix: str = "<ATTACK_",
    attack_token_mode: str = "suffix",
    attack_tokens_per_sample: Optional[int] = None,
    **load_dataset_kwargs: dict[str, Any],
) -> PreferenceDataset:
    """
    Configures a custom preference dataset comprising interactions between user and
    model assistant.

    This builder function can be used to configure a custom preference dataset directly from the yaml config
    as an alternative to :class:`~torchtune.datasets.PreferenceDataset`, as it is made to be config friendly.

    This function requires the dataset to have "chosen" and "rejected" columns. A single sample will share an
    identical system +/ user prompt between both "chosen" and "rejected" columns, followed by one or multiple
    turns of user and assistant messages::

        |  chosen                                |  rejected                              |
        |----------------------------------------|----------------------------------------|
        | [{"role": "user", "content": Q1},      | [{"role": "user", "content": Q1},      |
        |  {"role": "assistant", "content": C1}] |  {"role": "assistant", "content": R1}] |


    This example will be converted to:

    .. code-block:: python

        chosen_messages = [
            Message(role="user", content="Q1"),
            Message(role="assistant", content="C1"),
        ]

        rejected_messages = [
            Message(role="user", content="Q1"),
            Message(role="assistant", content="R1"),
        ]


    These lists of messages are then tokenized for model training. Currently, this function only supports
    conversations identical to :class:`~torchtune.data.OpenAIToMessages`, and does not support custom
    message formats.

    If your dataset does not follow this format, we recommend creating a custom message transform similar to
    :class:`~torchtune.data.ChosenRejectedToMessages` and using it in a custom dataset builder function similar
    to :class:`~torchtune.datasets.preference_dataset`.

    Masking of the prompt during training is controlled by the ``train_on_input`` flag, which is:
    set to ``False`` by default.

    - If ``train_on_input`` is True, the prompt is used during training and
      contributes to the loss.
    - If ``train_on_input`` is False, the prompt is masked out (tokens replaced with -100).

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text"), pass
            in the filepath in ``data_files``, and set ``split="train"``. See `Hugging Face's
            <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path>`_
            ``load_dataset`` for more details.
        column_map (Optional[dict[str, str]]): a mapping from the expected columns "chosen" and "rejected"
            in the message transform :class:`~torchtune.data.ChosenRejectedToMessages` to the new column names in
            the dataset. Keys should be "chosen" and "rejected" and values should be the actual column names.
            If None, keep the default columns "chosen" and "rejected".
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        new_system_prompt (Optional[str]): if specified, prepend a system message to every sample for both chosen
            and rejected. This can serve as instructions to guide the model response. Setting this will OVERRIDE
            any system messages already present in the dataset. Default is None.
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        split (str): ``split`` argument for ``datasets.load_dataset``. You can use this argument to load a subset
            of a given split, e.g. ``split="train[:10%]"``. Default is "train".
        **load_dataset_kwargs (dict[str, Any]): additional keyword arguments to pass to ``load_dataset``.

    Examples:

    ::

        my_preference_dataset.json
        [
            {
                "chosen_conversations": [
                    {
                        "content": "What do I do when I have a hole in my trousers?",
                        "role": "user"
                    },
                    { "content": "Fix the hole.", "role": "assistant" }
                ],
                "rejected_conversations": [
                    {
                        "content": "What do I do when I have a hole in my trousers?",
                        "role": "user"
                    },
                    { "content": "Take them off.", "role": "assistant" }
                ]
            }
        ]

    ::

        >>> from torchtune.datasets import preference_dataset
        >>> column_map = {
        ...     "chosen": "chosen_conversations",
        ...     "rejected": "rejected_conversations"
        >>> }
        >>> dataset = preference_dataset(
        ...     tokenizer=tokenizer,
        ...     source="json",
        ...     column_map=column_map,
        ...     data_files="my_preference_dataset.json",
        ...     train_on_input=False,
        ...     split="train",
        >>> )
        >>> tokenizer.decode(dataset[0]["chosen_input_ids"], skip_special_tokens=True)
        What do I do when I have a hole in my trousers?Fix the hole.
        >>> tokenizer.decode(dataset[0]["rejected_input_ids"], skip_special_tokens=True)
        What do I do when I have a hole in my trousers?Take them off.

    This can also be accomplished via the yaml config:

    .. code-block:: yaml

        dataset:
          _component_: torchtune.datasets.preference_dataset
          source: json
          data_files: my_preference_dataset.json
          column_map:
            chosen: chosen_conversations
            rejected: rejected_conversations
          train_on_input: False
          split: train


    Returns:
        PreferenceDataset: The preference dataset built from source paired data.
    """

    message_transform = ChosenRejectedToMessages(
        train_on_input=train_on_input,
        column_map=column_map,
        new_system_prompt=new_system_prompt,
    )

    return PreferenceDataset(
        source=source,
        message_transform=message_transform,
        tokenizer=tokenizer,
        filter_fn=filter_fn,
        split=split,
        num_attack_tokens=num_attack_tokens,
        attack_token_prefix=attack_token_prefix,
        attack_token_mode=attack_token_mode,
        attack_tokens_per_sample=attack_tokens_per_sample,
        **load_dataset_kwargs,
    )
