"""Data loading, vocabulary construction, and batching utilities for CoNLL-2003 NER.

This module keeps all dataset-specific logic in one place:
loading the Hugging Face dataset, building a word vocabulary from the train split,
wrapping examples as PyTorch datasets, and padding variable-length sentences inside
batches with a custom collate function.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset as TorchDataset


SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]


class Vocab:
    """A token-to-ID vocabulary with support for unknown and special tokens.

    Parameters
    ----------
    special_tokens:
        Tokens inserted before dataset tokens. The default order makes ``<pad>``
        have ID 0, which is convenient for padded embeddings.
    cutoff:
        Minimum frequency required for a token to be added during ``build``.
    """

    def __init__(self, special_tokens: Optional[Sequence[str]] = None, cutoff: int = 5) -> None:
        self.special_tokens = list(special_tokens or SPECIAL_TOKENS)
        self.cutoff = cutoff
        self.token_to_idx: Dict[str, int] = {}
        self.idx_to_token: List[str] = []
        self._add_special_tokens()

    def _add_special_tokens(self) -> None:
        """Insert special tokens in a deterministic order."""
        for token in self.special_tokens:
            self._add_token(token)

    def _add_token(self, token: str) -> int:
        """Add one token if it is absent and return its ID."""
        if token not in self.token_to_idx:
            self.token_to_idx[token] = len(self.idx_to_token)
            self.idx_to_token.append(token)
        return self.token_to_idx[token]

    def build(self, token_sequences: Iterable[Sequence[str]]) -> None:
        """Build the vocabulary from tokenized sentences.

        Tokens with frequency lower than ``self.cutoff`` are skipped. Dataset
        tokens are sorted by decreasing frequency and alphabetically as a stable
        tie-breaker, making IDs reproducible across runs.
        """
        counter: Counter[str] = Counter()
        for tokens in token_sequences:
            counter.update(tokens)

        for token, frequency in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            if frequency >= self.cutoff:
                self._add_token(token)

    def token_to_id(self, token: str) -> int:
        """Return the ID of ``token`` or the ``<unk>`` ID if it is unseen."""
        return self.token_to_idx.get(token, self.token_to_idx["<unk>"])

    def id_to_token(self, idx: int) -> str:
        """Return the token associated with ``idx``.

        Raises
        ------
        IndexError
            If the ID is outside the vocabulary range.
        """
        return self.idx_to_token[idx]

    def size(self) -> int:
        """Return the number of tokens in the vocabulary."""
        return len(self.idx_to_token)

    def __len__(self) -> int:
        return self.size()

    @property
    def pad_id(self) -> int:
        """ID used for padding input token sequences."""
        return self.token_to_id("<pad>")

    @property
    def unk_id(self) -> int:
        """ID used for unseen tokens."""
        return self.token_to_id("<unk>")


class ConllNERDataset(TorchDataset):
    """PyTorch dataset wrapper for token-level NER examples.

    Each item returns token IDs, NER label IDs, and the original sentence length.
    The Hugging Face dataset is already tokenized, so this class only maps tokens
    to vocabulary IDs and leaves label IDs unchanged.
    """

    def __init__(
        self,
        hf_dataset: Dataset,
        vocab: Vocab,
        token_column: str = "tokens",
        label_column: str = "ner_tags",
        label_to_id: Optional[Mapping[str, int]] = None,
    ) -> None:
        self.hf_dataset = hf_dataset
        self.vocab = vocab
        self.token_column = token_column
        self.label_column = label_column
        self.label_to_id = dict(label_to_id or {})

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        example = self.hf_dataset[index]
        tokens = example[self.token_column]
        labels = example[self.label_column]

        token_ids = torch.tensor([self.vocab.token_to_id(token) for token in tokens], dtype=torch.long)
        label_ids = torch.tensor([self._label_to_id(label) for label in labels], dtype=torch.long)
        return token_ids, label_ids, len(tokens)

    def _label_to_id(self, label: int | str) -> int:
        """Convert either integer label IDs or string labels to integer IDs."""
        if isinstance(label, str):
            return self.label_to_id[label]
        return int(label)


def make_collate_fn(pad_token_id: int, label_pad_id: int = -100):
    """Create a collate function that pads only to the longest sentence in a batch.

    Parameters
    ----------
    pad_token_id:
        Token ID used to pad input sequences.
    label_pad_id:
        Label ID used for padded positions. The default ``-100`` matches the
        default ignore index of ``torch.nn.CrossEntropyLoss``.
    """

    def collate_fn(batch: Sequence[Tuple[torch.Tensor, torch.Tensor, int]]):
        token_ids, label_ids, lengths = zip(*batch)
        padded_tokens = pad_sequence(token_ids, batch_first=True, padding_value=pad_token_id)
        padded_labels = pad_sequence(label_ids, batch_first=True, padding_value=label_pad_id)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)
        return padded_tokens, padded_labels, lengths_tensor

    return collate_fn


@dataclass
class DataBundle:
    """Container returned by ``create_dataloaders``."""

    raw_dataset: DatasetDict
    train_loader: DataLoader
    valid_loader: DataLoader
    test_loader: Optional[DataLoader]
    vocab: Vocab
    label_names: List[str]
    label_pad_id: int


def load_conll_dataset(dataset_name: str, revision: Optional[str] = None) -> DatasetDict:
    """Load the CoNLL dataset from Hugging Face Datasets.

    Newer versions of ``datasets`` no longer execute dataset loading scripts.
    The original ``eriktks/conll2003`` repository contains ``conll2003.py``, so
    this project prefers the Parquet-converted revision when that dataset is
    requested. A script-free mirror is tried as a final fallback.
    """
    load_attempts: List[Dict[str, Any]] = []

    if revision:
        load_attempts.append({"path": dataset_name, "revision": revision})

    if dataset_name == "eriktks/conll2003":
        load_attempts.append({"path": dataset_name, "revision": "convert/parquet"})
        load_attempts.append({"path": "lhoestq/conll2003"})

    load_attempts.append({"path": dataset_name})

    errors: List[str] = []
    seen_attempts = set()
    for kwargs in load_attempts:
        key = tuple(sorted(kwargs.items()))
        if key in seen_attempts:
            continue
        seen_attempts.add(key)
        try:
            return load_dataset(**kwargs)
        except Exception as exc:  # pragma: no cover - depends on network/HF state
            errors.append(f"{kwargs}: {exc}")

    raise RuntimeError(
        "Could not load the CoNLL dataset. Tried script-free/parquet options. "
        "Details:\n" + "\n".join(errors)
    )


def maybe_select_subset(dataset: Dataset, subset_size: Optional[int]) -> Dataset:
    """Return the first ``subset_size`` rows when a positive subset size is given."""
    if subset_size is None or subset_size <= 0:
        return dataset
    return dataset.select(range(min(subset_size, len(dataset))))


DEFAULT_CONLL_NER_LABELS = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-MISC",
    "I-MISC",
]


def get_label_names(raw_dataset: DatasetDict, split: str = "train", label_column: str = "ner_tags") -> List[str]:
    """Read NER label names from dataset metadata, with a CoNLL fallback.

    Parquet conversions normally preserve ``ClassLabel`` names. If metadata is
    absent, the function falls back to the standard CoNLL-2003 NER label order.
    """
    label_feature = raw_dataset[split].features[label_column].feature
    if hasattr(label_feature, "names") and label_feature.names:
        return list(label_feature.names)

    observed_ids = sorted({int(label) for example in raw_dataset[split][label_column] for label in example})
    if len(observed_ids) == len(DEFAULT_CONLL_NER_LABELS) and observed_ids == list(range(len(DEFAULT_CONLL_NER_LABELS))):
        return DEFAULT_CONLL_NER_LABELS

    return [f"LABEL_{idx}" for idx in observed_ids]


def create_dataloaders(config: MutableMapping) -> DataBundle:
    """Load data, build the vocabulary, and create train/validation/test loaders."""
    data_cfg = config["data"]
    train_cfg = config["training"]

    raw_dataset = load_conll_dataset(
        data_cfg["dataset_name"], revision=data_cfg.get("dataset_revision")
    )
    valid_split = "validation" if "validation" in raw_dataset else "dev"

    train_split = raw_dataset["train"]
    valid_split_ds = raw_dataset[valid_split]
    test_split_ds = raw_dataset["test"] if "test" in raw_dataset else None

    if data_cfg.get("use_subset", False):
        train_split = maybe_select_subset(train_split, int(data_cfg.get("train_subset_size", 512)))
        valid_split_ds = maybe_select_subset(valid_split_ds, int(data_cfg.get("valid_subset_size", 128)))
        if test_split_ds is not None:
            test_split_ds = maybe_select_subset(test_split_ds, int(data_cfg.get("test_subset_size", 128)))

    vocab = Vocab(cutoff=int(data_cfg.get("vocab_cutoff", 5)))
    vocab.build(train_split[data_cfg.get("token_column", "tokens")])

    label_names = get_label_names(raw_dataset, split="train", label_column=data_cfg.get("label_column", "ner_tags"))
    label_to_id = {name: idx for idx, name in enumerate(label_names)}
    label_pad_id = int(data_cfg.get("label_pad_id", -100))

    collate_fn = make_collate_fn(vocab.pad_id, label_pad_id=label_pad_id)
    loader_kwargs = {
        "batch_size": int(train_cfg.get("batch_size", 32)),
        "num_workers": int(data_cfg.get("num_workers", 0)),
        "collate_fn": collate_fn,
    }

    train_dataset = ConllNERDataset(train_split, vocab, label_to_id=label_to_id)
    valid_dataset = ConllNERDataset(valid_split_ds, vocab, label_to_id=label_to_id)
    test_dataset = ConllNERDataset(test_split_ds, vocab, label_to_id=label_to_id) if test_split_ds is not None else None

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs) if test_dataset is not None else None

    return DataBundle(
        raw_dataset=raw_dataset,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        vocab=vocab,
        label_names=label_names,
        label_pad_id=label_pad_id,
    )


def describe_dataset(raw_dataset: DatasetDict) -> str:
    """Return a compact text summary for Part 1 dataset analysis."""
    split_sizes = {split: len(raw_dataset[split]) for split in raw_dataset.keys()}
    columns = raw_dataset["train"].column_names
    features = raw_dataset["train"].features
    return (
        f"Splits: {split_sizes}\n"
        f"Columns/features: {columns}\n"
        f"Feature schema: {features}\n"
        "POS tags represent each token's grammatical part of speech.\n"
        "Chunk tags represent shallow syntactic phrase chunks in BIO/IOB style.\n"
        "NER tags represent named-entity labels such as person, organization, location, and miscellaneous."
    )
