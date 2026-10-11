"""Neural sequence tagger models for NER."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

RNNType = Literal["rnn", "lstm", "gru"]


class SequenceTagger(nn.Module):
    """Bidirectional RNN/LSTM/GRU encoder for token-level NER classification.

    The model architecture is:
    ``Embedding -> bidirectional recurrent encoder -> Linear classifier``.
    For every token position, the classifier predicts one NER label.
    """

    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embedding_size: int = 64,
        hidden_size: int = 64,
        padding_idx: int = 0,
        rnn_type: RNNType = "lstm",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)

        rnn_cls = self._get_rnn_class(self.rnn_type)
        self.encoder = rnn_cls(
            input_size=embedding_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    @staticmethod
    def _get_rnn_class(rnn_type: str):
        """Map a string model type to the matching PyTorch recurrent layer."""
        mapping = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
        if rnn_type not in mapping:
            raise ValueError(f"Unsupported rnn_type={rnn_type!r}. Choose from {list(mapping)}.")
        return mapping[rnn_type]

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return token-level logits of shape ``[batch, seq_len, num_labels]``."""
        embedded = self.dropout(self.embedding(input_ids))
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.encoder(packed)
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=input_ids.size(1),
        )
        return self.classifier(self.dropout(output))
