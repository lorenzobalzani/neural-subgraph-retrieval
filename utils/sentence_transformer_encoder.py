from typing import List

import torch
from pykeen.nn.text import TransformerTextEncoder
from sentence_transformers import SentenceTransformer


class SentenceTransformerEncoder(TransformerTextEncoder):
    def __init__(
        self, encoder_model_name_or_path: str, embedding_dim: int, device: str
    ) -> None:
        super().__init__()
        self._model = SentenceTransformer(
            encoder_model_name_or_path, device=device)
        self._embedding_dim: int = embedding_dim

    def forward_normalized(
        self, texts: List[str], show_progress_bar: bool = False, batch_size: int = 256
    ) -> torch.FloatTensor:
        encoder_output = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=show_progress_bar,
        )
        matryoshka = encoder_output[:, : self._embedding_dim]
        return (
            matryoshka
            / torch.linalg.matrix_norm(matryoshka, ord=2, keepdim=True)  # pylint: disable=not-callable
            .clamp(min=1e-12)
            .contiguous()
        )
