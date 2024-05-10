import gc
import os
import sys
from typing import Dict, Optional

import pandas as pd
import torch
from arg_parser import parse_arguments
from huggingface_hub import PyTorchModelHubMixin
from pykeen.models import DistMult

from pykeen.nn import TextRepresentation
from pykeen.pipeline import pipeline_from_path, PipelineResult
from pykeen.predict import predict_target, Predictions
from pykeen.triples import TriplesFactory
from utils.sentence_transformer_encoder import SentenceTransformerEncoder

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


class LinkPredictor(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        kg_triples_filename: str,
        encoder_model_name_or_path: Optional[str] = None,
        pykeen_checkpoint_file: Optional[str] = None,
        encoder_embedding_dim: int = 128,
        distmult_embedding_dim: int = 2000,
        pykeen_checkpoint_folder: str = "checkpoints",
        random_seed: int = 42,
    ) -> None:
        super().__init__()

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._triples_factory = TriplesFactory.from_path(kg_triples_filename)
        self._pykeen_checkpoint_file = pykeen_checkpoint_file
        self._pykeen_checkpoint_folder = pykeen_checkpoint_folder
        self._encoder_embedding_dim = encoder_embedding_dim
        self._distmult_embedding_dim = distmult_embedding_dim

        # Initialize the model(s)
        if encoder_model_name_or_path:
            self._entity_representations = TextRepresentation.from_triples_factory(
                triples_factory=self._triples_factory,
                encoder=SentenceTransformerEncoder,
                encoder_kwargs={
                    "encoder_model_name_or_path": encoder_model_name_or_path,
                    "embedding_dim": encoder_embedding_dim,
                    "device": self._device.type,
                },
            )

            self._pykeen_model = DistMult(
                triples_factory=self._triples_factory,
                embedding_dim=self._entity_representations.shape[0],
                entity_representations=self._entity_representations,
                random_seed=random_seed,
            )
        else:
            self._pykeen_model = DistMult(
                triples_factory=self._triples_factory,
                embedding_dim=distmult_embedding_dim,
                entity_initializer="xavier_uniform",
                relation_initializer="xavier_uniform",
                entity_constrainer="normalize",
                random_seed=random_seed,
            )

    def upload_model(self, huggingface_hub_url: str) -> None:
        checkpoint_path: str = os.path.join(
            self._pykeen_checkpoint_folder, self._pykeen_checkpoint_file
        )
        checkpoint: torch.Tensor = torch.load(
            checkpoint_path, map_location=self._device
        )
        self._pykeen_model.load_state_dict(checkpoint["model_state_dict"])
        self.push_to_hub(huggingface_hub_url)

    def _load_umls_kg(
        self,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        random_state: int,
    ):
        train_set, val_set, test_set = self._triples_factory.split(
            [train_ratio, val_ratio, test_ratio], random_state=random_state
        )
        return train_set, val_set, test_set

    @classmethod
    def print_metrics(cls, result: PipelineResult) -> Dict[str, str]:
        # MR (Mean Rank): [1, num_entities]; the lower it is, the better the model results are. It's out of 9958
        # MRR (Mean Reciprocal Rank): [0, 1]; the higher it is, the better the model results
        # IGMR (Inverse Geometric Mean Rank): [0, 1]; the higher it is, the
        # better the model results
        metrics = {
            metric: str(result.get_metric(metric)) for metric in ["mr", "mrr", "igmr"]
        }
        print(metrics)
        return metrics

    def train_pykeen_model(
        self,
        pykeen_config: str,
        n_epochs: int,
        batch_size: int = 8,
        checkpoint_frequency: int = 5,
        es_frequency: int = 10,
        es_patience: int = 2,
        es_relative_delta: float = 2e-3,
        random_state: int = 42,
    ) -> PipelineResult:
        gc.collect()
        print("Using device:", self._device)
        if torch.cuda.is_available():
            torch.cuda.set_device(self._device)
            torch.cuda.empty_cache()
            torch.manual_seed(random_state)

        train_set, val_set, test_set = self._load_umls_kg(0.8, 0.1, 0.1, random_state)

        print(f"Loading config from {pykeen_config}")

        pipeline_params = {
            "model": self._pykeen_model,
            "device": self._device,
            "path": pykeen_config,
            "training": train_set,
            "validation": val_set,
            "testing": test_set,
            "stopper": "early",
            "stopper_kwargs": {
                "frequency": es_frequency,
                "patience": es_patience,
                "relative_delta": es_relative_delta,
            },
            "random_seed": random_state,
            "use_tqdm": True,
            "training_kwargs": {
                "num_epochs": n_epochs,
                "batch_size": batch_size,
                "checkpoint_directory": self._pykeen_checkpoint_folder,
                "checkpoint_name=": self._pykeen_checkpoint_file,
                "checkpoint_frequency": checkpoint_frequency,
            },
        }

        pipeline_result = pipeline_from_path(**pipeline_params)
        pipeline_result.save_to_directory("training_" + self._pykeen_checkpoint_file)
        self.print_metrics(pipeline_result)
        return pipeline_result

    def forward(self, head: str, relation: str) -> Predictions:
        return predict_target(
            model=self._pykeen_model,
            head=head,
            relation=relation,
            triples_factory=self._triples_factory,
        )

    @torch.inference_mode()
    def inference(self, head: str, relation: str, k: int = 5) -> pd.DataFrame:
        self.eval()
        return self(head, relation).df.iloc[:k]


if __name__ == "__main__":
    args = parse_arguments()
    link_predictor = LinkPredictor(
        kg_triples_filename=args.umls_triples_path,
        encoder_model_name_or_path=args.encoder_model_name_or_path,
        pykeen_checkpoint_file=args.pykeen_checkpoint_file,
    )
    if args.umls_triples_path and args.n_epochs and args.pykeen_train_config:
        link_predictor.train_pykeen_model(args.pykeen_train_config, args.n_epochs)
    if args.pykeen_checkpoint_file and args.huggingface_hub_url:
        link_predictor.upload_model(args.huggingface_hub_url)
