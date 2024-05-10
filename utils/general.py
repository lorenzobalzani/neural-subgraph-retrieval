import os
from typing import Iterable, Optional, Sequence, Set, Tuple, TypeVar

import pandas as pd
import torch
from datasets import Dataset, DatasetDict


def extract_question_and_topic_entities(
    medqa_answers: pd.DataFrame,
    medqa_questions: pd.DataFrame,
    concept_names: pd.DataFrame,
    question_idx: int,
    n_answers: int = 4,
) -> Tuple[str, Sequence[str]]:
    """
    Extract the question and the topic entities from the MedQA dataset
    Parameters:
        medqa_answers: the MedQA answers DataFrame
        medqa_questions: the MedQA questions DataFrame
        concept_names: the concept names DataFrame
        question_idx: the question index
        n_answers: the number of answers per question. Default to 4.
    Returns:
        the question and the topic entities
    """
    # Extract the question
    question: str = medqa_answers.iloc[question_idx]["question"]["stem"]

    # Extract the question topic entities IDs, i.e., the union of the concept
    # IDs of the 4 answer choices
    topic_entities_ids: Set[str] = set()
    for ans_idx in range(n_answers):
        topic_entities_ids |= set(
            medqa_questions.iloc[question_idx * n_answers + ans_idx]["qc"]
        )

    # Convert topic entities IDs to names
    topic_entities_names = [
        concept_names.loc[
            concept_names["concept_id"] == concept_id, "concept_name"
        ].iloc[0]
        for concept_id in topic_entities_ids
    ]

    return question, topic_entities_names


def check_device() -> torch.device:
    # Check if GPU is available. If so, enable cuGraph and cuDF compatibility
    # with NetworkX and Pandas
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("Using GPU. Enabling cuGraph and cuDF compatibility.")
        os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = "cugraph"
        import cudf.pandas  # pylint: disable=import-error

        cudf.pandas.install()
    return device


def split_and_push_to_hf_hub(
    dataset: pd.DataFrame,
    dataset_url_hf_hub: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> None:
    """
    Split the dataset into train, validation, and test sets and push them to the HuggingFace Hub.
    The splitting is done sequentially, so the train set will contain the first train_ratio of the dataset, the
    validation set will contain the next val_ratio of the dataset, and the test set will contain the rest.
    """
    total_samples = len(dataset)
    train_samples = int(train_ratio * total_samples)
    val_samples = int(val_ratio * total_samples)

    train_data = dataset[:train_samples]
    eval_data = dataset[train_samples : train_samples + val_samples]
    test_data = dataset[train_samples + val_samples :]

    try:
        dataset_hf = DatasetDict(
            {
                "train": Dataset.from_pandas(train_data.reset_index()),
                "eval": Dataset.from_pandas(eval_data.reset_index()),
                "test": Dataset.from_pandas(test_data.reset_index()),
            }
        )
        dataset_hf.push_to_hub(dataset_url_hf_hub)
    except Exception:
        print(
            f"Error: Uploading the dataset requires being logged in as '{dataset_url_hf_hub.split('/')[0]}' on the "
            f"HuggingFace Hub."
        )


T = TypeVar("T")


class Stack(list):
    def __init__(self, items: Optional[Iterable[T]] = None) -> None:
        super().__init__(items if items else [])

    def peek(self) -> Optional[T]:
        if not self.is_empty():
            return self[-1]
        else:
            return None

    def push(self, item: T) -> None:
        self.append(item)

    def is_empty(self) -> bool:
        return len(self) == 0

    def size(self) -> int:
        return len(self)

    def copy(self) -> "Stack[T]":
        return Stack(self)
