import argparse


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a LM to predict relations in the UMLS KG and the MedQA dataset."
    )
    parser.add_argument(
        "--pretrained_model_name",
        type=str,
        required=True,
        help="Name of the pretrained model to fine-tune.",
    )
    parser.add_argument(
        "--model_huggingface_hub_url",
        type=str,
        required=True,
        help="URL of the HuggingFace Hub repository where the model is uploaded.",
    )
    parser.add_argument(
        "--dataset_huggingface_hub_url",
        type=str,
        required=True,
        help="Dataset URL on the HuggingFace Hub.",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        required=True,
        help="Number of epochs to fine-tune the model.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        required=False,
        default=8,
        help="Batch size to use for training and evaluation. Default to 4.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        required=False,
        default=8,
        help="Batch size to use for training and evaluation. Default to 4.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        required=False,
        default=2,
        help="Number of epochs to wait before early stopping. Default to 2.",
    )
    parser.add_argument(
        "--stats_folder",
        type=str,
        required=False,
        default=".",
        help="Folder to save the training statistics. Default to the current folder.",
    )
    return parser.parse_args()
