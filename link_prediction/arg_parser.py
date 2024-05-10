import argparse


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a link predictor")
    parser.add_argument(
        "--umls_triples_path",
        type=str,
        required=False,
        help="Path to the file containing UMLS triples (Optional). If not provided, the model will not be trained.",
    )
    parser.add_argument(
        "--pykeen_train_config",
        type=str,
        required=False,
        help="Path to the PyKEEN pipeline configuration file (Optional). If not provided, the model will not be "
        "trained.",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        required=False,
        help="Number of epochs to train the PyKEEN model (Optional). If not provided, the model will not be trained.",
    )
    parser.add_argument(
        "--encoder_model_name_or_path",
        type=str,
        required=False,
        default=None,
        help="Name of the encoder to be used to create vector representations of the entities/relations (Optional)."
        " If not, the model will use the default DistMult model.",
    )
    parser.add_argument(
        "--pykeen_checkpoint_file",
        type=str,
        required=True,
        help="Filename of the PyKEEN checkpoint to save inside the checkpoints folder. If not provided, the model "
        "will not be"
        "uploaded.",
    )
    parser.add_argument(
        "--huggingface_hub_url",
        type=str,
        required=False,
        help="URL to upload the model to Hugging Face Hub (Optional). If not provided, the model will not be uploaded.",
    )
    return parser.parse_args()
