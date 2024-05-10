import argparse


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a dataset for training a language model to predict relations in the UMLS KG and the MedQA "
        "dataset."
    )
    parser.add_argument(
        "--medqa_folder",
        type=str,
        required=True,
        help="Path to the folder containing the MedQA dataset files (train.statement.jsonl, train.grounded.json, "
        "concept_names.tsv).",
    )
    parser.add_argument(
        "--umls_triples_path",
        type=str,
        required=True,
        help="Path to the file containing UMLS triples.",
    )
    parser.add_argument(
        "--umls_embeddings_path",
        type=str,
        required=True,
        help="Path to the file containing UMLS embeddings.",
    )
    parser.add_argument(
        "--coder_path",
        type=str,
        required=True,
        help="URL or local path to the Coder model for encoding.",
    )
    parser.add_argument(
        "--huggingface_hub_url",
        type=str,
        required=True,
        help="URL of the HuggingFace Hub repository where the dataset will be uploaded.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        help="Path to write the tsv file (Optional).",
    )
    parser.add_argument(
        "--stats_folder",
        type=str,
        required=False,
        default=".",
        help="Path to write the statistics. Default to the current folder.",
    )
    return parser.parse_args()
