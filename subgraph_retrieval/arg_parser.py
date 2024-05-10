import argparse


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve subgraphs given a question idx in the MedQA dataset."
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
        "--next_relation_predictor",
        type=str,
        required=True,
        help="URL or local path to the Next Relation Predictor model.",
    )
    parser.add_argument(
        "--link_predictor",
        type=str,
        required=True,
        help="URL or local path to the Link Predictor model.",
    )
    parser.add_argument(
        "--encoder_model_name_or_path",
        type=str,
        required=False,
        default=None,
        help="Name or path of the transformer encoder to be used for the Link Predictor model.",
    )
    parser.add_argument(
        "--coder_path",
        type=str,
        required=True,
        help="URL or local path to the Coder model for encoding.",
    )
    parser.add_argument(
        "--question_idx",
        type=int,
        required=True,
        help="Index of the question to be retrieved from the MedQA dataset.",
    )
    return parser.parse_args()
