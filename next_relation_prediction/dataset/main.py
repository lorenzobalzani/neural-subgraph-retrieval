import datetime
import os
import sys
from typing import Dict, List, Optional, Set, Union

import networkx as nx
import pandas as pd
import torch
from arg_parser import parse_arguments
from tqdm import tqdm
from utils.coder import Coder
from utils.general import check_device, split_and_push_to_hf_hub
from utils.loaders import load_medqa_dataset, load_umls

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))


def find_shortest_path_per_question(
    kg: nx.Graph,
    coder: Coder,
    medqa_questions: pd.DataFrame,
    medqa_answers: pd.DataFrame,
    concept_names: pd.DataFrame,
    question_idx: int,
    normalize_nodes: bool = True,
    debug_print: bool = False,
) -> Dict[str, Union[List[Dict[str, str]], Dict[str, int]]]:
    """
    Finds the shortest paths in a knowledge graph from topic entities to the correct answer for a given question.

    Parameters:
    kg (nx.Graph): A NetworkX graph representing the knowledge graph.
    coder (CODER): The CODER (encoder-only) model to perform biomedical term-normalization.
    medqa_questions (pd.DataFrame): A Pandas DataFrame containing medical questions and topic entity IDs.
    medqa_answers (pd.DataFrame): A Pandas DataFrame containing medical question answers and choices.
    concept_names (pd.DataFrame): A Pandas DataFrame mapping concept IDs to concept names.
    question_idx (int): Index of the question to process.
    normalize_nodes (bool, optional): Whether to use CODER model to perform term-normalization on nodes. Default to True
    debug_print (bool, optional): Whether to print debug information. Default is False.

    Returns:
    dict: A dictionary containing the shortest paths and additional information.
        - 'paths' (list of dict): List of dictionaries, each containing the decomposed path.
        - 'info' (dict): Additional information including:
            - 'n_topic_entities' (int): Number of topic entities associated with the question.
            - 'not_found_nodes' (int): Number of topic/answer entities not found in the knowledge graph.
            - 'not_found_paths' (int): Number of paths not found between topic entities and the answer.
    """

    not_found_nodes: int = 0
    not_found_paths: int = 0
    all_topic_entities: Set[str] = set()
    dataset_rows: List[Dict[str, str]] = []

    # Extract the dataset row and the question
    row = medqa_answers.iloc[question_idx]
    question = row["question"]["stem"]
    # Get the 4 possible answers, with index and text, e.g. 3 and
    # "Nitrofurantoin" (idx ranges from 0 to 3).
    possible_choices = [
        {"idx": idx, "text": choice["text"]}
        for idx, choice in enumerate(row["question"]["choices"])
    ]

    for answer in tqdm(possible_choices, leave=False):
        if normalize_nodes:
            answer["text"] = coder.node_normalization([answer["text"]])[answer["text"]][
                0
            ]
        if debug_print:
            print(
                "\nComputing paths from all nodes to '{target}'.".format(
                    target=answer["text"]
                )
            )
        paths_to_answer: Dict[str, List[str]] = nx.single_target_shortest_path(
            kg, answer["text"]
        )

        # Extract the question topic entities IDs
        question_topic_entities_id = pd.Series(
            medqa_questions.iloc[question_idx * 4 + answer["idx"]]["qc"]
        )

        # Convert topic entities IDs to names
        question_topic_entities = question_topic_entities_id.map(
            lambda concept_id: concept_names.loc[
                concept_names["concept_id"] == concept_id, "concept_name"
            ].iloc[0]
        )

        if normalize_nodes:
            # for each concept, take the most similar one
            question_topic_entities = [
                similar_nodes[0]
                for similar_nodes in coder.node_normalization(
                    question_topic_entities
                ).values()
            ]

        # Find all shortest paths starting from each topic entity
        for topic_entity in question_topic_entities:
            try:
                all_topic_entities.add(topic_entity)

                # Extract all edges that form the path between the source and
                # the target nodes
                for path in [paths_to_answer[topic_entity]]:
                    entity_2_answer_relations = [
                        kg[path[i]][path[i + 1]]["relation"]
                        for i in range(len(path) - 1)
                    ]
                    question_2_answer_relations = (
                        [question] + entity_2_answer_relations + ["END"]
                    )
                    nodes = [path[i] for i in range(len(path))]

                    # Decompose path, e.g.
                    # [question; R_1; R_2; ...; R_N; END] ->
                    # [question][R_1]; [question; R_1][R_2] ... [question; R_1; ... ; R_N][END]
                    for idx in range(len(question_2_answer_relations) - 1):
                        dataset_rows.append(
                            {
                                "question_id": question_idx,
                                "input": ",".join(
                                    question_2_answer_relations[: idx + 1]
                                ),
                                "output": str(question_2_answer_relations[idx + 1]),
                            }
                        )

                    if debug_print:
                        print(
                            f"'{topic_entity}' -> '{answer}':\n",
                            f"Nodes: {nodes}\n",
                            f"Edges: {entity_2_answer_relations}\n\n",
                        )

            except (NameError, nx.NodeNotFound) as NodeException:
                not_found_nodes += 1
                print(f"Node not found: {NodeException}")
                continue

            except KeyError:
                not_found_paths += 1
                continue

    return {
        "paths": dataset_rows,
        "info": {
            "n_topic_entities": len(all_topic_entities),
            "not_found_nodes": not_found_nodes,
            "not_found_paths": not_found_paths,
        },
    }


def create_dataset(
    medqa_folder: str,
    umls_triples_path: str,
    umls_embeddings_path: str,
    coder_path: str,
    device: torch.device,
    dataset_output: Optional[str] = None,
    stats_folder: Optional[str] = None,
    until_idx: Optional[int] = None,
) -> pd.DataFrame:
    # Load UMLS and create the KG
    kg, relation_types = load_umls(umls_triples_path)

    # Load MedQA dataset
    medqa_answers, medqa_questions, concept_names = load_medqa_dataset(
        os.path.join(medqa_folder, "train.statement.jsonl"),
        os.path.join(medqa_folder, "train.grounded.json"),
        os.path.join(medqa_folder, "concept_names.tsv"),
    )

    coder = Coder(coder_path, list(kg.nodes()), device)
    coder.load_umls_embeddings(umls_embeddings_path)
    n_topic_entities: int = 0
    not_found_nodes: int = 0
    not_found_paths: int = 0
    n_errors: int = 0

    dataset: List[Dict[str, str]] = []

    for question_idx in tqdm(
        range(len(medqa_answers[:until_idx])), leave=True, desc="Creating dataset"
    ):
        try:
            result = find_shortest_path_per_question(
                kg,
                coder,
                medqa_questions,
                medqa_answers,
                concept_names,
                question_idx,
            )
            dataset += result["paths"]
            n_topic_entities += result["info"]["n_topic_entities"]
            not_found_nodes += result["info"]["not_found_nodes"]
            not_found_paths += result["info"]["not_found_paths"]
        except Exception as e:
            print(f"Exception encountered at question {question_idx}.\nError:", e)
            n_errors += 1

    dataset: pd.DataFrame = (
        pd.DataFrame(dataset)
        .drop_duplicates(subset=["input", "output"])
        .reset_index()
        .drop(["index"], axis=1)
        .astype({"question_id": "uint16"})
    )

    debug_info = "\n".join(
        [
            f"\nNot found nodes: {not_found_nodes}",
            f"Ratio of not found nodes: {not_found_nodes / n_topic_entities:.2%}",
            f"Not found paths: {not_found_paths}",
            f"Ratio of not found paths: {not_found_paths / n_topic_entities:.2%}",
            f"Dataset length: {len(dataset)}",
            f"Topic entities: {n_topic_entities}",
            f"Errors: {n_errors}",
        ]
    )
    print(debug_info)

    if dataset_output:
        dataset.to_csv(dataset_output, sep="\t", header=True, index_label="id")

    if stats_folder:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(os.path.join(stats_folder, f"stats_{timestamp}.info"), "w") as file:
            file.write(debug_info)

    return dataset


if __name__ == "__main__":
    args = parse_arguments()
    dataframe = create_dataset(
        args.medqa_folder,
        args.umls_triples_path,
        args.umls_embeddings_path,
        args.coder_path,
        check_device(),
        args.output_path,
        args.stats_folder,
    )
    split_and_push_to_hf_hub(dataframe, args.huggingface_hub_url)
