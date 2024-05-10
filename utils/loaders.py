from typing import Set, Tuple

import networkx as nx
import pandas as pd


def load_medqa_dataset(
    medqa_answers_path: str, medqa_questions_path: str, concept_names_path: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load MedQA dataset and related files.

    This function reads the MedQA dataset, including the answers, questions, and concept names,
    and performs some initial checks.

    Returns:
    - medqa_answers: Pandas DataFrame containing answers
    - medqa_questions: Pandas DataFrame containing questions
    - concept_names: Pandas DataFrame containing concept names

    Note:
    The MedQA dataset should be structured as follows:
    - 'train.statement.jsonl': JSON file containing answers
    - 'train.grounded.json': JSON file containing questions
    - 'concept_names.tsv': Tab-separated values file containing concept names
    - The number of questions in 'train.grounded.json' should be four times the number of answers in
        'train.statement.jsonl'.
    - Each question in the MedQA dataset should have only one correct answer.

    Example usage:
    medqa_answers, medqa_questions, concept_names = load_medqa_data()
    """
    medqa_answers = pd.DataFrame(pd.read_json(medqa_answers_path, lines=True))
    medqa_questions = pd.DataFrame(pd.read_json(medqa_questions_path))
    concept_names = pd.DataFrame(pd.read_csv(concept_names_path, sep="\t"))

    # Check if the number of questions is four times the number of answers
    if medqa_questions.shape[0] != 4 * medqa_answers.shape[0]:
        raise ValueError(
            "The number of questions should be four times the number of answers."
        )

    return medqa_answers, medqa_questions, concept_names


def load_umls(file_path: str) -> Tuple[nx.DiGraph, Set[str]]:
    """
    Load UMLS (Unified Medical Language System) data from a file and create a directed graph.

    Parameters:
    - file_path (str): The path to the UMLS data file.

    Returns:
    - nx.DiGraph: A directed graph representing the UMLS data, where nodes are concepts,
                 and edges represent relationships between concepts.

    The function reads triplets from the specified file and constructs a directed graph,
    where each triplet is represented as a directed edge between two nodes, with a
    labeled relationship. The UMLS data should be in tab-separated format with the
    structure (head, relation, tail).

    Example:
    - Input file format:
      Concept1   IsA       Concept2
      Concept2   PartOf   Concept3
      ...

    - Output graph:
      Nodes number: <number_of_nodes>
      Edges number: <number_of_edges>

    Note: The function skips the header line in the input file.

    Requires the 'networkx' library for graph creation and manipulation.
    """

    kg = nx.DiGraph()
    relation_types: Set[str] = set()
    with open(file_path, "r") as file:
        next(file)  # skip header
        for line in file:
            triplet = line.strip().split("\t")

            if len(triplet) == 3:
                head, relation, tail = triplet
                kg.add_node(head)
                kg.add_node(tail)
                kg.add_edge(head, tail, relation=relation)
                relation_types.add(relation)

    print(f"Nodes number: {kg.number_of_nodes()}")
    print(f"Edges number: {kg.number_of_edges()}")
    print(f"Relation types: {len(relation_types)}")
    return kg, relation_types
