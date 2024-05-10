import os
import sys
from collections import Counter
from functools import reduce
from typing import Dict, List, Optional, Sequence, Union

import networkx as nx

from huggingface_hub import PyTorchModelHubMixin
from link_prediction.link_predictor import LinkPredictor
from torch import nn
from tqdm import tqdm
from transformers import pipeline
from utils.coder import Coder
from utils.general import check_device, extract_question_and_topic_entities, Stack
from utils.loaders import load_medqa_dataset, load_umls

from subgraph_retrieval.arg_parser import parse_arguments

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


class SubgraphRetrieval(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        kg_triples_filename: str,
        next_relation_predictor_name: str,
        coder_path: str,
        umls_embeddings_path: str,
        link_predictor_name: str,
        encoder_model_name_or_path: Optional[str],
    ) -> None:
        super().__init__()

        # Set device (GPU if available)
        self._device = check_device()

        # Init link predictor
        self.link_predictor = LinkPredictor.from_pretrained(
            link_predictor_name,
            kg_triples_filename=kg_triples_filename,
            encoder_model_name_or_path=encoder_model_name_or_path,
        )

        # Load UMLS and create the KG
        self._kg, self._relation_types = load_umls(kg_triples_filename)

        # Init coder
        self._coder = Coder(coder_path, list(self._kg.nodes()), self._device)
        self._coder.load_umls_embeddings(umls_embeddings_path)

        # Init next relation predictor
        self.next_relation_predictor = pipeline(
            "text-classification", model=next_relation_predictor_name
        )

    @staticmethod
    def _merge_graphs(
        graphs: List[nx.DiGraph], min_node_occurences: Union[str, int] = "all"
    ) -> nx.DiGraph:
        """
        Take N subgraphs and merge them into a single graph.
        First, a counter is used to count the number of occurences of each node in all subgraphs.
        Then, the nodes that occur at least min_occurences times are used to create the combined graph.
        Finally, the edges that connect nodes that are already in the combined graph are added to the combined graph.
        Args:
            graphs: the list of subgraphs.
            subgraphs min_node_occurences: the minimum number of occurences of a node (in all subgraphs) to be included
                in the combined graph.

        Returns: the combined graph
        TODO: add entities linked to common nodes
        """
        combined_graph = nx.DiGraph()

        # Remove nodes that do not occur at least min_occurences times and use
        # the rest to create the combined graph
        node_counter = Counter(
            [
                node
                for graph_list in graphs
                for graph in graph_list
                for node in graph.nodes()
            ]
        )
        common_nodes = [
            node
            for node, node_counter in node_counter.items()
            if node_counter
            >= (len(graphs) if min_node_occurences == "all" else min_node_occurences)
        ]
        combined_graph.add_nodes_from(common_nodes)

        combined_graph.add_edges_from(
            [
                edge
                for graph in graphs
                for edge in graph.edges()
                if edge[0] in combined_graph and edge[1] in combined_graph
            ]
        )

        return combined_graph

    def _expand_paths(
        self,
        query: str,
        top_k: int,
        k_cut_off: int = 3,
        threshold: float = 0.1,
    ) -> List[List[Dict]]:
        """
        Explore all the paths in the KG starting from a query and a starting node, i.e., the topic entity.
        Args:

            query: the query
            top_k: the beam search size, i.e., the number of next relations to consider
            k_cut_off: the beam search keeps expanding each path in an exponential way. This parameter
                sets how often to cut off the beam search, keeping just the top-k paths. Optional, default is 5
            threshold: the threshold to consider a next relation. Optional, default is 0.1
        Returns:
            the top-k paths, a list of lists of dictionaries, where each dictionary contains the relation and the score
        """

        paths: Stack[List[Dict]] = Stack([[]])
        curr_level: int = -1
        keep_expanding: bool = True

        while keep_expanding:
            curr_level += 1
            if not any([len(path) == curr_level for path in paths]):
                keep_expanding = False

            new_paths = Stack()
            while not paths.is_empty():
                path = paths.pop()

                # If a path is not at the current level, don't process it
                if not len(path) == curr_level:
                    new_paths.push(path)
                    continue

                # Predict the next relations
                new_query: str = ",".join([query] + [path["relation"] for path in path])
                next_relations: List[Dict[str, Union[str, int]]] = [
                    {"relation": pred["label"], "score": pred["score"]}
                    for pred in self.next_relation_predictor(new_query, top_k=top_k)
                    if pred["score"] > threshold and pred["label"] != "END"
                ]

                # If there are next relations, expand the paths, otherwise,
                # keep the path
                if next_relations:
                    for next_relation in next_relations:
                        new_paths.append(path + [next_relation])
                else:
                    new_paths.append(path)

            # Batch update
            paths = new_paths.copy()

            # Cut off the beam search
            if curr_level % k_cut_off == 0 or not keep_expanding:
                # if the number of paths is greater than top_k, keep the top_k
                # paths
                if paths.size() > top_k:
                    scores = [
                        reduce(
                            lambda x, y: x * y,
                            [next_relation["score"] for next_relation in path],
                        )
                        for path in paths
                    ]
                    top_k_idx = sorted(
                        range(len(scores)), key=lambda i: scores[i], reverse=True
                    )[:top_k]
                    paths = Stack([paths[i] for i in top_k_idx])

        return paths

    def _induce_subgraphs(
        self, top_k_paths: List[List[Dict]], topic_entity: str, top_k: int
    ) -> List[nx.DiGraph]:
        subgraphs: List[nx.DiGraph] = []

        for path in top_k_paths:
            # Create a graph from the path
            graph = nx.DiGraph()
            current_node = topic_entity
            for next_relation in path:
                next_relation = next_relation["relation"]
                # Predict the next node given the current node and the next
                # relation
                potential_nodes = self.link_predictor.inference(
                    head=current_node, relation=next_relation
                )["tail_label"]

                # Force avoid navigating to the same node
                next_node = [node for node in potential_nodes if node != current_node][
                    0
                ]

                graph.add_nodes_from([current_node, next_node])
                graph.add_edge(current_node, next_node, relation=next_relation)
                current_node = next_node
            subgraphs.append(graph)

        return subgraphs

    def _retrieve_subgraphs_from_topic_entity(
        self, query: str, topic_entity: str, top_k: int, max_hops: int = 15
    ) -> nx.DiGraph:
        """
        Retrieve a subgraph given a topic entity
        Parameters:
            query: the query (medQA question) to start the subgraph retrieval
            topic_entity: the topic entity
            top_k: the beam search size, i.e., the number of next relations to consider,
                iff the next relation is not END and the confidence is > 0.5)
            max_hops: the maximum number of hops to retrieve the subgraph
        Returns:
            the retrieved subgraph
        """

        # Retrieve the top-k paths
        print(f"Retrieving paths starting from the topic entity: {topic_entity}")
        top_k_paths: List[List[Dict]] = self._expand_paths(query, top_k)

        # Induce subgraphs from the list of the top-k paths
        print(f"Inducing subgraphs from the top-k paths: {top_k_paths}")
        top_k_subgraphs: List[nx.DiGraph] = self._induce_subgraphs(
            top_k_paths, topic_entity, top_k
        )

        # take the union of the subgraphs, i.e., nodes and edges that appear at
        # least once in all subgraphs
        return self._merge_graphs(top_k_subgraphs, min_node_occurences=1)

    def forward(
        self,
        query: str,
        topic_entities: Sequence[str],
        beam_search_size: int = 3,
    ) -> nx.DiGraph:
        print("Question:", query)
        # Normalize the topic entities
        topic_entities = [
            similar_nodes[0]
            for similar_nodes in self._coder.node_normalization(topic_entities).values()
        ]

        # Retrieve the subgraphs starting from each topic entity
        retrieved_subgraphs: List[nx.DiGraph] = [
            self._retrieve_subgraphs_from_topic_entity(
                query, topic_entity, beam_search_size
            )
            for topic_entity in tqdm(topic_entities)
        ]

        # Return the intersection of all subgraphs
        return self._merge_graphs(retrieved_subgraphs)


if __name__ == "__main__":
    args = parse_arguments()
    # Load the MedQA dataset
    medqa_answers, medqa_questions, concept_names = load_medqa_dataset(
        os.path.join(args.medqa_folder, "train.statement.jsonl"),
        os.path.join(args.medqa_folder, "train.grounded.json"),
        os.path.join(args.medqa_folder, "concept_names.tsv"),
    )

    subgraph_retriever = SubgraphRetrieval(
        args.umls_triples_path,
        args.next_relation_predictor,
        args.coder_path,
        args.umls_embeddings_path,
        args.link_predictor,
        args.encoder_model_name_or_path,
    )

    question, question_topic_entities = extract_question_and_topic_entities(
        medqa_answers, medqa_questions, concept_names, args.question_idx
    )

    subgraph: nx.DiGraph = subgraph_retriever(question, question_topic_entities)
    print(subgraph)
