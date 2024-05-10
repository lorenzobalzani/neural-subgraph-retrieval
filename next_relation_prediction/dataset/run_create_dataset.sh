#!/bin/bash
python3.9 main.py \
  --medqa_folder ../../data/medqa \
  --umls_triples_path ../../data/umls/UMLS_KG_triplets_sample.txt \
  --umls_embeddings_path ../../data/umls/umls_coder_embeddings_float32.pt \
  --coder_path balzanilo/UMLSBert_ENG \
  --output_path ../../data/paths_dataset.tsv \
  --huggingface_hub_url neural-subgraph-retrieval/umls-nrp-dataset \
  --stats_folder ../../data/umls-next-relation-prediction
