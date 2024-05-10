#!/bin/bash
python3.9 main.py \
  --medqa_folder ../data/medqa \
  --umls_triples_path ../data/umls/UMLS_KG_triplets_sample.txt \
  --umls_embeddings_path ../data/umls/umls_coder_embeddings_float32.pt \
  --coder_path balzanilo/UMLSBert_ENG \
  --next_relation_predictor neural-subgraph-retrieval/umls-nrp-BioLinkBERT-base \
  --link_predictor neural-subgraph-retrieval/umls-link-predictor-pubmedbert \
  --encoder_model_name_or_path NeuML/pubmedbert-base-embeddings-matryoshka \
  --question_idx 0
