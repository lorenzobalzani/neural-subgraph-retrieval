#!/bin/bash
python3.9 link_predictor.py \
  --umls_triples_path ../data/umls/UMLS_KG_triplets_sample.txt \
  --huggingface_hub_url neural-subgraph-retrieval/umls-link-predictor-pubmedbert \
  --encoder_model_name_or_path NeuML/pubmedbert-base-embeddings-matryoshka \
  --pykeen_train_config pykeen_config.json \
  --n_epochs 50 \
  --pykeen_checkpoint_file pubmedbert.pt
