# Neural Subgraph Retrieval

As a starting point, everything is supposed to be done inside a Docker container. Simply run the script `launch_container.sh` located inside the `docker` folder to build the image, if needed, and then launch the container.

## End-to-end Subgraph Retrieval System
The complete subgraph retriever will soon be made available.

## Separate Components
### Predicting Next Relations
#### Data
The dataset designed for next relation prediction is open-source and can be downloaded from this [link](https://huggingface.co/datasets/neural-subgraph-retrieval/umls-nrp-dataset) on the HuggingFace Hub.

#### Model
The model is open-source and accessible for download via this [link](https://huggingface.co/neural-subgraph-retrieval/umls-nrp-BioLinkBERT-base) on the HuggingFace Hub.

### Link Prediction
#### Model
The model is open-source and accessible for download via this [link](https://huggingface.co/neural-subgraph-retrieval/umls-link-predictor) on the HuggingFace Hub.
