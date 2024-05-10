from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import transformers


class Coder:
    def __init__(
        self,
        model_name: str,
        all_nodes: List[str],
        device: torch.device,
        print_specs: bool = False,
    ) -> None:
        self.__umls_coder_embeddings = None
        self.__coder_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.__device = device
        self.__coder = transformers.AutoModel.from_pretrained(model_name).to(device)
        self.__coder.output_hidden_states = False
        self.__all_nodes = all_nodes
        self.__embeddings_cache: Dict[str, List[str]] = {}

        # Same number of parameters of PubMedBERT ("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
        # i.e., they release only the fine-tuned BERT model after deep metric
        # learning
        if print_specs:
            print(self.__coder)
            print(f"{model_name} parameters: {self.__coder.num_parameters():_}")

    def __matrix_cos_sim(
        self, queries: torch.FloatTensor, keys: torch.FloatTensor, dim: int = 1
    ) -> torch.Tensor:
        # normalize by the 2-norm
        queries_norm: torch.FloatTensor = torch.nn.functional.normalize(
            queries, p=2, dim=dim
        ).to(self.__device)
        keys_norm: torch.FloatTensor = torch.nn.functional.normalize(
            keys, p=2, dim=dim
        ).to(self.__device)
        return torch.mm(queries_norm, keys_norm.transpose(0, 1))

    # Best CODER results are with [CLS] representations and normalization
    # (default)
    def __get_coder_embeddings(
        self,
        phrase_list: List[str],
        batch_size: int = 64,
        max_length: int = 32,
        normalize: bool = True,
        summary_method: str = "CLS",
    ) -> torch.FloatTensor:
        # TOKENIZATION
        text_input: Dict[str, List[int]] = {"input_ids": [], "attention_mask": []}
        for phrase in phrase_list:
            # (1) Tokenize the sentence.
            # (2) Prepend the `[CLS]` token to the start.
            # (3) Append the `[SEP]` token to the end.
            # (4) Map tokens to their IDs.
            # (5) Pad or truncate the sentence to `max_length`
            # (6) Create attention masks for [PAD] tokens.
            tokenized_input = self.__coder_tokenizer(
                phrase,
                max_length=max_length,  # UMLS terms are short
                add_special_tokens=True,
                truncation=True,
                padding="max_length",
            )
            text_input["input_ids"].append(tokenized_input["input_ids"])
            text_input["attention_mask"].append(tokenized_input["attention_mask"])

        # INFERENCE MODE ON
        self.__coder.eval()

        # COMPUTE EMBEDDINGS ACCORDING TO THE SPECIFIED BATCH-SIZE
        # (e.g., max_length=32, batch_size=64 --> 2 phrase embeddings at a time)

        count = len(text_input["input_ids"])  # n total tokens

        now_count: int = 0
        with torch.no_grad():
            while now_count < count:
                batch_input_ids = torch.LongTensor(
                    text_input["input_ids"][
                        now_count : min(now_count + batch_size, count)
                    ]
                ).to(self.__device)
                batch_attention_mask = torch.LongTensor(
                    text_input["attention_mask"][
                        now_count : min(now_count + batch_size, count)
                    ]
                ).to(self.__device)
                if summary_method == "CLS":
                    embed = self.__coder(
                        input_ids=batch_input_ids, attention_mask=batch_attention_mask
                    )[1]
                if summary_method == "MEAN":
                    embed = torch.mean(
                        self.__coder(
                            input_ids=batch_input_ids,
                            attention_mask=batch_attention_mask,
                        )[0],
                        dim=1,
                    )
                if normalize:
                    embed_norm = torch.linalg.matrix_norm(embed, ord=2, keepdim=True  # pylint: disable=not-callable
                                                          ).clamp(min=1e-12)
                embed = embed / embed_norm
                # Move embedding on CPU and convert it to a numpy array
                embed_np = embed.cpu().detach().numpy()
                # Update indexes for batch processing
                if now_count == 0:
                    output = embed_np
                else:
                    output = np.concatenate((output, embed_np), axis=0)
                now_count = min(now_count + batch_size, count)

        return torch.FloatTensor(output)  # 32-bit tensor

    def create_umls_embeddings(
        self,
        save_tensor_path: Optional[str] = None,
        float_16: bool = False,
        subsample: bool = False,
        subsample_ratio: float = 0.05,
    ) -> torch.FloatTensor:
        if subsample:
            until_idx: int = int(len(self.__all_nodes) * subsample_ratio)
            nodes = self.__all_nodes[:until_idx]
        else:
            nodes = self.__all_nodes
        self.__umls_coder_embeddings = self.__get_coder_embeddings(nodes)
        if float_16:
            self.__umls_coder_embeddings = self.__umls_coder_embeddings.to(
                torch.float16
            )
        if save_tensor_path:
            torch.save(self.__umls_coder_embeddings, save_tensor_path)
        self.__umls_coder_embeddings.to(self.__device)
        return self.__umls_coder_embeddings

    def load_umls_embeddings(self, tensor_path: str) -> None:
        self.__umls_coder_embeddings = torch.load(
            tensor_path, map_location=torch.device(self.__device)
        )

    def node_normalization(
        self, nodes: Sequence[str], k: int = 1
    ) -> dict[str, list[str]]:
        normalization_results: Dict[str, List[str]] = {}

        for node in nodes:
            if node in self.__embeddings_cache.keys():
                cached_results: List[str] = self.__embeddings_cache[node]
                until_k: int = min(len(cached_results), k)
                normalization_results[node] = cached_results[:until_k]
            else:
                node_feat = self.__get_coder_embeddings([node])  # (1, 768)
                # Compute the cosine similarity with all terms
                cos_sim_mat = self.__matrix_cos_sim(
                    node_feat, self.__umls_coder_embeddings
                )
                # Get the index of the top-k terms with maximum similarity
                # (descending order)
                pred_top_k = torch.topk(cos_sim_mat, k=k).indices.cpu()
                if pred_top_k.shape[1] > 1:
                    pred_top_k = pred_top_k.squeeze()
                normalization_results[node] = [
                    self.__all_nodes[idx] for idx in pred_top_k
                ]
                self.__embeddings_cache[node] = normalization_results[node]

        return normalization_results
