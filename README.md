# EvidenceBench: A Benchmark for Extracting Evidence from Biomedical Papers

The test set is available under the CC-BY License. The training and development sets are available under the CC-BY-NC-SA License

## Paper abstract
We study the task of finding evidence for a hypothesis in the biomedical literature. Finding relevant evidence is a necessary precursor for evaluating the validity of scientific hypotheses, and for applications such as automated meta-analyses and scientifically grounded question-answering systems. We develop a pipeline for high quality, sentence-by-sentence annotation of biomedical papers for this task. The pipeline leverages expert judgments of scientific relevance, and is validated using teams of human annotators. We evaluate a diverse set of language models and retrieval systems on the benchmark, which consists of more than 400 fully annotated papers and 80k sentence judgments. The performance of the best models still falls significantly short of human-level on this task. By providing a standardized benchmark and evaluation framework, this work will support the development of tools which automate evidence synthesis and hypothesis testing. 

## Dataset Description:


EvidenceBench uses a train, dev, test split. All three subsets have the same structure. The nth data instance in the test set has a unique identifier `evidencebench_test_id_n` where n is a numerical number. Similarly for train and dev, the nth data instance is `evidencebench_train_id_n` and `evidencebench_dev_id_n`.

Each data instance has the following features, represented as JSON keys:
- `hypothesis`: string format, the main query, the biomedical hypothesis.
- `paper_as_candidate_pool`: an ordered tuple of strings. Each string is one sentence from the paper. Note, the tuple order matches the order of sentences in the paper.
- `aspect_list_ids`: a list of strings, each string is an aspect. Each aspect has the following format, `evidencebench_test_id_n_aspect_m` where n, m are integers.
- `results_aspect_list_ids`: a list of strings, each string is an aspect that is labeled as "Results", which means related to experimental outcomes and analyses.
- `aspect2sentence_indices`: a mapping (i.e. dictionary) between aspect and all sentence indices that independently are source of information for that aspect.
- `sentence_index2aspects`: a mapping (i.e. dictionary) between sentence index and all aspects that this sentence is the source of information of.
- `evidence_retrieval_at_optimal_evaluation`: This is a dictionary that contains the necessary information for evaluating your model's performance on the task Evidence Retrieval @Optimal.
  - `optimal`: A positive integer, which is the smallest number of sentences needed to cover the largest number of aspects.
  - `one_selection_of_sentences`: a list of sentence indices. The list size is the optimal number. The list of sentences covers the largest number of aspects. Note, there are potentially other lists of sentences that have the same size and also cover the largest number of aspects.
  - `covered_aspects`: the list of aspects that are covered. In this case, the list of aspects is all the aspects.
- `evidence_retrieval_at_10_evaluation`: This is a dictionary that contains the necessary information for evaluating your model's performance on the task Evidence Retrieval @10.
  - `one_selection_of_sentences`: a list of sentence indices. The list size is 10. The list of sentences covers the largest number of aspects that can be covered under the restriction of 10 sentences. Note, there are potentially other lists of sentences that have size 10 and cover the same number of aspects.
  - `covered_aspects`: the list of aspects that are covered. In this case, this list of aspects may not be all the aspects. Since in the paper, we calculate aspect recall by dividing the number of aspects covered by the model's retrieved sentences against the total number of aspects, for ER @10, the maximum possible performance is not 100%.
- `results_evidence_retrieval_at_optimal_evaluation`: This is a dictionary that contains the necessary information for evaluating a model's performance on the task Results Evidence Retrieval @Optimal.
  - `optimal`: A positive integer, which is the smallest number of sentences needed to cover the largest number of aspects labeled as "Results".
  - `one_selection_of_sentences`: see above
  - `covered_aspects`: the list of "Results" aspects that are covered. In this case, the list of "Results" aspects is all the "Results" aspects.
- `results_evidence_retrieval_at_5_evaluation`: This is a dictionary that contains the necessary information for evaluating your model's performance on the task Results Evidence Retrieval @5.
  - `one_selection_of_sentences`: see above
  - `covered_aspects`: see above. Note, for Results ER @5, the maximum possible performance is not 100%.
- `sentence_types_in_candidate_pool`: a tuple of strings, each string is a sentence type. There are three possible sentence types: section_name, abstract, and normal_paragraph. If the third string is 'abstract' that means the third sentence (sentence index =2) has sentence type 'abstract', i.e., it is a sentence that comes from the abstract.
- `paper_id`: the id of the paper used as the candidate pool.

Note, the train set and dev set have exactly the same structure. The dataset is available for download in this repository.


## Evaluation

We provide code for end-to-end evaluation on our benchmark using text embedding models or text generation models. The current pipeline supports all the embedding and generation models mentioned in our paper.



### Evaluation with Generation Models


To run the evaluation for generation models, use the following command:

```bash
cd Evaluation
bash end_to_end_eval.sh <dataset_path> <max_tokens> <prompt_template_name> <model_name> <exp_name> -1 <limits> <regeneration> False
```

Most of the arguments are same as the embedding model case.

`<prompt_template_name>` The prompt template to use for generation.

`<regeneration>` A boolean indicating whether the model will regenerate if it retrieves more than the specified number of sentences.

The results of both `embedding_pipeline.sh` and `end_to_end_eval.sh` will be recorded in `Evaluation/post_process/logs.csv` under the specified `exp_name`.


### Evaluation with Embedding Models

To run the end-to-end evluation for embeddding models, run the following:

```bash
cd Evaluation
bash embedding_pipeline.sh <dataset_path> <instruction_template_name> <model_name> <exp_name> <limits> <use_api> <cuda> <batch_size>
```

`<dataset_path>`: The path to the dataset to evaluate, typically located in the `dataset/` directory.

`<instruction_template_name>`: The instruction added before each query.

`<model_name>`: The name of the embedding model used for evaluation. Supported models include: `['e5', 'e5_mistral', 'voyage-large-2-instruct', 'text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002']`

`<exp_name>`: The name under which the evaluation results will be documented in the experiment logs.

`<limits>`: The limit used for calculating recall@limits during evaluation.

`<use_api>`: A boolean indicating whether the embedding model requires API calls. Set to True for voyage or OpenAI-related models.

`<cuda>`: A string specifying which CUDA devices to use for local embedding models, e.g., `"0,1,2"` or `"0"`.

`<batch_size>`: The batch size for local embedding models.

