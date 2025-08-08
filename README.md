<div align="center">
<h1>EvidenceBench: A Benchmark for Extracting Evidence from Biomedical Papers (COLM 2025)</h1>

<p style="font-family: serif; font-size: 16px; line-height: 1.4;">
  <b>Jianyou Wang<sup>1*</sup>, Weili Cao<sup>1*</sup>, Kaicheng Wang<sup>1</sup>, Xiaoyue Wang<sup>1</sup>, Ashish Dalvi<sup>1</sup>, Gino Prasad<sup>1</sup>, Qishan Liang<sup>3</sup>, Hsuan-lin Her<sup>3</sup>, Ming Wang<sup>4</sup>, Qin Yang<sup>5</sup>, Gene W. Yeo<sup>3</sup>, David E. Neal<sup>2</sup>, Maxim Khan<sup>2</sup>, Christopher D. Rosin<sup>2</sup>, Ramamohan Paturi<sup>1</sup>, Leon Bergen<sup>1</sup></b>
</p>

<p class="affils">
  <sup>1</sup>Laboratory for Emerging Intelligence, University of California, San Diego<br>
  <sup>2</sup>Elsevier<br>
  <sup>3</sup>Department of Cellular and Molecular Medicine, University of California, San Diego<br>
  <sup>4</sup>Sichuan Cancer Hospital &amp; Institute<br>
  <sup>5</sup>The Third People’s Hospital of Chengdu
</p>

[![COLM 2025](https://img.shields.io/badge/COLM-2025-purple.svg)](https://colmweb.org/)&nbsp; [![arXiv](https://img.shields.io/badge/arXiv-2504.18736-<COLOR>.svg)](https://arxiv.org/abs/2504.18736)

</div>

## Table of Contents
* [Paper abstract](#paper-abstract)
* [Dataset Description](#dataset-description)
  * [Original EvidenceBench](#original-evidencebench)
  * [EvidenceBench-100k](#evidencebench-100k)
* [Evaluation](#evaluation)
  * [Evaluation with Embedding Models](#evaluation-with-embedding-models)
  * [Evaluation with Generation Models](#evaluation-with-generation-models)
* [License](#license)

## Paper abstract
We study the task of finding evidence for a hypothesis in the biomedical literature. Finding relevant evidence is a necessary precursor for evaluating the validity of scientific hypotheses, and for applications such as automated meta-analyses and scientifically grounded question-answering systems. To this end, we present EvidenceBench, an open-sourced, comprehensive and large scale dataset of over 100,000 datapoints designed for evaluating and fine-tuning models' hypothesis understanding and evidence retrieval ability. Each datapoint is a biomedical paper from over 200 publication venues, covering diverse topics such as cardiology, neurology, infectious disease, public health, and nutrition. EvidenceBench is made from a novel and scalable LLM-based pipeline that can quickly generate expert-level quality, sentence-by-sentence annotation of biomedical papers, where sentences important for hypotheses are highlighted. Our pipeline generated over 150 million sentence judgments under 24 hours. Our pipeline's quality is validated using teams of human expert annotators. We evaluate a diverse set of language models and retrieval systems on EvidenceBench. The performance of the best models still falls significantly short of expert-level on this task. Our fine-tuned E5 embedding model (335M) and Llama3 language model (8B) show significant improvements over their baselines and achieve performance comparable with some Large Language Models (Claude3, Gemini1.5), though still lags behind GPT4.  EvidenceBench will support the development of tools which automate evidence synthesis and hypothesis testing, as well as the long-context global reasoning and instruction-following abilities for Large Language Models (LLM) and embedding-based IR systems. 

## Dataset Description

### Original EvidenceBench

The original EvidenceBench consists of 426 datapoints created from International Agency for Research on Cancer (IARC) monograph. The dataset has a train, dev, test split of (96, 37, 293) points. These three datasets are placed under `datasets` folder as `evidencebench_train_set.json`, `evidencebench_dev_set.json`, `evidencebench_test_set.json`. All three subsets have the same structure. The nth data instance in the test set has a unique identifier `evidencebench_test_id_n` where n is a numerical number. Similarly for train and dev, the nth data instance is `evidencebench_train_id_n` and `evidencebench_dev_id_n`.

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

### EvidenceBench-100k

EvidenceBench-100k is a larger dataset of 107,461 datapoints created from biomedical systematic reviews. The dataset has a train, test split of (87,461, 20,000) points. These two datasets are available at [huggingface](https://huggingface.co/datasets/EvidenceBench/EvidenceBench-100k) as `evidencebench_100k_train_set.json`, `evidencebench_100k_test_set.json`. 

We highly recommend you to download and place the downloaded datasets into the `datasets` folder using the following commands:
```
git clone https://huggingface.co/datasets/EvidenceBench/EvidenceBench-100k
cp -r EvidenceBench-100k/*.json datasets/
rm -r EvidenceBench-100k
```

Both subsets have the same structure as described above.



## Evaluation

We provide code for end-to-end evaluation on our benchmark using text embedding models or text generation models. The current pipeline supports all the embedding and generation models mentioned in our paper.

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

`<limits>`: The limit used for calculating recall@limits during evaluation. **Set to -1 for calculating recall@optimal**.

`<use_api>`: A boolean indicating whether the embedding model requires API calls. Set to True for voyage or OpenAI-related models.

`<cuda>`: A string specifying which CUDA devices to use for local embedding models, e.g., `"0,1,2"` or `"0"`.

`<batch_size>`: The batch size for local embedding models.

We record pids of the running scripts under folder `logs`. You could use ```kill <pid>``` to stop them.

### Evaluation with Generation Models


To run the evaluation for generation models, use the following command:

```bash
cd Evaluation
bash end_to_end_eval.sh <dataset_path> <max_tokens> <prompt_template_name> <model_name> <exp_name> <limits> <regeneration>
```

Most of the arguments are same as the embedding model case.

`<prompt_template_name>` The prompt template to use for generation.

`<regeneration>` A boolean indicating whether the model will regenerate if it retrieves more than the specified number of sentences.

For example, you can test gpt-4o on the baseline prompt using

``` bash end_to_end_eval.sh ../datasets/evidencebench_test_set.json 2048 original_final gpt-4o-2024-05-13 test_experiment_1 -1 True ```

The results of both `embedding_pipeline.sh` and `end_to_end_eval.sh` will be recorded in `Evaluation/post_process/logs.csv` under the specified `exp_name`.

## License

For the original EvidenceBench dataset: The test set is available under the CC-BY License. The training and development sets are available under the CC-BY-NC-SA License.

The EvidenceBench-100k dataset is available under the CC-BY-NC License.
