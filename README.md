# EvidenceBench: A Benchmark for Extracting Evidence from Biomedical Papers

The test set is available under the CC-BY License. The training and development sets are available under the CC-BY-NC-SA License

## Paper abstract
We study the task of finding evidence for a hypothesis in the biomedical literature. Finding relevant evidence is a necessary precursor for evaluating the validity of scientific hypotheses, and for applications such as automated meta-analyses and scientifically grounded question-answering systems. We develop a pipeline for high quality, sentence-by-sentence annotation of biomedical papers for this task. The pipeline leverages expert judgments of scientific relevance, and is validated using teams of human annotators. We evaluate a diverse set of language models and retrieval systems on the benchmark, which consists of more than 400 fully annotated papers and 80k sentence judgments. The performance of the best models still falls significantly short of human-level on this task. By providing a standardized benchmark and evaluation framework, this work will support the development of tools which automate evidence synthesis and hypothesis testing. 

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

`<limits>`: The limit used for calculating recall@limits during evaluation.

`<use_api>`: A boolean indicating whether the embedding model requires API calls. Set to True for voyage or OpenAI-related models.

`<cuda>`: A string specifying which CUDA devices to use for local embedding models, e.g., `"0,1,2"` or `"0"`.

`<batch_size>`: The batch size for local embedding models.


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

