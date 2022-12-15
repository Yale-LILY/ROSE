# ROSE 🌹

This repo contiains the RoSE benchmark of our paper "Revisiting the Gold Standard:
Grounding Summarization Evaluation with Robust Human Evaluation".

Please visit here for a demo [page](https://yale-lily.github.io/ROSE/) of this project.


## RoSE 🌹 Benchmark

RoSE can be downloaded with Huggin Face Datasets under `Salesforce/rose`.
We provide a notebook, [demo.ipynb](demo.ipynb), for basic usage of our dataset.

### ACU Annotations

RoSE benchmark contains system outputs annotated with our ACU protocol. 
It contains four parts:
- CNNDM, test set annotations
- CNNDM, validation set annotations
- XSum, test set annotations
- SamSum, test set annotations

We summarize the statistics below.

| Dataset | Split | #Doc. | #Sys. | #Total Summ. | HF Name
| --- | --- | --- | --- | --- | --- |
| CNNDM | Test | 500 | 12 | 6000 | `cnndm_test` |
| CNNDM | Validation | 1000 | 8 | 8000 | `cnndm_validation` |
| XSum  | Test | 500 | 8 | 4000 | `xsum` |
| SamSum  | Test | 500 | 8 | 4000 | `samsum` |

###  Human Annotations with Different Evalaution Protocols

We have system outputs annotated with four different human evaluation protocols in total.
We summarize them below.

| Protocol | w/ Input Document | w/ Reference Summary | Fine-grained |
| --- | --- | --- | --- |
| Prior |  ✗ | ✗ | ✗ | 
| Ref-free | ✓ | ✗ | ✗ |
| Ref-based | ✗ | ✓ | ✗ |
| ACU | ✗ | ✓ | ✓ |

We annotated two sets of system summaries.

1. Summaries of 12 fine-tuned systems. The huggingface data split name is `cnndm_protocol`.
2. Zero-shot summaries from large langauge models (GPT3, T0), together with summaries from BRIO and BART. The huggingface data split name is `cnndm_protocol_gpt3`.
















