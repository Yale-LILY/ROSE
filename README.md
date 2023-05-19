# ROSE 🌹

This repo contiains the RoSE benchmark of our paper ["Revisiting the Gold Standard:
Grounding Summarization Evaluation with Robust Human Evaluation"](https://arxiv.org/abs/2212.07981).

**Please visit [here](https://yale-lily.github.io/ROSE/) for a demo page of this project.**


## RoSE 🌹 Benchmark

RoSE can be downloaded with Hugging Face Datasets under [`Salesforce/rose`](https://huggingface.co/datasets/Salesforce/rose).
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

###  Human Annotations with Different Evaluation Protocols

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

## Citation

Please cite our paper if you use RoSE in your work:
```bibtex
@inproceedings{Liu2022RevisitingTG,
  title={Revisiting the Gold Standard: Grounding Summarization Evaluation with Robust Human Evaluation},
  author={Yixin Liu and Alexander R. Fabbri and Pengfei Liu and Yilun Zhao and Linyong Nan and Ruilin Han and Simeng Han and Shafiq R. Joty and Chien-Sheng Wu and Caiming Xiong and Dragomir R. Radev},
  booktitle={Proceedings of the 61th Annual Meeting of the Association for Computational Linguistics},
  year={2023},
}
```
















