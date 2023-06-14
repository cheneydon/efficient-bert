# EfficientBERT: Progressively Searching Multilayer Perceptron via Warm-up Knowledge Distillation
This repository contains the code for the paper in Findings of EMNLP 2021: ["EfficientBERT: Progressively Searching Multilayer Perceptron via Warm-up Knowledge Distillation"](https://aclanthology.org/2021.findings-emnlp.123).


## Requirements
```shell
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

pip install -r requirements.txt
```

## Download checkpoints
Download the vocabulary file of BERT-base (uncased) from [HERE](https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt), and put it into `./pretrained_ckpt/`.  
Download the pre-trained checkpoint of BERT-base (uncased) from [HERE](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin), and put it into `./pretrained_ckpt/`.  
Download the 2nd general distillation checkpoint of TinyBERT from [HERE](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT), and extract them into `./pretrained_ckpt/`.


## Prepare dataset
Download the latest dump of Wikipedia from [HERE](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2), and extract it into `./dataset/pretrain_data/download_wikipedia/`.  
Download a mirror of BooksCorpus from [HERE](https://t.co/lww3BGREp7?amp=1), and extract it into `./dataset/pretrain_data/download_bookcorpus/`.

### - Pre-training data
```shell
bash create_pretrain_data.sh
bash create_pretrain_feature.sh
```
The features of Wikipedia, BooksCorpus, and their concatenation will be saved into `./dataset/pretrain_data/wikipedia_nomask/`,
`./dataset/pretrain_data/bookcorpus_nomask/`, and `./dataset/pretrain_data/wiki_book_nomask/`, respectively.

### - Fine-tuning data
Download the GLUE dataset using the script in [HERE](https://github.com/nyu-mll/GLUE-baselines/blob/master/download_glue_data.py), and put the files into `./dataset/glue/`.  
Download the SQuAD v1.1 and v2.0 datasets from the following links:  
- [squad-v1.1-train](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
- [squad-v1.1-dev](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
- [squad-v2.0-train](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json)
- [squad-v2.0-dev](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json)

and put them into `./dataset/squad/`.


## Pre-train the supernet
```shell
bash pretrain_supernet.sh
```
The checkpoints will be saved into `./exp/pretrain/supernet/`, 
and the names of the sub-directories should be modified into `stage1_2` and `stage3` correspondingly.

We also provide the checkpoint of the supernet in stage 3 (pre-trained with both Wikipedia and BooksCorpus) 
at [HERE](https://drive.google.com/file/d/15DKJ61ulrrjEvBDhKTRu3r4eJz9shAGa/view?usp=sharing). 


## Train the teacher model (BERT-base)
```shell
bash train.sh
```
The checkpoints will be saved into `./exp/train/bert_base/`, 
and the names of the sub-directories should be modified into the corresponding task name
(i.e., `mnli`, `qqp`, `qnli`, `sst-2`, `cola`, `sts-b`, `mrpc`, `rte`, `wnli`, `squad1.1`, and `squad2.0`). 
Each sub-directory contains a checkpoint named `best_model.bin`.


## Conduct NAS (including search stage 1, 2, and 3)
```shell
bash ffn_search.sh
```
The checkpoints will be saved into `./exp/ffn_search/`.


## Distill the student model

### - TinyBERT-4, TinyBERT-6
```shell
bash finetune.sh
```
The checkpoints will be saved into `./exp/downstream/tiny_bert/`.


### - EfficientBERT-tiny, EfficientBERT, EfficientBERT+, EfficientBERT++
```shell
bash nas_finetune.sh
```
The above script will first pre-train the student models based on the pre-trained checkpoint of the supernet in stage 3, 
and save the pre-trained checkpoints into `./exp/pretrain/auto_bert/`. 
Then fine-tune it on the downstream datasets,
and save the fine-tuned checkpoints into `./exp/downstream/auto_bert/`.

We also provide the pre-trained checkpoints of the student models 
(including EfficientBERT-TINY, EfficientBERT, and EfficientBERT++) 
at [HERE](https://drive.google.com/file/d/15DKJ61ulrrjEvBDhKTRu3r4eJz9shAGa/view?usp=sharing).


### - EfficientBERT (TinyBERT-6)
```shell
bash nas_finetune_transfer.sh
```
The pre-trained and fine-tuned checkpoints will be saved into 
`./exp/pretrain/auto_tiny_bert/` and `./exp/downstream/auto_tiny_bert/`, respectively.


## Test on the GLUE dataset
```shell
bash test.sh
```
The test results will be saved into `./test_results/`.


## Reference
If you find this code helpful for your research, please cite the following paper.
```BibTex
@inproceedings{dong2021efficient-bert,
  title     = {{E}fficient{BERT}: Progressively Searching Multilayer Perceptron via Warm-up Knowledge Distillation},
  author    = {Chenhe Dong and Guangrun Wang and Hang Xu and Jiefeng Peng and Xiaozhe Ren and Xiaodan Liang},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2021},
  year      = {2021}
}
```
