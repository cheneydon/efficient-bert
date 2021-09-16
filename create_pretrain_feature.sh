#!/bin/bash

VOCAB_PATH='./pretrained_ckpt/bert-base-uncased-vocab.txt'
WIKI_DIR='./dataset/pretrain_data/wikipedia_nomask'
BOOK_DIR='./dataset/pretrain_data/bookcorpus_nomask'
CONCATE_DATA_DIR='./dataset/pretrain_data/wiki_book_nomask'


# Wikipedia only
python create_pretrain_feature.py --lowercase --vocab_path $VOCAB_PATH --wiki_dir $WIKI_DIR

# Wikipedia + BooksCorpus
python create_pretrain_feature.py --lowercase --vocab_path $VOCAB_PATH --wiki_dir $WIKI_DIR --book_dir $BOOK_DIR --concate_data_dir $CONCATE_DATA_DIR
