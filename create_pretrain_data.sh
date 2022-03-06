#!/bin/bash

VOCAB_PATH='./pretrained_ckpt/bert-base-uncased-vocab.txt'
FORMAT_DATA_DIR='./dataset/pretrain_data/format_data'

WIKI_NAME='wikipedia_en_format.txt'
WIKI_XML_PATH='./dataset/pretrain_data/download_wikipedia/enwiki-latest-pages-articles.xml'
EXTRACTED_WIKI_DIR='./dataset/pretrain_data/download_wikipedia/extracted_wikipedia'
FORMAT_WIKI_PATH='./dataset/pretrain_data/format_data/'$WIKI_NAME
WIKI_SAVE_DIR='./dataset/pretrain_data/wikipedia_nomask'

BOOK_NAME='bookcorpus_format.txt'
DOWNLOAD_BOOK_DIR='./dataset/pretrain_data/download_bookcorpus'
FORMAT_BOOK_PATH='./dataset/pretrain_data/format_data/'$BOOK_NAME
BOOK_SAVE_DIR='./dataset/pretrain_data/bookcorpus_nomask'


# Wikipedia
python pretrain_data_scripts/wiki_extractor.py \
  --input $WIKI_XML_PATH \
  -b '100M' \
  --processes 4 \
  -o $EXTRACTED_WIKI_DIR
python pretrain_data_scripts/text_formatting.py \
  --wiki_dir $EXTRACTED_WIKI_DIR \
  --wiki_name $WIKI_NAME \
  --output_dir $FORMAT_DATA_DIR
python pretrain_data_scripts/create_data.py \
  --train_corpus $FORMAT_WIKI_PATH \
  --output_dir $WIKI_SAVE_DIR \
  --vocab_path $VOCAB_PATH \
  --lowercase \
  --epochs_to_generate 5 \
  --max_seq_len 128 \
  --max_predictions_per_seq 0


# Bookscorpus
python pretrain_data_scripts/text_formatting.py \
  --book_dir $DOWNLOAD_BOOK_DIR \
  --book_name $BOOK_NAME \
  --output_dir $FORMAT_DATA_DIR \
python pretrain_data_scripts/create_data.py \
  --train_corpus $FORMAT_BOOK_PATH \
  --output_dir $BOOK_SAVE_DIR \
  --vocab_path $VOCAB_PATH \
  --lowercase \
  --epochs_to_generate 5 \
  --max_seq_len 128 \
  --max_predictions_per_seq 0
