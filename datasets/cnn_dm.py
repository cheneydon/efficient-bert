import os
import torch
import logging
import hashlib
import models
from tqdm import tqdm
from torch.utils.data.dataset import TensorDataset


DM_SINGLE_CLOSE_QUOTE = "\u2019"  # unicode
DM_DOUBLE_CLOSE_QUOTE = "\u201d"
# acceptable ways to end a sentence
END_TOKENS = [".", "!", "?", "...", "'", "`", '"', DM_SINGLE_CLOSE_QUOTE, DM_DOUBLE_CLOSE_QUOTE, ")"]


def _read_text_file(text_file):
    lines = []
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def _get_url_hashes(path):
    """Get hashes of urls in file."""
    urls = _read_text_file(path)

    def url_hash(u):
        h = hashlib.sha1()
        try:
            u = u.encode("utf-8")
        except UnicodeDecodeError:
            logging.error("Cannot hash url: %s", u)
        h.update(u)
        return h.hexdigest()

    return {url_hash(u): True for u in urls}


def _get_hash_from_path(p):
    """Extract hash from path."""
    basename = os.path.basename(p)
    return basename[0 : basename.find(".story")]


def _find_files(data_dir, publisher, url_dict):
    """Find files corresponding to urls."""
    if publisher == "cnn":
        top_dir = os.path.join(data_dir, "cnn", "stories")
    elif publisher == "dm":
        top_dir = os.path.join(data_dir, "dailymail", "stories")
    else:
        logging.fatal("Unsupported publisher: %s", publisher)
    files = sorted(os.listdir(top_dir))

    ret_files = []
    for p in files:
        if _get_hash_from_path(p) in url_dict:
            ret_files.append(os.path.join(top_dir, p))
    return ret_files


def _subset_filenames(data_dir, split):
    """Get filenames for a particular split."""
    if split == 'train':
        urls = _get_url_hashes(os.path.join(data_dir, 'all_train.txt'))
    elif split == 'val':
        urls = _get_url_hashes(os.path.join(data_dir, 'all_val.txt'))
    elif split == 'test':
        urls = _get_url_hashes(os.path.join(data_dir, 'all_test.txt'))
    else:
        logging.fatal("Unsupported split: %s", split)
    cnn = _find_files(data_dir, "cnn", urls)
    dm = _find_files(data_dir, "dm", urls)
    return cnn + dm


def _get_art_abs(story_file):
    """Get abstract (highlights) and article from a story file path."""
    # Based on https://github.com/abisee/cnn-dailymail/blob/master/
    #     make_datafiles.py

    lines = _read_text_file(story_file)

    # The github code lowercase the text and we removed it in 3.0.0.

    # Put periods on the ends of lines that are missing them
    # (this is a problem in the dataset because many image captions don't end in
    # periods; consequently they end up in the body of the article as run-on
    # sentences)
    def fix_missing_period(line):
        """Adds a period to a line that is missing a period."""
        if "@highlight" in line:
            return line
        if not line:
            return line
        if line[-1] in END_TOKENS:
            return line
        return line + " ."

    lines = [fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for line in lines:
        if not line:
            continue  # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    # Make article into a single string
    article = " ".join(article_lines)
    abstract = " ".join(highlights)

    return article, abstract


def create_cnn_dm_examples(data_dir, split, local_rank):
    filenames = _subset_filenames(data_dir, split)
    all_articles, all_highlights = [], []
    for p in tqdm(filenames, total=len(filenames), disable=local_rank != 0):
        article, highlights = _get_art_abs(p)
        if not article or not highlights:
            continue
        all_articles.append(article)
        all_highlights.append(highlights)
    return all_articles, all_highlights


def create_cnn_dm_dataset(model_name, task, data_dir, tokenizer, max_seq_len, max_query_len, split, n_samples,
                          local_rank, cache_dir=''):
    if model_name in models.gpt_models:
        model = 'gpt'
    else:
        raise KeyError('model name \'{}\' is not valid'.format(model_name))

    cache_file = os.path.join(cache_dir, 'cnn_dm', '_'.join([model, task, split, str(max_seq_len), str(max_query_len)]))
    if tokenizer.lowercase:
        cache_file = os.path.join(cache_dir, 'cnn_dm', '_'.join([model, task, split, str(max_seq_len), str(max_query_len), 'lowercase']))

    if os.path.exists(cache_file):
        if local_rank == 0:
            logging.info('Loading {} cache file from \'{}\''.format(split, cache_file))
        cache_data = torch.load(cache_file)
        examples = cache_data['examples']
        encoded_inputs = cache_data['encoded_inputs']
    else:
        if local_rank == 0:
            logging.info('Creating {} examples'.format(split))
        examples = create_cnn_dm_examples(data_dir, split, local_rank)
        articles, highlights = examples

        if local_rank == 0:
            logging.info('Creating {} dataset'.format(split))
        encoded_inputs = [tokenizer.encode(text_a, text_b)
                          for text_a, text_b in tqdm(zip(articles, highlights), total=len(articles), disable=local_rank != 0)]

        if cache_dir and local_rank == 0:
            logging.info('Saving {} cache file to \'{}\''.format(split, cache_file))
            cache_dir = os.path.join(cache_dir, 'cnn_dm')
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            torch.save({'examples': examples, 'encoded_inputs': encoded_inputs}, cache_file)

    token_ids = torch.tensor([inp.token_ids for inp in encoded_inputs], dtype=torch.long)
    position_ids = torch.tensor([inp.position_ids for inp in encoded_inputs], dtype=torch.long)
    attn_mask = torch.tensor([inp.attn_mask for inp in encoded_inputs], dtype=torch.long)

    if n_samples is not None:
        token_ids = token_ids[:n_samples]
        position_ids = position_ids[:n_samples]
        attn_mask = attn_mask[:n_samples]

    dataset = TensorDataset(token_ids, position_ids, attn_mask)
    return examples, encoded_inputs, dataset
