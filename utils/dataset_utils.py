import torch
import torch.distributed as dist
import numpy as np
import logging
import json
import random
import datasets
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, BatchSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from collections import namedtuple

InputFeatures = namedtuple('InputFeatures', 'token_ids segment_ids position_ids attn_mask lm_label_ids is_next')


class PretrainDataset(Dataset):
    def __init__(self, all_data_dir, epoch, tokenizer, num_data_epochs, local_rank, split,
                 train_ratio=0.9, val_ratio=0.1, concate_data_dir=None, read_only=True):
        data_epoch = int(epoch % num_data_epochs)

        if not isinstance(all_data_dir, list) or (isinstance(all_data_dir, list) and len(all_data_dir) == 1):
            if not isinstance(all_data_dir, list):
                data_dir = all_data_dir
            else:
                data_dir = all_data_dir[0]

            num_samples, seq_len, token_ids, segment_ids, position_ids, attn_mask, lm_label_ids, is_nexts = \
                self.process_single_task_dataset(data_dir, data_epoch, tokenizer, split, local_rank, read_only)

            if split == 'train':
                self.num_samples = int(num_samples * train_ratio)
                assert self.num_samples > 0
                self.seq_len = seq_len
                self.token_ids = token_ids[:self.num_samples]
                self.segment_ids = segment_ids[:self.num_samples]
                self.position_ids = position_ids[:self.num_samples]
                self.attn_mask = attn_mask[:self.num_samples]
                self.lm_label_ids = lm_label_ids[:self.num_samples]
                self.is_nexts = is_nexts[:self.num_samples]
            else:
                self.num_samples = int(num_samples * val_ratio)
                assert 0 < self.num_samples <= int(num_samples * (1 - train_ratio)) + 1
                self.seq_len = seq_len
                self.token_ids = token_ids[-self.num_samples:]
                self.segment_ids = segment_ids[-self.num_samples:]
                self.position_ids = position_ids[-self.num_samples:]
                self.attn_mask = attn_mask[-self.num_samples:]
                self.lm_label_ids = lm_label_ids[-self.num_samples:]
                self.is_nexts = is_nexts[-self.num_samples:]
        else:
            total_num_samples = 0
            for data_dir in all_data_dir:
                metrics_file = data_dir / 'epoch_{}_metrics.json'.format(data_epoch)
                metrics = json.loads(metrics_file.read_text())
                cur_num_samples = metrics['num_training_examples']
                seq_len = metrics['max_seq_len']
                total_num_samples += cur_num_samples

            concate_save_dir = concate_data_dir / ('data_epoch_' + str(data_epoch))
            if concate_save_dir.exists():
                if local_rank == 0:
                    logging.info('Loading {} concate pretrain dataset from \'{}\''.format(split, concate_save_dir))

                concate_token_ids = np.memmap(filename=concate_save_dir / 'token_ids.memmap', shape=(total_num_samples, seq_len), mode='r', dtype=np.int32)
                concate_segment_ids = np.memmap(filename=concate_save_dir / 'segment_ids.memmap', shape=(total_num_samples, seq_len), mode='r', dtype=np.int32)
                concate_position_ids = np.memmap(filename=concate_save_dir / 'position_ids.memmap', shape=(total_num_samples, seq_len), mode='r', dtype=np.int32)
                concate_attn_mask = np.memmap(filename=concate_save_dir / 'attn_mask.memmap', shape=(total_num_samples, seq_len), mode='r', dtype=np.int32)
                concate_lm_label_ids = np.memmap(filename=concate_save_dir / 'lm_label_ids.memmap', shape=(total_num_samples, seq_len), mode='r', dtype=np.int32)
                concate_is_nexts = np.memmap(filename=concate_save_dir / 'is_nexts.memmap', shape=(total_num_samples,), mode='r', dtype=np.bool)
            else:
                if read_only:
                    raise RuntimeError('Pretrain dataset is currently read only, if the dataset has not been created yet, '
                                       'please run \'create_pretrain_feature.py\' to create it before training.')

                if local_rank == 0:
                    logging.info('Writing {} concate pretrain dataset to \'{}\''.format(split, concate_save_dir))

                concate_save_dir.mkdir(parents=True, exist_ok=True)
                concate_token_ids = np.memmap(filename=concate_save_dir / 'token_ids.memmap', shape=(total_num_samples, seq_len), mode='w+', dtype=np.int32)
                concate_segment_ids = np.memmap(filename=concate_save_dir / 'segment_ids.memmap', shape=(total_num_samples, seq_len), mode='w+', dtype=np.int32)
                concate_position_ids = np.memmap(filename=concate_save_dir / 'position_ids.memmap', shape=(total_num_samples, seq_len), mode='w+', dtype=np.int32)
                concate_attn_mask = np.memmap(filename=concate_save_dir / 'attn_mask.memmap', shape=(total_num_samples, seq_len), mode='w+', dtype=np.int32)
                concate_lm_label_ids = np.memmap(filename=concate_save_dir / 'lm_label_ids.memmap', shape=(total_num_samples, seq_len), mode='w+', dtype=np.int32)
                concate_is_nexts = np.memmap(filename=concate_save_dir / 'is_nexts.memmap', shape=(total_num_samples,), mode='w+', dtype=np.bool)

                start_idx = 0
                for data_dir in all_data_dir:
                    cur_num_samples, seq_len, token_ids, segment_ids, position_ids, attn_mask, lm_label_ids, is_nexts = \
                        self.process_single_task_dataset(data_dir, data_epoch, tokenizer, split, local_rank, read_only)
                    end_idx = start_idx + cur_num_samples

                    concate_token_ids[start_idx:end_idx, :] = token_ids
                    concate_segment_ids[start_idx:end_idx, :] = segment_ids
                    concate_position_ids[start_idx:end_idx, :] = position_ids
                    concate_attn_mask[start_idx:end_idx, :] = attn_mask
                    concate_lm_label_ids[start_idx:end_idx, :] = lm_label_ids
                    concate_is_nexts[start_idx:end_idx] = is_nexts
                    start_idx += cur_num_samples

            if split == 'train':
                self.num_samples = int(total_num_samples * train_ratio)
                assert self.num_samples > 0
                self.seq_len = seq_len
                self.token_ids = concate_token_ids[:self.num_samples]
                self.segment_ids = concate_segment_ids[:self.num_samples]
                self.position_ids = concate_position_ids[:self.num_samples]
                self.attn_mask = concate_attn_mask[:self.num_samples]
                self.lm_label_ids = concate_lm_label_ids[:self.num_samples]
                self.is_nexts = concate_is_nexts[:self.num_samples]
            else:
                self.num_samples = int(total_num_samples * val_ratio)
                assert 0 < self.num_samples <= int(total_num_samples * (1 - train_ratio)) + 1
                self.seq_len = seq_len
                self.token_ids = concate_token_ids[-self.num_samples:]
                self.segment_ids = concate_segment_ids[-self.num_samples:]
                self.position_ids = concate_position_ids[-self.num_samples:]
                self.attn_mask = concate_attn_mask[-self.num_samples:]
                self.lm_label_ids = concate_lm_label_ids[-self.num_samples:]
                self.is_nexts = concate_is_nexts[-self.num_samples:]

    @staticmethod
    def process_single_task_dataset(data_dir, data_epoch, tokenizer, split, local_rank, read_only):
        data_file = data_dir / 'epoch_{}.json'.format(data_epoch)
        metrics_file = data_dir / 'epoch_{}_metrics.json'.format(data_epoch)
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        seq_len = metrics['max_seq_len']

        save_dir = data_dir / ('data_epoch_' + str(data_epoch))
        if save_dir.exists():
            if local_rank == 0:
                logging.info('Loading {} pretrain dataset from \'{}\''.format(split, save_dir))

            token_ids = np.memmap(filename=save_dir / 'token_ids.memmap', shape=(num_samples, seq_len), mode='r', dtype=np.int32)
            segment_ids = np.memmap(filename=save_dir / 'segment_ids.memmap', shape=(num_samples, seq_len), mode='r', dtype=np.int32)
            position_ids = np.memmap(filename=save_dir / 'position_ids.memmap', shape=(num_samples, seq_len), mode='r', dtype=np.int32)
            attn_mask = np.memmap(filename=save_dir / 'attn_mask.memmap', shape=(num_samples, seq_len), mode='r', dtype=np.int32)
            lm_label_ids = np.memmap(filename=save_dir / 'lm_label_ids.memmap', shape=(num_samples, seq_len), mode='r', dtype=np.int32)
            is_nexts = np.memmap(filename=save_dir / 'is_nexts.memmap', shape=(num_samples,), mode='r', dtype=np.bool)
        else:
            if read_only:
                raise RuntimeError('Pretrain dataset is currently read only, if the dataset has not been created yet, '
                                   'please run \'create_pretrain_feature.py\' to create it before training.')

            save_dir.mkdir(parents=True, exist_ok=True)
            if local_rank == 0:
                logging.info('Writing {} pretrain dataset to \'{}\''.format(split, save_dir))

            token_ids = np.memmap(filename=save_dir / 'token_ids.memmap', shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            segment_ids = np.memmap(filename=save_dir / 'segment_ids.memmap', shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            position_ids = np.memmap(filename=save_dir / 'position_ids.memmap', shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            attn_mask = np.memmap(filename=save_dir / 'attn_mask.memmap', shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            lm_label_ids = np.memmap(filename=save_dir / 'lm_label_ids.memmap', shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            is_nexts = np.memmap(filename=save_dir / 'is_nexts.memmap', shape=(num_samples,), mode='w+', dtype=np.bool)

            lm_label_ids[:] = -1
            with data_file.open() as f:
                for i, line in enumerate(tqdm(f, total=num_samples, desc='Training examples')):
                    line = line.strip()
                    example = json.loads(line)
                    features = convert_example_to_features(example, tokenizer, seq_len)
                    token_ids[i] = features.token_ids
                    segment_ids[i] = features.segment_ids
                    position_ids[i] = features.position_ids
                    attn_mask[i] = features.attn_mask
                    lm_label_ids[i] = features.lm_label_ids
                    is_nexts[i] = features.is_next

        return num_samples, seq_len, token_ids, segment_ids, position_ids, attn_mask, lm_label_ids, is_nexts

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.token_ids[item], dtype=torch.long),
                torch.tensor(self.segment_ids[item], dtype=torch.long),
                torch.tensor(self.position_ids[item], dtype=torch.long),
                torch.tensor(self.attn_mask[item], dtype=torch.long),
                torch.tensor(self.lm_label_ids[item], dtype=torch.long))


class MultiTaskDataset(Dataset):
    def __init__(self, all_datasets):
        self.all_datasets = all_datasets

    def __len__(self):
        return sum(len(dataset) for dataset in self.all_datasets)

    def __getitem__(self, idx):
        task_id, sample_id = idx
        return self.all_datasets[task_id][sample_id]


class MultiTaskBatchSampler(BatchSampler):
    def __init__(self, all_datasets, batch_size, distributed=False, shuffle=True):
        self.all_datasets = all_datasets
        self.batch_size = batch_size
        self.distributed = distributed
        self.shuffle = shuffle
        self.batch_indices = self._gen_batch_indices(shuffle=False)
        if self.distributed:
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()

    def _gen_batch_indices(self, shuffle):
        all_batch_indices = []
        for dataset in self.all_datasets:
            batch_indices = [list(range(i, min(i + self.batch_size, len(dataset))))
                             for i in range(0, len(dataset), self.batch_size)]
            if shuffle:
                random.shuffle(batch_indices)
            all_batch_indices.append(batch_indices)
        return all_batch_indices

    def _gen_task_indices(self):
        all_indices = []
        for batch_task_id in range(len(self.batch_indices)):
            all_indices += [batch_task_id] * len(self.batch_indices[batch_task_id])
        if self.shuffle:
            random.shuffle(all_indices)
        return all_indices

    def __iter__(self):
        batch_indices = self._gen_batch_indices(self.shuffle)
        index_iters = [iter(item) for item in batch_indices]
        flatten_task_indices = self._gen_task_indices()
        for batch_task_id in flatten_task_indices:
            batch = next(index_iters[batch_task_id])
            samples = [(batch_task_id, sample_id) for sample_id in batch]

            select_samples = samples
            if self.distributed:
                while len(samples) % self.num_replicas != 0:
                    samples += samples[:(self.num_replicas - len(samples) % self.num_replicas)]
                select_samples = samples[self.rank::self.num_replicas]
                assert len(select_samples) == len(samples) // self.num_replicas
            yield select_samples

    def __len__(self):
        return sum(len(data) for data in self.batch_indices)


def convert_example_to_features(example, tokenizer, max_seq_length):
    tokens = example['tokens']
    segment_ids = example['segment_ids']
    is_random_next = example['is_random_next']
    masked_lm_positions = example['masked_lm_positions']
    masked_lm_labels = example['masked_lm_labels']

    if len(tokens) > max_seq_length:
        tokens = tokens[:max_seq_length]

    if len(tokens) != len(segment_ids):
        segment_ids = [0] * len(tokens)

    assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
    input_ids = tokenizer._tokens_to_ids(tokens)

    input_array = np.zeros(max_seq_length, dtype=np.int)
    input_array[:len(input_ids)] = input_ids

    # mask_array = np.zeros(max_seq_length, dtype=np.bool)
    # mask_array[:len(input_ids)] = 1
    mask_array = np.ones(max_seq_length, dtype=np.bool)
    mask_array[:len(input_ids)] = 0

    segment_array = np.zeros(max_seq_length, dtype=np.bool)
    segment_array[:len(segment_ids)] = segment_ids
    position_array = np.arange(max_seq_length, dtype=np.int)

    lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
    if masked_lm_positions is not None:
        masked_label_ids = tokenizer._tokens_to_ids(masked_lm_labels)
        lm_label_array[masked_lm_positions] = masked_label_ids

    features = InputFeatures(token_ids=input_array,
                             segment_ids=segment_array,
                             position_ids=position_array,
                             attn_mask=mask_array,
                             lm_label_ids=lm_label_array,
                             is_next=is_random_next)
    return features


def create_pretrain_dataset(data_dir, epoch, tokenizer, num_data_epochs, local_rank, batch_size, use_gpu,
                            distributed, split, train_ratio, val_ratio, concate_data_dir=None, read_only=True,
                            num_workers=4):
    dataset = PretrainDataset(
        data_dir, epoch, tokenizer, num_data_epochs, local_rank, split, train_ratio, val_ratio, concate_data_dir,
        read_only)
    loader = _create_dataset_loader(dataset, batch_size, use_gpu, distributed, split, num_workers)
    return dataset, loader


def create_dataset(model_name, task, data_dir, tokenizer, max_seq_len, max_query_len, trunc_stride, batch_size,
                   use_gpu, distributed, split, local_rank, cache_dir, num_workers=4, n_samples=None):
    if split == 'dev':
        if task == 'swag':
            split = 'val'
        elif task == 'race':
            split = 'test'

    if task in datasets.glue_tasks:
        examples, encoded_inputs, dataset = datasets.create_glue_dataset(
            model_name, task, data_dir, tokenizer, max_seq_len, split, local_rank, cache_dir)
    elif task in datasets.squad_tasks:
        examples, encoded_inputs, dataset = datasets.create_squad_dataset(
            model_name, task, data_dir, tokenizer, max_seq_len, max_query_len, trunc_stride, split, local_rank,
            cache_dir)
    elif task in datasets.multi_choice_tasks:
        examples, encoded_inputs, dataset = datasets.create_multi_choice_dataset(
            model_name, task, data_dir, tokenizer, max_seq_len, max_query_len, split, local_rank, cache_dir)
    elif task == 'cnn_dm':
        examples, encoded_inputs, dataset = datasets.create_cnn_dm_dataset(
            model_name, task, data_dir, tokenizer, max_seq_len, max_query_len, split, n_samples, local_rank, cache_dir)
    else:
        raise KeyError('task \'{}\' is not valid'.format(task))

    loader = _create_dataset_loader(dataset, batch_size, use_gpu, distributed, split, num_workers)
    return examples, encoded_inputs, dataset, loader


def create_multi_task_dataset(model_name, all_tasks, data_dir, tokenizer, max_seq_len, max_query_len, trunc_stride, batch_size,
                              train_ratio, val_ratio, use_gpu, distributed, split, local_rank, cache_dir, num_workers=4):
    examples, encoded_inputs, all_datasets = datasets.create_ensemble_glue_dataset(
        model_name, all_tasks, data_dir, tokenizer, max_seq_len, train_ratio, val_ratio, split, local_rank, cache_dir)

    if split == 'train':
        train_batch_size = batch_size * dist.get_world_size() if distributed else batch_size
        loader = _create_multi_task_dataset_loader(all_datasets, train_batch_size, use_gpu, distributed, num_workers)
    else:  # split == 'val' or split == 'dev'
        loader = [_create_dataset_loader(dataset, batch_size, use_gpu, distributed, split, num_workers) for dataset in all_datasets]
    return examples, encoded_inputs, all_datasets, loader


def create_split_dataset(model_name, task, data_dir, tokenizer, max_seq_len, max_query_len, trunc_stride, batch_size,
                         train_ratio, val_ratio, use_gpu, distributed, split, local_rank, cache_dir, num_workers=4):
    examples, encoded_inputs, dataset = datasets.create_split_glue_dataset(
        model_name, task, data_dir, tokenizer, max_seq_len, train_ratio, val_ratio, split, local_rank, cache_dir)
    loader = _create_dataset_loader(dataset, batch_size, use_gpu, distributed, split, num_workers)
    return examples, encoded_inputs, dataset, loader


def _create_dataset_loader(dataset, batch_size, use_gpu, distributed, split, num_workers=4):
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=True if split == 'train' else False)
    else:
        sampler = RandomSampler(dataset) if split == 'train' else SequentialSampler(dataset)

    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=use_gpu)
    return loader


def _create_multi_task_dataset_loader(all_datasets, batch_size, use_gpu, distributed, num_workers=4):
    batch_sampler = MultiTaskBatchSampler(all_datasets, batch_size, distributed)
    dataset = MultiTaskDataset(all_datasets)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=use_gpu)
    return loader
