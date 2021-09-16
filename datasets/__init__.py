from .glue import create_glue_examples, create_glue_dataset, create_ensemble_glue_dataset, create_split_glue_dataset
from .squad import SquadResult, create_squad_examples, create_squad_dataset
from .multi_choice import create_multi_choice_examples, create_multi_choice_dataset
from .cnn_dm import create_cnn_dm_examples, create_cnn_dm_dataset, END_TOKENS

glue_tasks = ['mnli', 'mnli-mm', 'qqp', 'qnli', 'sst-2', 'cola', 'sts-b', 'mrpc', 'rte', 'wnli', 'ax']
squad_tasks = ['squad1.1', 'squad2.0']
multi_choice_tasks = ['swag', 'race']
summarization_tasks = ['cnn_dm', 'xsum']
all_tasks = glue_tasks + squad_tasks + multi_choice_tasks + summarization_tasks

glue_train_tasks = ['mnli', 'qqp', 'qnli', 'sst-2', 'cola', 'sts-b', 'mrpc', 'rte']
glue_train_tasks_to_ids = {k: v for v, k in enumerate(glue_train_tasks)}
glue_train_ids_to_tasks = {k: v for k, v in enumerate(glue_train_tasks)}

glue_labels = {
    'mrpc': ['0', '1'],
    'mnli': ['contradiction', 'entailment', 'neutral'],
    'mnli-mm': ['contradiction', 'entailment', 'neutral'],
    'ax': ['contradiction', 'entailment', 'neutral'],
    'cola': ['0', '1'],
    'sst-2': ['0', '1'],
    'sts-b': [None],
    'qqp': ['0', '1'],
    'qnli': ['entailment', 'not_entailment'],
    'rte': ['entailment', 'not_entailment'],
    'wnli': ['0', '1']
}

glue_num_classes = {
    'mrpc': 2,
    'mnli': 3,
    'mnli-mm': 3,
    'ax': 3,
    'cola': 2,
    'sst-2': 2,
    'sts-b': 1,
    'qqp': 2,
    'qnli': 2,
    'rte': 2,
    'wnli': 2
}