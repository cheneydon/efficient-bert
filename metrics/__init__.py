from .glue import simple_accuracy, acc_and_f1, pearson_and_spearman, compute_glue_metrics
from .squad import compute_squad_metrics

all_glue_select_metrics = {
    'mnli': 'acc',
    'mnli-mm': 'acc',
    'qqp': 'acc_and_f1',
    'qnli': 'acc',
    'sst-2': 'acc',
    'cola': 'mcc',
    'sts-b': 'corr',
    'mrpc': 'acc_and_f1',
    'rte': 'acc',
    'wnli': 'acc',
}

all_squad_select_metrics = {
    'squad1.1': 'exact_and_f1',
    'squad2.0': 'exact_and_f1',
}

all_summarization_select_metrics = {
    'cnn_dm': 'rouge-1',
}
