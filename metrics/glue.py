from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {'acc': acc, 'f1': f1, 'acc_and_f1': (acc + f1) / 2}


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {'pearsonr': pearson_corr, 'spearmanr': spearman_corr, 'corr': (pearson_corr + spearman_corr) / 2}


def compute_glue_metrics(task_name, preds, labels):
    if task_name == 'cola':
        return {'mcc': matthews_corrcoef(labels, preds)}
    elif task_name == 'sst-2':
        return {'acc': simple_accuracy(preds, labels)}
    elif task_name == 'mrpc':
        return acc_and_f1(preds, labels)
    elif task_name == 'sts-b':
        return pearson_and_spearman(preds, labels)
    elif task_name == 'qqp':
        return acc_and_f1(preds, labels)
    elif task_name == 'mnli':
        return {'acc': simple_accuracy(preds, labels)}
    elif task_name == 'mnli-mm':
        return {'acc': simple_accuracy(preds, labels)}
    elif task_name == 'qnli':
        return {'acc': simple_accuracy(preds, labels)}
    elif task_name == 'rte':
        return {'acc': simple_accuracy(preds, labels)}
    elif task_name == 'wnli':
        return {'acc': simple_accuracy(preds, labels)}
