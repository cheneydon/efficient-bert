import os
import re
import random
import torch
import numpy as np
import lightgbm as lgb
import pandas as pd
import logging
from copy import deepcopy
from deap import gp
from sklearn.model_selection import train_test_split
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe, fmin
from contextlib import redirect_stdout


def get_linear_idx(expr):
    return '_'.join(re.search(r'linear(\d+)_(\d+)', str(expr)).groups())


def get_entire_linear_idx(entire_ind):
    entire_expr = [str(expr).replace(' ', '') for expr in entire_ind]
    entire_linear_idx = [get_linear_idx(expr) for expr in entire_expr]
    return entire_linear_idx


def get_entire_params(param_list, entire_ind):
    return param_list['embed_fit_dense'] + \
           sum([(param_list['attn'] + param_list[idx]) for idx in get_entire_linear_idx(entire_ind)])


def get_hash_key_acc(entire_expr, pset, fixed_mat_dir, hash_dict=None, is_stage2=False):
    entire_func = [gp.compile(expr, pset) for expr in entire_expr]
    fixed_x = torch.load(os.path.join(fixed_mat_dir, 'fixed_x.bin'))
    output = fixed_x
    for layer_id, expr in enumerate(entire_expr):
        if is_stage2:
            fixed_wb = torch.load(os.path.join(fixed_mat_dir, 'fixed_wb' + str(layer_id + 1) + '_1_1.bin'))
        else:
            linear_idx = get_linear_idx(expr)
            fixed_wb = torch.load(os.path.join(fixed_mat_dir, 'fixed_wb' + str(layer_id + 1) + '_' + linear_idx + '.bin'))
        output = entire_func[layer_id](output, *fixed_wb)

    hash_key = hash(round(output.sum().item(), 2))
    if hash_dict is None:
        return hash_key

    hash_acc = None
    if hash_key in hash_dict.keys():
        hash_acc = hash_dict[hash_key]
    return hash_key, hash_acc


def mutUniform(individual, expr, pset, min_, max_):
    index = random.randrange(len(individual))
    slice_ = individual.searchSubtree(index)
    type_ = individual[index].ret
    individual[slice_] = expr(pset=pset, type_=type_, min_=min_, max_=max_)
    return individual,


def check_individual(individual, pset, min_height, max_height):
    expr = str(individual).replace(' ', '')

    try:
        func = gp.compile(individual, pset)
        x, wb = torch.zeros(1, 1, 1), [torch.zeros(1, 1), torch.zeros(1)]
        num_wb = len([arg for arg in pset.arguments if arg.startswith('wb')])
        func(x, *([wb] * num_wb))
    except KeyError:
        return False

    if individual.height < min_height or individual.height > max_height:
        return False

    # Ensure x exists
    res = re.search(r'x', expr)
    if res is None:
        return False

    # Ensure each linear only appears once and arrange all linears in order
    res_wb1 = re.findall(r'wb1', expr)
    res_wb2 = re.findall(r'wb2', expr)
    if len(res_wb1) == 1 and len(res_wb2) == 1:
        wb1_start_idx = re.search(r'wb1', expr).start()
        wb2_start_idx = re.search(r'wb2', expr).start()
        if wb1_start_idx >= wb2_start_idx:
            return False
    else:
        return False

    # Avoid nested activations
    all_act = '(' + '|'.join(pset.activations) + ')'
    res = re.search(all_act + r'\(' + all_act, expr)
    if res is not None:
        return False

    # Avoid nested and inconsistent linears
    all_linear = '(' + '|'.join([op for op in pset.context if op.startswith('linear')]) + ')'
    res1 = re.search(all_linear + r'\(' + all_linear, expr)
    res2 = re.findall(all_linear, expr)
    if res1 is not None or res2[0] != res2[1]:
        return False

    # Roughly avoid same inputs of add, mul, max
    res1 = re.search(r'add\(([^,]+),(\1)\)', expr)
    res2 = re.search(r'mul\(([^,]+),(\1)\)', expr)
    res3 = re.search(r'max\(([^,]+),(\1)\)', expr)
    if res1 is not None or res2 is not None or res3 is not None:
        return False

    return True


def sample_individual(expr, pset, min_height, max_height, is_stage2):
    max_sample = 400 if is_stage2 else 1000

    individual = gp.PrimitiveTree.from_string(expr, pset)
    while max_sample:
        cur_ind = deepcopy(individual)
        cur_ind, = mutUniform(cur_ind, gp.genFull, pset, 0, 3)
        cur_ind, = gp.mutNodeReplacement(cur_ind, pset)
        cur_ind, = gp.mutEphemeral(cur_ind, mode='one')
        cur_ind, = gp.mutInsert(cur_ind, pset)

        if check_individual(cur_ind, pset, min_height, max_height):
            return cur_ind
        max_sample -= 1

    return individual


# Sample entire individual for stage 1 and 2
def sample_entire_individual(all_init_expr, pset, param_list, num_layers, min_height, max_height,
                             min_params, max_params, is_stage2=False, max_sample=1000):
    def _gen_entire_ind():
        if is_stage2:
            all_cand_expr = [expr.split(', ') for expr in all_init_expr]
            new_cand_expr = np.array([[re.sub(r'linear(\d+)_(\d+)', r'linear', str(expr)) for expr in cand_expr]
                                      for cand_expr in all_cand_expr])
            return [sample_individual(random.choice(new_cand_expr[:, i]), pset, min_height, max_height, is_stage2)
                    for i in range(num_layers)]
        else:
            return [sample_individual(random.choice(all_init_expr), pset, min_height, max_height, is_stage2)
                    for _ in range(num_layers)]

    entire_ind = _gen_entire_ind()
    if is_stage2:
        return entire_ind

    while max_sample:
        entire_params = get_entire_params(param_list, entire_ind)
        if min_params <= entire_params <= max_params:
            return entire_ind
        entire_ind = _gen_entire_ind()
        max_sample -= 1

    return entire_ind


# Sample entire individual for stage 3
def sample_entire_individual2(all_init_expr, pset, param_list, num_layers, min_params, max_params, max_sample=10000):
    def _gen_entire_ind():
        all_cand_expr = np.array([expr.split(', ') for expr in all_init_expr])
        entire_ind = [gp.PrimitiveTree.from_string(random.choice(all_cand_expr[:, i]), pset) for i in range(num_layers)]
        return entire_ind

    entire_ind = _gen_entire_ind()
    while max_sample:
        entire_params = get_entire_params(param_list, entire_ind)
        if min_params <= entire_params <= max_params:
            return entire_ind
        entire_ind = _gen_entire_ind()
        max_sample -= 1

    return entire_ind


# Sample individual ids for stage 1 and 2
def sample_individual_ids(all_init_expr, pset, param_list, num_layers, min_height, max_height, min_params, max_params,
                          max_pad_len=20, is_init=False, is_stage2=False):
    inp_args = pset.arguments
    inp_arg_name_map = {k: v for k, v in zip(inp_args, ['ARG' + str(i) for i in range(len(inp_args))])}
    all_args = [inp_arg_name_map[k] if k in inp_args else k for k in pset.mapping.keys()]
    all_arg_id_map = {k: v for v, k in enumerate(all_args)}

    if is_init:
        all_individuals = []
        if is_stage2:
            for i, expr in enumerate(all_init_expr):
                entire_ind = [gp.PrimitiveTree.from_string(re.sub(r'linear(\d+)_(\d+)', r'linear', str(expr_)), pset)
                              for expr_ in expr.split(', ')]
                all_individuals.append(entire_ind)
        else:
            for i, expr in enumerate(all_init_expr):
                if i == 0:
                    entire_ind = [gp.PrimitiveTree.from_string(expr, pset) for _ in range(num_layers)]
                    all_individuals.append(entire_ind)

        all_ind_ids = []
        for entire_ind in all_individuals:
            entire_ind_ids = []
            for layer_ind in entire_ind:
                cur_ind_ids = [all_arg_id_map[arg.name] for arg in layer_ind]
                cur_ind_ids += [-1] * (max_pad_len - len(cur_ind_ids))
                entire_ind_ids += cur_ind_ids
            all_ind_ids.append(entire_ind_ids)
        return all_ind_ids, all_individuals
    else:
        entire_ind = sample_entire_individual(
            all_init_expr, pset, param_list, num_layers, min_height, max_height, min_params, max_params, is_stage2)
        entire_ind_ids = []
        for layer_ind in entire_ind:
            cur_ind_ids = [all_arg_id_map[arg.name] for arg in layer_ind]
            cur_ind_ids += [-1] * (max_pad_len - len(cur_ind_ids))
            entire_ind_ids += cur_ind_ids
        return entire_ind_ids, entire_ind


# Sample individual ids for stage 3
def sample_individual_ids2(all_init_expr, pset, param_list, num_layers, min_params, max_params, is_init=False):
    linear_id_map = {k: v for v, k in enumerate(pset.all_linear_ids)}

    if is_init:
        all_individuals = []
        for i, expr in enumerate(all_init_expr):
            if i == 0:
                entire_ind = [gp.PrimitiveTree.from_string(expr_, pset) for expr_ in expr.split(', ')]
                all_individuals.append(entire_ind)

        all_ind_ids = []
        for entire_ind in all_individuals:
            entire_linear_idx = get_entire_linear_idx(entire_ind)
            entire_ind_ids = [linear_id_map[linear_id] for linear_id in entire_linear_idx]
            all_ind_ids.append(entire_ind_ids)
        return all_ind_ids, all_individuals
    else:
        entire_ind = sample_entire_individual2(all_init_expr, pset, param_list, num_layers, min_params, max_params)
        entire_linear_idx = get_entire_linear_idx(entire_ind)
        entire_ind_ids = [linear_id_map[linear_id] for linear_id in entire_linear_idx]
        return entire_ind_ids, entire_ind


def train_classifier(x, y):
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    train_data = lgb.Dataset(x_train, label=y_train)
    valid_data = lgb.Dataset(x_val, label=y_val)

    params = {
        'objective': 'binary',
        'metric': 'binary_error',
        'verbosity': -1,
        'seed': np.random.randint(1000),
        'num_threads': 4,
    }

    space = {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.5)),
        'max_depth': hp.choice('max_depth', [-1, 2, 3, 4, 5, 6]),
        'num_leaves': hp.choice('num_leaves', np.linspace(10, 200, 50, dtype=int)),
        'feature_fraction': hp.quniform('feature_fraction', 0.5, 1.0, 0.1),
        'bagging_fraction': hp.quniform('bagging_fraction', 0.5, 1.0, 0.1),
        'bagging_freq': hp.choice('bagging_freq', np.linspace(0, 50, 10, dtype=int)),
        'reg_alpha': hp.uniform('reg_alpha', 0, 2),
        'reg_lambda': hp.uniform('reg_lambda', 0, 2),
        'min_child_weight': hp.uniform('min_child_weight', 0.5, 10),
    }

    def objective(hyperparams):
        model = lgb.train({**params, **hyperparams}, train_data, 100, valid_data, early_stopping_rounds=100, verbose_eval=0)
        score = model.best_score['valid_0'][params['metric']]
        return {'loss': score, 'status': STATUS_OK}

    trials = Trials()
    with open(os.devnull, 'w+') as file, redirect_stdout(file):
        tpe.logger.setLevel(logging.ERROR)
        best = fmin(fn=objective, space=space, trials=trials, algo=tpe.suggest, max_evals=10, verbose=False, rstate=np.random.RandomState(1))
        hyperparams = space_eval(space, best)
        model = lgb.train({**params, **hyperparams}, train_data, 100, valid_data, early_stopping_rounds=100, verbose_eval=0)
    return model


class SearchPhase(object):
    def __init__(self, all_init_expr, pset, param_list, num_layers, min_expr_height, max_expr_height,
                 min_params, max_params, fixed_mat_dir, n_init_samples, train_interval, n_total_samples,
                 is_stage2=False, is_stage3=False, height_level=(400, 800, 1600, 3200)):
        self.all_init_expr = all_init_expr
        self.pset = pset
        self.param_list = param_list
        self.num_layers = num_layers
        self.min_expr_height = min_expr_height
        self.max_expr_height = max_expr_height
        self.min_params = min_params
        self.max_params = max_params
        self.fixed_mat_dir = fixed_mat_dir
        self.n_init_samples = n_init_samples
        self.train_interval = train_interval
        self.n_total_samples = n_total_samples
        self.is_stage2 = is_stage2
        self.is_stage3 = is_stage3
        self.height_level = height_level
        self.n_init_yield = len(self.all_init_expr)
        self.individuals, self.x, self.y = [], [], []
        self.ind_set, self.hash_key_set = set(), set()

    def height(self):
        return np.searchsorted(self.height_level, len(self.y)) + 1

    def sample_individual_ids(self, is_init=False):
        if self.is_stage3:
            return sample_individual_ids2(
                self.all_init_expr, self.pset, self.param_list, self.num_layers, self.min_params, self.max_params,
                is_init=is_init)
        else:
            return sample_individual_ids(
                self.all_init_expr, self.pset, self.param_list, self.num_layers, self.min_expr_height,
                self.max_expr_height, self.min_params, self.max_params, is_init=is_init, is_stage2=self.is_stage2)

    def get_hash_key_acc(self, entire_ind, hash_dict=None):
        return get_hash_key_acc(entire_ind, self.pset, self.fixed_mat_dir, hash_dict, self.is_stage2)

    def entire_ind_to_string(self, entire_ind):
        return ', '.join([str(ind).replace(' ', '') for ind in entire_ind])

    def is_unique(self, entire_ind, add_to_set=False):
        hash_key = self.get_hash_key_acc(entire_ind)
        entire_ind_str = self.entire_ind_to_string(entire_ind)
        if entire_ind_str in self.ind_set or hash_key in self.hash_key_set:
            return False
        if add_to_set:
            self.ind_set.add(entire_ind_str)
            self.hash_key_set.add(hash_key)
        return True

    def sampler(self):
        while len(self.y) < self.n_init_yield:
            init_ind_ids, init_individuals = self.sample_individual_ids(is_init=True)
            self.n_init_yield = len(init_individuals)
            yield init_ind_ids[len(self.y)], init_individuals[len(self.y)]

        while len(self.y) < self.n_init_samples:
            cur_x, cur_ind = self.sample_individual_ids()
            while not self.is_unique(cur_ind):
                cur_x, cur_ind = self.sample_individual_ids()
            yield cur_x, cur_ind

        self.classifier = LearningPhase(self.x, self.y, self.height())
        while True:
            cur_select = 0
            while cur_select < self.train_interval:
                cur_x, cur_ind = self.classifier.sample(self.sample_individual_ids, self.is_unique)
                cur_select += 1
                yield cur_x, cur_ind
            self.classifier = LearningPhase(self.x, self.y, self.height())

    def back_propagate(self, ind_str, x, acc):
        self.individuals.append(ind_str)
        self.x.append(x)
        self.y.append(acc)

    def run(self, train_val_function, save_dir, disp_freq, max_disp=50, local_rank=0):
        if self.is_stage3:
            train_function, val_function = train_val_function
            train_function()

        sampler = self.sampler()
        while len(self.y) < self.n_total_samples:
            pop_idx = len(self.y) + 1
            cur_x, cur_ind = next(sampler)
            ind_str = self.entire_ind_to_string(cur_ind)

            if self.is_stage2:
                params = get_entire_params(self.param_list, self.all_init_expr[0].split(', '))
            else:
                params = get_entire_params(self.param_list, cur_ind)

            if self.is_unique(cur_ind, add_to_set=True):
                if local_rank == 0:
                    logging.info('Params: {:.2f}M  Expr: {}'.format(params, ind_str))
                if self.is_stage2:
                    entire_ffn_func = [gp.compile(re.sub(r'linear(\d+)_(\d+)', r'linear', str(ind)), self.pset)
                                       for ind in cur_ind]
                    entire_linear_idx = get_entire_linear_idx(self.all_init_expr[0].split(', '))
                else:
                    entire_ffn_func = [gp.compile(ind, self.pset) for ind in cur_ind]
                    entire_linear_idx = get_entire_linear_idx(cur_ind)

                if self.is_stage3:
                    # Model and data loader is set to None here to use the default settings
                    # (student_model and downstream_dev_loader respectively)
                    acc = val_function([entire_ffn_func, entire_linear_idx], is_search=True)
                else:
                    acc = train_val_function(entire_ffn_func, entire_linear_idx)
                self.back_propagate(ind_str, cur_x, acc)

                if local_rank == 0:
                    logging.info('-' * 50)
                    logging.info('Pop idx: {}  Params: {:.2f}M  Acc: {}  Expr: {}'.format(pop_idx, params, acc, ind_str))
                    logging.info('-' * 50)
            else:
                if local_rank == 0:
                    logging.info('-' * 50)
                    logging.info('Evaluated individual: {}'.format(ind_str))
                    logging.info('-' * 50)

            if pop_idx % disp_freq == 0:
                sorted_idx = np.argsort(self.y)[::-1]
                sorted_ind = np.array(self.individuals)[sorted_idx]
                sorted_acc = np.array(self.y)[sorted_idx]

                if local_rank == 0:
                    num_disp = min(len(self.individuals), max_disp)
                    logging.info('-' * 50)
                    logging.info('Top {} individuals'.format(max_disp))
                    for i, (ind_str, acc) in enumerate(zip(sorted_ind[:num_disp], sorted_acc[:num_disp])):
                        if self.is_stage2:
                            cur_params = get_entire_params(self.param_list, self.all_init_expr[0].split(', '))
                        else:
                            cur_params = get_entire_params(self.param_list, ind_str.split(', '))
                        logging.info('[{}/{}]  Params: {:.2f}M  Acc: {}  Expr: {}'.format(i + 1, num_disp, cur_params, acc, ind_str))
                    logging.info('-' * 50)

                    save_path = os.path.join(save_dir, 'hall_of_fame.bin')
                    hall_of_fame = {'individual': sorted_ind, 'accuracy': sorted_acc}
                    torch.save(hall_of_fame, save_path)


class LearningPhase(object):
    def __init__(self, x, y, height):
        self.x = np.stack(x)
        self.y = np.stack(y)
        self.height = height
        self.value = np.median(self.y)
        self.threshold = np.median(self.y)

        if self.height > 0:
            self.model = train_classifier(self.x, self.y >= self.value)
            build_trial = 0
            while self.model.best_score['valid_0']['binary_error'] >= 0.5 + max(0, (build_trial - 100.0) / 1000.0 * 0.3):
                build_trial += 1
                self.model = train_classifier(self.x, self.y >= self.value)

            self.left = LearningPhase(self.x[self.y >= self.value], self.y[self.y >= self.value], self.height - 1)
            self.right = LearningPhase(self.x[self.y < self.value], self.y[self.y < self.value], self.height - 1)
            self.threshold = self.right.value

    def sample(self, sample_function, check_unique_function, max_sample=20000):
        i = 0
        while i < max_sample:
            cur_x, cur_ind = sample_function()
            if self.constrain(cur_x) and check_unique_function(cur_ind):
                return cur_x, cur_ind
            i += 1

        i = 0
        while i < max_sample:
            cur_x, cur_ind = sample_function()
            if check_unique_function(cur_ind):
                return cur_x, cur_ind
            i += 1

        cur_x, cur_ind = sample_function()
        return cur_x, cur_ind

    def constrain(self, x):
        path_model, path_node = self.ucb_select()
        x = pd.DataFrame(np.array(x).reshape([1, -1]))
        for m, n in zip(path_model, path_node):
            predict = m.model.predict(x)
            if not ((n == 'l' and predict >= 0.5) or (n == 'r' and predict < 0.5)):
                return False
        return True

    def ucb_select(self):
        c = self
        path_model, path_node = [], []
        while c.height > 0:
            path_model.append(c)
            l_ucb = self.get_ucb(c.left.value, c.threshold)
            r_ucb = self.get_ucb(c.right.value, c.threshold)
            c = c.left if l_ucb >= r_ucb else c.right
            if l_ucb >= r_ucb:
                path_node.append('l')
            else:
                path_node.append('r')
        return path_model, path_node

    @staticmethod
    def get_ucb(value, threshold):
        value = value - threshold
        return value
