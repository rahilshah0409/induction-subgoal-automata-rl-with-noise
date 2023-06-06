import os
import shutil
import numpy as np
import json
from functools import reduce
from operator import mul


def get_param(param_dict, param_name, default_value=None):
    if param_dict is not None and param_name in param_dict:
        return param_dict[param_name]
    return default_value


def mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def rm_dir(dir_name):
    if path_exists(dir_name):
        shutil.rmtree(dir_name, ignore_errors=True)


def rm_dirs(dir_list):
    for dir_name in dir_list:
        rm_dir(dir_name)


def rm_file(filename):
    if os.path.exists(filename):
        os.remove(filename)


def rm_files(file_list):
    for filename in file_list:
        rm_file(filename)


def is_file_empty(path):
    return os.path.getsize(path) == 0


def path_exists(path):
    return os.path.exists(path)


def sort_by_ord(input_list):
    input_list.sort(key=lambda s: ord(s[1].lower()))

def sort_by_ord_env_vocab(input_list):
    input_list.sort(key=lambda s: ord(s.lower()))

def pair_sort_by_ord(input_list):
    input_list.sort(key=lambda s: ord(s[0].lower()))


def randargmax(input_vector):
    return np.random.choice(np.flatnonzero(input_vector == np.max(input_vector)))


def read_json_file(filepath):
    with open(filepath) as f:
        return json.load(f)


def write_json_obj(obj, filepath):
    with open(filepath, 'w') as f:
        json.dump(obj, f)


def min_t_norm_operator(scores):
    return min(scores)
    

def product_t_norm_operator(scores):
    return reduce(mul, scores)

def average_score(scores):
    return np.mean(scores)
