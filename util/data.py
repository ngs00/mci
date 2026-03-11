import pandas
import numpy
from tqdm import tqdm


class Data:
    def __init__(self, x, y):
        self.x = numpy.array(x, dtype=float)
        self.y = numpy.array(y, dtype=float)


class Dataset:
    def __init__(self, data, input_vars, target_var):
        self.data = data
        self.input_vars = input_vars
        self.target_var = target_var
        self.x = numpy.vstack([d.x for d in self.data])
        self.y = numpy.array([d.y for d in self.data], dtype=float)

    def __len__(self):
        return len(self.data)

    def split(self, ratio_train=0.8, random_seed=None):
        if random_seed is not None:
            numpy.random.seed(random_seed)

        idx_rand = numpy.random.permutation(len(self))
        num_train = int(ratio_train * len(self))
        dataset_train = Dataset([self.data[idx] for idx in idx_rand[:num_train]], self.input_vars, self.target_var)
        dataset_test = Dataset([self.data[idx] for idx in idx_rand[num_train:]], self.input_vars, self.target_var)

        return dataset_train, dataset_test

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)


def load_dataset(path_dataset, idx_inputs, idx_target):
    _idx_inputs = numpy.atleast_1d(idx_inputs)

    if path_dataset.split('.')[-1] == 'xlsx':
        df = pandas.read_excel(path_dataset)
    else:
        df = pandas.read_csv(path_dataset)

    cols = list(df.columns)
    input_vars = [cols[i] for i in _idx_inputs]
    target_var = cols[idx_target]
    data = list()

    for d in tqdm(df.values.tolist()):
        data.append(Data([d[idx] for idx in _idx_inputs], d[idx_target]))

    return Dataset(data, input_vars, target_var)


def read_problem_instruction(file_path):
    with open(file_path, 'r') as f:
        instruction = f.read()

    return instruction
