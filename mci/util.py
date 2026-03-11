import numpy
import re
import ast
import os
import sys
from importlib.util import spec_from_file_location, module_from_spec


LLM_LLAMA = 'llama3.1:8b'
LLM_MIXTRAL = 'mixtral:8x7b'
LLM_GEMMA = 'gemma3:12b'
SCORE_INVALID = -1e+9
AST_HEIGHT_MIN = 5
AST_HEIGHT_MAX = 20


def parse_code_str(code_str):
    return ast.parse(code_str, mode='exec')


def calculate_depth(node):
    if not isinstance(node, ast.AST):
        return 0

    child_nodes = [
        n for field in ast.iter_fields(node)
        for n in (field[1] if isinstance(field[1], list) else [field[1]])
        if isinstance(n, ast.AST)
    ]

    if not child_nodes:
        return 1

    return 1 + max(calculate_depth(child) for child in child_nodes)


def load_code_file(path_code_file):
    module_name = os.path.splitext(os.path.basename(path_code_file))[0]
    spec = spec_from_file_location(module_name, path_code_file)
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def load_equation(code):
    try:
        with open('equation.py', 'w', encoding='utf-8') as fp:
            fp.write(code)
        eq_obj = load_code_file('equation.py').equation
        os.remove('equation.py')
        return eq_obj
    except:
        if os.path.exists('equation.py'):
            os.remove('equation.py')
        return None


def get_num_params(code):
    param_str = ','.join(re.findall(r"params\[\d+\]", code))
    param_idx = re.findall(r"\[\d+\]", param_str)
    param_idx = [int(s.replace('[', '').replace(']', '')) for s in param_idx]

    if len(param_idx) == 0:
        for line in code.split('\n'):
            if '=params' in line or '= params' in line:
                return line.count(',') + 1

    return max(param_idx) + 1


def normalize_num_params(params, max_num_params):
    if params is None:
        return None

    if params.shape[0] > max_num_params:
        return None

    if params.shape[0] <= max_num_params:
        return 0

    return (params.shape[0] - max_num_params) / (2 * max_num_params - max_num_params)


def normalize_ast_height(h):
    if h is None:
        return None

    if h > AST_HEIGHT_MAX:
        return None

    if h < AST_HEIGHT_MIN:
        return 0

    return (h - AST_HEIGHT_MIN) / (AST_HEIGHT_MAX - AST_HEIGHT_MIN)


def calc_score(y_true, y_pred, ast_complexity, ast_num_params):
    try:
        if y_pred is None:
            return SCORE_INVALID

        if isinstance(y_pred, list):
            if len(y_pred) == 0:
                return SCORE_INVALID

        if numpy.any(numpy.isnan(y_pred)) or numpy.any(numpy.isinf(y_pred)):
            return SCORE_INVALID

        if ast_complexity is None:
            return SCORE_INVALID

        if ast_num_params is None:
            return SCORE_INVALID

        loss = numpy.sum((y_true - y_pred) ** 2)

        return -(loss + ast_complexity + ast_num_params)
    except:
        return SCORE_INVALID


def nmse(y_true, y_pred):
    return numpy.sum((y_true - y_pred)**2) / numpy.sum((y_true - numpy.mean(y_true))**2)


def make_eq_with_params(code, params):
    for i in range(0, params.shape[0]):
        code = code.replace('params[{}]'.format(i), str(params[i]))

    return code


def wmape(y_true, y_pred):
    y_true, y_pred = numpy.array(y_true), numpy.array(y_pred)
    total_error = numpy.sum(numpy.abs(y_true - y_pred))
    total_actual = numpy.sum(numpy.abs(y_true))

    return total_error / total_actual
