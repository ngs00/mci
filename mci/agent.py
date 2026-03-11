import ollama
import pickle
from scipy.optimize import minimize
from mci.util import *


class State:
    def __init__(self, path_code_file, code, ast_depth):
        self.path_code_file = path_code_file
        self.code = code
        self.ast_depth = ast_depth
        self.opt_params = None
        self.score = SCORE_INVALID

    def exec(self, x, dataset_prior):
        try:
            eq_obj = load_equation(self.code)

            _x = dict()
            for i in range(0, len(dataset_prior.input_vars)):
                _x[dataset_prior.input_vars[i]] = numpy.concat([dataset_prior.x[:, i], x[:, i]])

            return eq_obj(self.opt_params, **_x)[len(dataset_prior):]
        except:
            return None

    def optimize_params(self, dataset_train):
        try:
            equation = load_equation(self.code)
            num_params = get_num_params(self.code)
            params_init = numpy.array([1.0 for _ in range(0, num_params)])

            _x = dict()
            for i in range(0, len(dataset_train.input_vars)):
                _x[dataset_train.input_vars[i]] = dataset_train.x[:, i]

            def loss_func(params):
                return numpy.mean((dataset_train.y - equation(params, **_x))**2)

            return minimize(loss_func, params_init, method='BFGS').x
        except:
            return None

    def save(self, path_file):
        with open(path_file, 'wb') as fp:
            pickle.dump(self, fp)

    @staticmethod
    def load(path_file):
        with open(path_file, 'rb') as fp:
            return pickle.load(fp)


class Agent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.states = list()
        self.reset_count = 0
        self.p_best_state = None

    @property
    def current_state(self):
        return self.states[-1]

    def exec(self, prompt, code_dir, llm_model, temperature=0.8):
        path_code_file = '{}/agent_{}.py'.format(code_dir, self.agent_id)
        response = ollama.generate(model=llm_model,
                                   prompt=prompt,
                                   options={'temperature': temperature})['response']

        try:
            code_str = re.findall(r"```(?:python)?\n(.*?)```", response, re.DOTALL | re.IGNORECASE)[0]
            import_str = ''

            if 'import numpy as np' not in code_str:
                import_str += 'import numpy as np\n'

            if 'from scipy.special import ndtr, logit, expit, softmax, gamma' not in code_str:
                import_str += 'from scipy.special import ndtr, logit, expit, softmax, gamma\n'

            code_str = import_str + '\n' + code_str

            with open(path_code_file, 'w') as f:
                f.write(code_str)

            ast_depth = calculate_depth(parse_code_str(code_str))
            self.states.append(State(path_code_file, code_str, ast_depth))
        except:
            self.states.append(State(path_code_file, None, None))

    def optimize_eq_params(self, x, y):
        try:
            module = load_code_file(self.current_state.path_code_file)
            equation = module.equation
            num_params = get_num_params(self.current_state.code)
            params_init = numpy.array([1.0 for _ in range(0, num_params)])

            def loss_func(params):
                return numpy.mean((y - equation(params, **x))**2)

            return minimize(loss_func, params_init, method='BFGS').x
        except:
            return None
