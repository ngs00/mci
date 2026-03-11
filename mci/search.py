import time
import ollama
from copy import deepcopy
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from mci.agent import Agent
from mci.prompt import *
from mci.util import *


def optimize_params(agent, x, y):
    return agent.optimize_eq_params(x, y)


class Search:
    def __init__(self, task_domain, dataset, instruction, num_agents, device,
                 llm=LLM_MIXTRAL, init_temperature=1.0):
        self.device = device
        self.llm = llm
        self.task_domain = task_domain
        self.dataset = dataset
        self.instruction = instruction
        self.code_dir = 'search_res_{}'.format(time.time())
        self.agents = [Agent(agent_id=i) for i in range(0, num_agents)]
        self.g_best_state = None
        self.g_best_analysis = None
        self.x_kwargs = dict()
        self.init_temperature = init_temperature
        self.num_input_vars = len(self.dataset.input_vars)

        self.max_num_params = self._get_max_params()
        if self.max_num_params is None:
            raise ValueError

        for i in range(0, len(self.dataset.input_vars)):
            self.x_kwargs[self.dataset.input_vars[i]] = self.dataset.x[:, i]

        os.mkdir(self.code_dir)

    def _get_max_params(self):
        lines = self.instruction.split('\n')
        for line in lines:
            if 'MAX_NPARAMS' in line:
                return int(line.split('=')[1])

        return None

    def reset_agent(self, agent):
        agent.exec(write_init_prompt(self.instruction), self.code_dir, llm_model=self.llm, temperature=self.init_temperature)
        agent.p_best_state = agent.current_state
        agent.reset_count = 0

    def init_agents(self):
        init_prompt = write_init_prompt(self.instruction)
        for agent in self.agents:
            agent.exec(init_prompt, self.code_dir, llm_model=self.llm, temperature=self.init_temperature)
            agent.p_best_state = agent.current_state

    def evaluate_agents(self):
        results = [None] * len(self.agents)
        num_procs = len(self.agents) if len(self.agents) < 100 else 100

        with ProcessPool(max_workers=num_procs) as pool:
            list_ar = list()
            for i in range(0, len(self.agents)):
                ar = pool.schedule(optimize_params, args=[self.agents[i], self.x_kwargs, self.dataset.y], timeout=120)
                list_ar.append((i, ar))

            for i, ar in list_ar:
                try:
                    results[i] = ar.result()
                except TimeoutError:
                    results[i] = None

        for i in range(0, len(self.agents)):
            agent = self.agents[i]
            agent.current_state.opt_params = results[i]
            agent.current_state.score = self.eval_state(agent.current_state)

            if agent.current_state.score == SCORE_INVALID:
                agent.states[-1] = deepcopy(agent.p_best_state)
            else:
                if agent.current_state.score > agent.p_best_state.score:
                    agent.p_best_state = deepcopy(agent.current_state)
                else:
                    agent.reset_count += 1

        for i in range(0, len(self.agents)):
            print(i, self.agents[i].current_state.score)
        print('-----------------------------------')

    def eval_state(self, state):
        y_pred = state.exec(self.dataset.x, self.dataset)
        ast_complexity = normalize_ast_height(state.ast_depth)
        ast_num_params = normalize_num_params(state.opt_params, self.max_num_params)

        return calc_score(self.dataset.y, y_pred, ast_complexity, ast_num_params)

    def set_best_state(self):
        scores = [agent.current_state.score for agent in self.agents]
        current_best_agent = self.agents[numpy.argmax(scores)]

        if self.g_best_state is None:
            self.g_best_state = deepcopy(current_best_agent.current_state)
        else:
            if current_best_agent.current_state.score > self.g_best_state.score:
                self.g_best_state = deepcopy(current_best_agent.current_state)

    def generate(self):
        analysis_prompt = write_eq_analysis_prompt(self.task_domain, self.g_best_state.code)
        analysis_global = ollama.generate(model=self.llm, prompt=analysis_prompt)['response']
        self.g_best_analysis = deepcopy(analysis_global)

        for agent in self.agents:
            if agent.current_state.score <= SCORE_INVALID:
                self.reset_agent(agent)
                continue

            if agent.reset_count == 10:
                self.reset_agent(agent)
                continue

            code = agent.current_state.code
            y_pred = agent.current_state.exec(self.dataset.x, self.dataset)
            update_prompt = write_guided_update_prompt(y_pred, self.dataset, code, analysis_global, self.g_best_state.code)
            agent.exec(update_prompt, self.code_dir, llm_model=self.llm)

    def eval_agent(self, state, dataset_train, dataset_test):
        y_pred = state.exec(dataset_test.x, dataset_train)

        if y_pred is None:
            return -SCORE_INVALID
        else:
            return wmape(dataset_test.y, y_pred)

    def run(self, max_iters, log_file):
        log_file = open(log_file, 'w', encoding='utf-8')

        self.init_agents()
        self.evaluate_agents()
        self.set_best_state()

        for i in range(0, max_iters):
            self.generate()
            self.evaluate_agents()
            self.set_best_state()
            self.g_best_state.save(self.code_dir + '/best_state_{}.pkl'.format(i))

            print('Iteration [{}/{}]\tBest Score: {:.6f}'
                  .format(i + 1, max_iters, self.g_best_state.score), file=log_file, flush=True)
            print('Train WMAPE: {:.9f}'.format(self.eval_agent(self.g_best_state, self.dataset, self.dataset)),
                  file=log_file, flush=True)
            print(self.g_best_analysis, file=log_file, flush=True)
            print('---------------------------------------------------------------', file=log_file, flush=True)
            print(self.g_best_state.code, file=log_file, flush=True)
            print('===============================================================', file=log_file, flush=True)

        log_file.close()

        return self.g_best_state
