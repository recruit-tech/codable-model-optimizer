# Copyright 2022 Recruit Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional
from pathlib import Path
import copy
from concurrent import futures
import multiprocessing

import numpy as np

from codableopt.solver.optimizer.method.optimizer_method import OptimizerMethod
from codableopt.solver.optimizer.optimizer import Optimizer
from codableopt.solver.formulation.solver_problem import SolverProblem
from codableopt.solver.sampler.var_value_array_sampler import VarValueArraySampler


class OptimizationSolver:
    """最適化を複数回の実行管理する関数。
    """

    def __init__(
            self,
            debug: bool = False,
            debug_step_unit: int = 1000,
            debug_log_file_path: Optional[Path] = None):
        self._debug = debug
        self._debug_step_unit = debug_step_unit
        self._debug_log_file_path = debug_log_file_path

    def solve(
            self,
            problem: SolverProblem,
            method: OptimizerMethod,
            round_times: int,
            num_to_tune_penalty: int,
            num_to_select_init_answer: int,
            penalty_strength: float,
            n_jobs: int):
        """最適化を行う関数。

        Args:
            problem: 最適化問題
            method: 最適化手法
            round_times (int): 初期解を変えて、問題を解く回数
            num_to_tune_penalty (int): 初期のペナルティ係数を調整する際に利用する解答をランダム生成する数
            num_to_select_init_answer (int): 初期解を選択する時に、選択する元となる解答をランダム生成する数
            penalty_strength: ペナルティ係数の強さ、大きくするほど強くなる
            n_jobs: 並列実行数
        Returns:
            最適解、制約充足フラグ
        """
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        sampler = VarValueArraySampler()
        answers_to_tune = sampler.generate(problem, num_to_tune_penalty)
        random_answers = sampler.generate(problem, num_to_select_init_answer)
        init_answers = sampler.choice(random_answers, round_times)

        if n_jobs == 1:
            results = []
            for round_no, init_answer in enumerate(init_answers):
                state = self.__optimize(
                    init_var_value_array=np.array(init_answer),
                    answers_to_tune=answers_to_tune,
                    penalty_strength_to_tune=penalty_strength,
                    method=copy.deepcopy(method),
                    problem=copy.deepcopy(problem),
                    round_no=round_no
                )
                results.append(state)
        else:
            with futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
                tasks = [
                    executor.submit(
                        self.__optimize,
                        init_var_value_array=np.array(init_answer),
                        answers_to_tune=answers_to_tune,
                        penalty_strength_to_tune=penalty_strength,
                        method=copy.deepcopy(method),
                        problem=copy.deepcopy(problem),
                        round_no=round_no)
                    for round_no, init_answer in enumerate(init_answers)
                ]
                results = [task.result() for task in tasks]

        best_state = None
        for state in results:
            if best_state is None:
                best_state = state
            elif not best_state.exist_feasible_answer and state.exist_feasible_answer:
                best_state = state
            elif best_state.exist_feasible_answer and state.exist_feasible_answer \
                    and best_state.best_score.score > state.best_score.score:
                best_state = state
            elif not best_state.exist_feasible_answer and not state.exist_feasible_answer \
                    and best_state.best_score.score > state.best_score.score:
                best_state = state

        return best_state.to_answer(best_state.best_var_array), best_state.exist_feasible_answer

    def __optimize(
            self,
            init_var_value_array: np.array,
            answers_to_tune: np.array,
            penalty_strength_to_tune,
            method: OptimizerMethod,
            problem: SolverProblem,
            round_no: int):
        optimizer = Optimizer(
            method=method,
            problem=problem,
            round_no=round_no,
            init_var_value_array=init_var_value_array,
            var_value_arrays_to_tune=answers_to_tune,
            penalty_strength=penalty_strength_to_tune,
            debug_log_file_path=self._debug_log_file_path
        )
        return optimizer.optimize(self._debug, self._debug_step_unit)
