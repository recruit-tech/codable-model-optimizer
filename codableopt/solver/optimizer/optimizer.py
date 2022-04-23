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

from pathlib import Path
from typing import Optional
from logging import getLogger, StreamHandler, FileHandler, INFO
from time import time

import numpy as np

from codableopt.solver.formulation.solver_problem import SolverProblem
from codableopt.solver.optimizer.method.optimizer_method import OptimizerMethod
from codableopt.solver.optimizer.optimization_state import OptimizationState


class Optimizer:
    """最適化を行うクラス。手法は、methodに従う。
    """

    def __init__(
            self,
            problem: SolverProblem,
            method: OptimizerMethod,
            round_no: int,
            init_var_value_array: np.array,
            var_value_arrays_to_tune: Optional,
            penalty_strength,
            debug_log_file_path: Optional[Path] = None):
        self._logger = getLogger(__name__)
        self._logger.setLevel(INFO)

        self._state = OptimizationState(problem, init_var_value_array)
        if var_value_arrays_to_tune is not None:
            self._state.tune_penalty(var_value_arrays_to_tune, np.double(penalty_strength))
        self._state.init_scores()

        self._problem = problem
        self._method = method
        self._round_no = round_no

        if len(self._logger.handlers) == 0:
            self._logger.addHandler(StreamHandler())
            if debug_log_file_path is not None:
                self._logger.addHandler(FileHandler(filename=debug_log_file_path))

    def optimize(self, debug: bool = False, debug_unit_step: int = 100) -> OptimizationState:
        """最適化を行う関数。

        Args:
            debug: デバッグフラッグ
            debug_unit_step:  デバッグ情報を表示するステップ間隔

        Returns:
            最適解, 最適解のスコア
        """
        if debug:
            self._logger.info(f'log_type:method,round:{self._round_no},'
                              f'method_name:{self._method.name()}')

        start_time = time()

        step = 0
        move_cnt = 0

        while step < self._method.steps:
            step += 1
            self._method.initialize_of_step(self._state, step)

            proposed_move_list = self._method.propose(self._state, step)
            self._state.propose(proposed_move_list)

            move_flg = self._method.judge(self._state, step)

            if move_flg:
                move_cnt += 1
                self._state.apply_proposal()

            self._method.finalize_of_step(self._state, step)

            if not move_flg:
                self._state.cancel_propose()

            if debug and step % debug_unit_step == 0:
                self._logger.info(
                    f'log_type:optimize,'
                    f'round:{self._round_no},'
                    f'step:{step},'
                    f'move_count:{move_cnt}/{debug_unit_step},'
                    f'best_score:{self._state.best_score.score},'
                    f'best_is_feasible:{self._state.exist_feasible_answer},'
                    f'score:{self._state.current_score.score},'
                    f'objective_score:{self._state.current_score.objective_score},'
                    f'penalty_score:{self._state.current_score.penalty_score}')
                move_cnt = 0

        end_time = time()
        if debug:
            self._logger.info(
                f'log_type:time,'
                f'round:{self._round_no},'
                f'step:{step},'
                f'calculation_time:{end_time - start_time},'
                f'best_score:{self._state.best_score.score},'
                f'best_is_feasible:{self._state.exist_feasible_answer}')

        return self._state

    @property
    def problem(self) -> SolverProblem:
        return self._problem
