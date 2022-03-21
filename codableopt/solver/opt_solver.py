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

from codableopt.interface.interface import Problem
from codableopt.solver.formulation.solver_problem import SolverProblem
from codableopt.solver.optimizer.optimization_solver import OptimizationSolver
from codableopt.solver.optimizer.method.optimizer_method import OptimizerMethod


class OptSolver:
    """最適化ソルバークラス。
    """

    def __init__(
            self,
            round_times: int = 1,
            num_to_tune_penalty: int = 1000,
            num_to_select_init_answer: int = 1000,
            debug: bool = False,
            debug_unit_step: int = 1000,
            debug_log_file_path: Optional[Path] = None):
        """最適化ソルバーのオブジェクト生成関数。

        Args:
            round_times (int): 初期解を変えて、問題を解く回数
            num_to_tune_penalty (int): 初期のペナルティ係数を調整する際に利用する解答をランダム生成する数
            num_to_select_init_answer (int): 初期解を選択する時に、選択する元となる解答をランダム生成する数
            debug (bool): デバックprintの有無。
            debug_unit_step: デバックprintの表示step間隔
            debug_log_file_path: デバックログのファイル出力先を指定
        """
        if num_to_tune_penalty <= 1:
            raise ValueError('answer_num_to_tune_penalty must be greater than 1.')

        if num_to_select_init_answer < round_times:
            raise ValueError('answer_num_to_select_init_answer '
                             'must be greater than solve_times.')
        elif num_to_select_init_answer <= 1:
            raise ValueError('answer_num_to_select_init_answer must be greater than 1.')

        self._round_times = round_times
        self._num_to_tune_penalty = num_to_tune_penalty
        self._num_to_select_init_answer = num_to_select_init_answer
        self._debug = debug
        self._debug_unit_step = debug_unit_step
        self._debug_log_file_path = debug_log_file_path

    def solve(
            self,
            problem: Problem,
            method: OptimizerMethod,
            penalty_strength: float = 1.0,
            n_jobs: int = 1):
        """最適化を実施する関数。

        Args:
            problem (OptimizationProblem): 最適化問題
            method (OptimizerMethod): 最適化手法
            penalty_strength: ペナルティ係数の強さ、大きくするほど強くなる
            n_jobs: 並列実行数

        Returns:
            最適化の答え, 制約充足フラグ
        """
        system_problem = problem.compile()
        solver_problem = SolverProblem(system_problem)

        opt_solver = OptimizationSolver(
            debug=self._debug,
            debug_step_unit=self._debug_unit_step,
            debug_log_file_path=self._debug_log_file_path
        )
        answer, is_feasible = opt_solver.solve(
            solver_problem,
            method,
            self._round_times,
            self._num_to_tune_penalty,
            self._num_to_select_init_answer,
            penalty_strength,
            n_jobs)

        return answer, is_feasible
