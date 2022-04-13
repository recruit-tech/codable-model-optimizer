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

import random
from typing import List, Optional, Dict, Tuple
from random import choice

import numpy as np

from codableopt.solver.optimizer.entity.proposal_to_move import ProposalToMove
from codableopt.solver.optimizer.optimization_state import OptimizationState
from codableopt.solver.optimizer.method.optimizer_method import OptimizerMethod
from codableopt.solver.formulation.variable.solver_integer_variable import SolverIntegerVariable
from codableopt.solver.formulation.variable.solver_double_variable import SolverDoubleVariable


class PenaltyAdjustmentMethod(OptimizerMethod):

    def __init__(
            self,
            steps: int,
            proposed_rate_of_random_movement: float = 0.95,
            delta_to_update_penalty_rate: float = 0.2,
            steps_threshold_to_judge_local_solution: Optional[int] = 100,
            history_value_size: int = 5,
            range_std_rate: float = 3.0):
        """ペナルティ係数調整手法の生成。

        Args:
            steps: 最適化の計算を繰り返すステップ数
            proposed_rate_of_random_movement: 解の遷移を提案するときにランダムな遷移を提案する割合
            delta_to_update_penalty_rate: ペナルティ係数を変えるときの割合（0.5を設定した場合、ペナルティを強くする時は係数を1+0.5倍、ペナルティを弱くする時は係数を1-0.5倍する）
            steps_threshold_to_judge_local_solution: 局所解とみなすために必要な連続で解が遷移しないステップ数（局所解とみなされた時にペナルティ係数を調整する）
            history_value_size: 数値が範囲指定のランダム遷移をする際に参考とする直近のデータ履歴のデータ件数
            range_std_rate: 数値が範囲指定のランダム遷移をする際に平均値から何倍の標準偏差まで離れている範囲を対象とするか指定する値（1.5なら、「平均値 - 1.5 * 標準偏差値」から「平均値 + 1.5 * 標準偏差値」を範囲とする）
        """
        # 設定パラメータ
        super().__init__(steps)

        if delta_to_update_penalty_rate <= 0 or delta_to_update_penalty_rate >= 1.0:
            raise ValueError('delta_to_update_penalty_rate must be '
                             '"0 < delta_to_update_penalty_rate < 1"')

        self._random_movement_rate = proposed_rate_of_random_movement
        self._delta_penalty_rate = delta_to_update_penalty_rate
        self._steps_threshold = steps_threshold_to_judge_local_solution

        # Method内変数
        self._proposal_list: List[ProposalToMove] = []
        # 直前のペナルティ係数の更新フラグ
        self._changed_penalty_flg: bool = False
        # 局所解に達してからのステップ数
        self._steps_while_not_improve_score: int = 0

        # 数値のmoveの範囲指定時のパラメータ関連
        self._history_value_size = history_value_size
        self._range_std_rate = range_std_rate
        # 数値変数の履歴
        self._number_variables_history: Dict[str, Tuple[np.array, np.double, np.double]] = {}

    def name(self) -> str:
        return f'penalty_adjustment_method,' \
               f'steps:{self._steps},' \
               f'steps_threshold:{self._steps_threshold}'

    def initialize_of_step(self, state: OptimizationState, step: int):
        pass

    def propose(self, state: OptimizationState, step: int) -> List[ProposalToMove]:
        while True:
            variable = choice(state.problem.variables)
            if random.random() <= self._random_movement_rate:
                if isinstance(variable, (SolverIntegerVariable, SolverDoubleVariable)):
                    if variable.name not in self._number_variables_history.keys():
                        self._proposal_list = variable.propose_random_move(state.var_array)
                    else:
                        value_array, average, std = self._number_variables_history[variable.name]
                        if std == 0:
                            self._proposal_list = variable.propose_random_move(state.var_array)
                        else:
                            # 平均+-標準偏差値のn倍の範囲から探索
                            self._proposal_list = variable.propose_random_move_with_range(
                                state.var_array,
                                lower=np.double(average - self._range_std_rate * std),
                                upper=np.double(average + self._range_std_rate * std))
                else:
                    self._proposal_list = variable.propose_random_move(state.var_array)
            else:
                self._proposal_list = variable.propose_low_penalty_move(state)

            if len(self._proposal_list) > 0:
                return self._proposal_list

    def judge(self, state: OptimizationState, step: int) -> bool:
        if self._changed_penalty_flg:
            # ペナルティ係数が変わっているので計算しなおし、解の遷移は空のリストとする
            penalty_scores = state.calculate_penalties(proposals=[])
            state.previous_score.set_scores(
                state.previous_score.objective_score, sum(penalty_scores))
            self._changed_penalty_flg = False

        # 移動判定
        delta_energy = state.current_score.score - state.previous_score.score
        move_flg = delta_energy < 0.0

        # 局所解判定
        if delta_energy >= 0:
            self._steps_while_not_improve_score += 1
        else:
            self._steps_while_not_improve_score = 0

        # 整数値、連続値の変更履歴を取得
        if move_flg and len(self._proposal_list) == 1:
            propose = self._proposal_list[0]
            variable = state.problem.variables[propose.var_no]
            if variable.name in self._number_variables_history.keys():
                value_array, average, std = self._number_variables_history[variable.name]
                if len(value_array) > self._history_value_size:
                    value_array[0:-1] = value_array[1:]
                    value_array[-1] = propose.new_value
                else:
                    value_array = np.append(value_array, propose.new_value)

                if len(value_array) == self._history_value_size:
                    average = value_array.mean()
                    std = value_array.std()
                self._number_variables_history[variable.name] = (value_array, average, std)
            else:
                self._number_variables_history[variable.name] = \
                    (np.array([propose.new_value]), np.double(0), np.double(0))

        return move_flg

    def finalize_of_step(self, state: OptimizationState, step: int):
        # 局所解に達したら、ペナルティ係数を調整する
        if self._steps_while_not_improve_score >= self._steps_threshold:
            # ペナルティ係数を調整したらリセットする
            self._steps_while_not_improve_score = 0
            self._changed_penalty_flg = True
            if state.previous_score.penalty_score > 0:
                # 実行解がしばらく見つからない場合に満たされていない制約式のペナルティ係数を上げる

                # 制約式の違反量のスケールから正規化を行う
                vio_amounts = \
                    state.problem.calc_violation_amounts(state.var_array,
                                                         state.cashed_liner_constraint_sums)
                normalized_vio_amounts = \
                    [vio_amount / rate for vio_amount, rate
                     in zip(vio_amounts, state.constraints_scale)]

                # 制約式の違反量が最大なものをベースに割合に変換する
                max_normalized_vio_amounts = max(normalized_vio_amounts)
                if max_normalized_vio_amounts == 0:
                    # 変更なし
                    return

                normalized_vio_amount_rates = \
                    [x / max_normalized_vio_amounts for x in normalized_vio_amounts]

                # 制約式の違反量の割合を基準にペナルティ係数を上げる
                state.penalty_coefficients = [
                    penalty * (1 + self._delta_penalty_rate * rate)
                    for penalty, rate
                    in zip(state.penalty_coefficients, normalized_vio_amount_rates)
                ]

            else:
                # 実行可能解に達している場合は、全ての制約式のペナルティ係数を減らす
                state.penalty_coefficients = [
                    penalty * (1 - self._delta_penalty_rate)
                    for penalty in state.penalty_coefficients
                ]
