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

from typing import List, Optional
from collections.abc import Callable

import numpy as np

from codableopt.solver.formulation.args_map.solver_args_map import SolverArgsMap
from codableopt.solver.optimizer.entity.proposal_to_move import ProposalToMove


class SolverObjective:
    """Solverのための目的関数クラス。
    """

    def __init__(
            self,
            objective: Callable,
            delta_objective: Optional[Callable],
            args_map: SolverArgsMap,
            is_max_problem: bool,
            exist_delta_function: bool):
        """Solverのための目的関数オブジェクトの生成関数。

        Args:
            objective: 目的関数値を計算する関数
            delta_objective: 目的関数値を差分計算によって計算する関数
            args_map: 目的関数の引数に参照するマップ情報
            is_max_problem: 最大化問題フラグ
            exist_delta_function: 差分計算できる目的関数の有無
        """
        self._objective = objective
        self._delta_objective = delta_objective
        self._args_map = args_map
        self._is_max_problem = is_max_problem
        self._exist_delta_function = exist_delta_function

    def calc_objective(
            self,
            var_value_array: np.array,
            proposals: List[ProposalToMove],
            cashed_objective_score: Optional[np.double]) -> np.double:
        """目的関数値を計算する関数。

        Args:
            var_value_array: 決定変数の値リスト
            proposals: 目的関数値の計算前にベース解答に適用する変数の遷移提案リスト
            cashed_objective_score: キャッシュされている遷移前の目的関数値

        Returns:
            目的関数値
        """
        if not self._exist_delta_function or len(proposals) == 0 or cashed_objective_score is None:
            # 通常の目的関数の計算
            return self._calc_objective(var_value_array, proposals)
        else:
            # 差分計算による目的関数の計算
            return self.calc_objective_by_delta(var_value_array, proposals, cashed_objective_score)

    def _calc_objective(
            self,
            var_value_array: np.array,
            proposals: List[ProposalToMove]):
        # 目的関数を変更するために、一時的に値を変更
        for proposal in proposals:
            var_value_array[proposal.var_index] = proposal.new_value

        # 目的関数の引数を更新
        self._args_map.update_args(var_value_array)

        # 目的関数を計算
        score = self._objective(self._args_map.args)

        # 最大化問題の場合、-で最小化問題に変換
        if self._is_max_problem:
            score = -score

        # 目的関数を変更していた値を元に戻す
        for proposal in proposals:
            var_value_array[proposal.var_index] = proposal.pre_value

        return score

    def calc_objective_by_delta(
            self,
            var_value_array: np.array,
            proposals: List[ProposalToMove],
            cashed_objective_score: np.double) -> np.double:
        # 目的関数の引数の更新前の値を更新
        self._args_map.update_previous_args(var_value_array)

        # 目的関数を変更するために、一時的に値を変更
        for proposal in proposals:
            var_value_array[proposal.var_index] = proposal.new_value

        # 目的関数の引数を更新
        self._args_map.update_args(var_value_array)

        # 目的関数値のキャッシュ値を設定（キャッシュは目的関数のスコアとなっているのでフラグを修正）
        pre_objective_score = cashed_objective_score * (-1 if self._is_max_problem else 1)

        # 目的関数を計算
        score = pre_objective_score + self._delta_objective(self._args_map.args)

        # 最大化問題の場合、-で最小化問題に変換
        if self._is_max_problem:
            score = -score

        # 目的関数を変更していた値を元に戻す
        for proposal in proposals:
            var_value_array[proposal.var_index] = proposal.pre_value

        return score
