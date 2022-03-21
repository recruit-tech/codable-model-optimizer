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

from typing import List, Dict
import math

import numpy as np

from codableopt.solver.optimizer.entity.proposal_to_move import ProposalToMove
from codableopt.interface.system_constraint import SystemLinerConstraint


class SolverLinerConstraints:
    """Solverのための問題に含まれる全ての線形制約式を管理するクラス。
    """

    # 制約式に=が含まれていない場合において、右項と左項の値が等しい時にペナルティ値が0より大きくするための基準値
    INITIAL_PENALTY = 1.0

    def __init__(
            self,
            liner_constraints: List[SystemLinerConstraint],
            coefficients):
        """Solverのための問題に含まれる全ての線形制約式を管理するオブジェクト生成関数。
        引数の線形制約式リストを変換し、制約式ごとではなく、全ての制約式をまとめて扱えるようになる。
        SolverLinerConstantsは、Variableに依存しないために、constraints_coefficientsはProblem内で計算する。

        Args:
            liner_constraints (List[SystemLinerConstraint]): 線形制約式のリスト
            coefficients: 各制約式の各変数の係数の二次元配列
        """
        # 各制約式の各変数の係数の二次元配列
        self._coefficients = coefficients
        # 各制約式の定数項の配列
        self._constants = np.array([x.constant for x in liner_constraints])
        # 各制約式の制約式に等号を含むかのフラグ配列
        self._include_no_zero_flags = np.array([0.0 if x.include_equal_to_zero else 1.0
                                                for x in liner_constraints])
        # keyが変数のインデックス番号、valueがkeyで指定された変数が含まれている制約式のインデックス番号のリスト
        self._non_zero_coefficients_index_dict: Dict[int, List[int]] = \
            {x: self._coefficients[:, x] != 0 for x in range(coefficients.shape[1])}
        # 各制約式のペナルティ係数の配列、初期値は1.0で固定とする
        self._init_penalty_coefficients = [np.double(1.0) for _ in liner_constraints]

    def calc_violation_amounts(self, cashed_constraint_sums) -> List[np.double]:
        return [
            0.0
            if (formula_sum > 0 if include_zero else formula_sum >= 0)
            else (abs(formula_sum) + SolverLinerConstraints.INITIAL_PENALTY)
            for formula_sum, include_zero
            in zip(cashed_constraint_sums, self._include_no_zero_flags)]

    def calc_constraint_sums(self, var_values: np.array) -> np.array:
        """全ての制約式の左項の合計値を計算する関数。

        Args:
            var_values: 全ての制約式の左項を計算するための解答

        Returns:
            全ての制約式の左項の合計値の配列
        """
        return np.dot(self._coefficients, var_values) + self._constants

    def apply_proposal_to_constraint_sums(
            self,
            proposals: List[ProposalToMove],
            cashed_constraint_sums: np.array) -> np.array:
        """解の遷移を適用し、全ての制約式の左項の合計値を更新する関数。

        Args:
            proposals: 適用する解の遷移リスト
            cashed_constraint_sums: 解の遷移前の制約式の左項の合計値キャッシュ

        Returns:
            解の遷移後の制約式の左項の合計値キャッシュ
        """
        # deepcopyを排除する方が早いので、copyせず更新した答えを再度更新して元に戻す方法を採用
        constraint_sums = cashed_constraint_sums
        for proposal in proposals:
            constraint_sums += np.dot(self._coefficients[:, proposal.var_index],
                                      proposal.new_value - proposal.pre_value)

        return constraint_sums

    def cancel_proposal_to_constraint_sums(
            self,
            proposals: List[ProposalToMove],
            cashed_constraint_sums: np.array) -> np.array:
        """適用した解の遷移を元に戻し、全ての制約式の左項の合計値を更新する関数。

        Args:
            proposals: 適用する解の遷移リスト
            cashed_constraint_sums: 解の遷移後の制約式の左項の合計値キャッシュ

        Returns:
            解の遷移前の制約式の左項の合計値キャッシュ
        """
        # deepcopyを排除する方が早いので、copyせず更新した答えを再度更新して元に戻す方法を採用
        constraint_sums = cashed_constraint_sums
        for proposal in proposals:
            constraint_sums -= np.dot(self._coefficients[:, proposal.var_index],
                                      proposal.new_value - proposal.pre_value)

        return constraint_sums

    def calc_var_range_in_feasible(
            self,
            var_value_array: np.array,
            var_index: int,
            cashed_constraint_sums: np.array,
            is_int: bool):
        """制約式を全て満たす時に、指定した変数がとりえる値の範囲を計算する関数。

        Args:
            var_value_array: 変数の値
            var_index: 計算する対象の変数のインデックス番号
            cashed_constraint_sums: 制約式の左項の合計値キャッシュ
            is_int: 計算する対象の変数の整数フラグ

        Returns:
            制約式を満たす変数の値の範囲
        """
        non_zero_indexes = self._non_zero_coefficients_index_dict[var_index]
        non_zero_coefficients = self._coefficients[:, var_index][non_zero_indexes]
        include_no_zero_flags = self._include_no_zero_flags[non_zero_indexes]
        delta_threshold = np.divide(
            (cashed_constraint_sums
             - np.dot(self._coefficients[:, var_index], var_value_array[var_index]))
            [non_zero_indexes],
            non_zero_coefficients)
        eps = np.finfo(np.float32).eps

        min_upper_delta_threshold = None
        upper_delta_thresholds = -delta_threshold[non_zero_coefficients < 0]
        if len(upper_delta_thresholds) > 0:
            upper_delta_thresholds -= include_no_zero_flags[non_zero_coefficients < 0] * eps
            min_upper_delta_threshold = upper_delta_thresholds.min()

        max_lower_delta_threshold = None
        lower_delta_threshold = delta_threshold[non_zero_coefficients > 0]
        if len(lower_delta_threshold) > 0:
            lower_delta_threshold += include_no_zero_flags[non_zero_coefficients > 0] * eps
            max_lower_delta_threshold = lower_delta_threshold.max()

        if is_int and max_lower_delta_threshold is not None:
            max_lower_delta_threshold = math.ceil(max_lower_delta_threshold)
        if is_int and min_upper_delta_threshold is not None:
            min_upper_delta_threshold = math.floor(min_upper_delta_threshold)

        return max_lower_delta_threshold, min_upper_delta_threshold

    @property
    def init_penalty_coefficients(self) -> np.array:
        return self._init_penalty_coefficients
