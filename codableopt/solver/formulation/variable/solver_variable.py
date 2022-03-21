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

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from codableopt.solver.optimizer.entity.proposal_to_move import ProposalToMove
from codableopt.solver.optimizer.optimization_state import OptimizationState


class SolverVariable(ABC):
    """ソルバー用の変数クラス。
    """

    VALUE_WHEN_NO_LIMIT = 100000

    def __init__(
            self,
            var_no: int,
            var_index: int,
            name: str):
        self._name = name
        self._var_no = var_no
        self._var_index = var_index

    @abstractmethod
    def array_size(self) -> int:
        """変数値をarrayに変換した時のarrayの長さを取得する関数。

        Returns:
            arrayに変換した時のarrayの長さ
        """
        raise NotImplementedError('array_size is not implemented!')

    @abstractmethod
    def propose_low_penalty_move(self, state: OptimizationState) -> List[ProposalToMove]:
        """なるべくペナルティの少ない解への遷移案を取得する関数。

        Args:
            state: 最適化の計算状態

        Returns:
            解の遷移リスト
        """
        raise NotImplementedError('move is not implemented!')

    @abstractmethod
    def propose_random_move_with_range(self, var_value_array: np.array, lower: np.double, upper: np.double) \
            -> List[ProposalToMove]:
        """指定された範囲の解への遷移案を取得する関数。

        Args:
            var_value_array: 現状の全変数の値を現したarray
            lower: 下界値
            upper: 上界値

        Returns:
            解の遷移リスト
        """
        raise NotImplementedError('propose_random_move_with_range is not implemented!')

    @abstractmethod
    def propose_random_move(self, var_value_array: np.array) -> List[ProposalToMove]:
        """ランダムな解への遷移案を取得する関数。

        Args:
            var_value_array: 現状の全変数の値を現したarray

        Returns:
            解の遷移リスト
        """
        raise NotImplementedError(
            'move_on_force_to_improve is not implemented!')

    @abstractmethod
    def values(self, var_value_array):
        """変数の値を引数の変数の値の集合から取得する関数。

        Args:
            var_value_array: 問題に含まれる全変数の値を現したarray

        Returns:
            変数の値
        """
        raise NotImplementedError('value is not implemented!')

    @abstractmethod
    def random_values(self):
        """変数のランダムな値を取得する関数。

        Returns:
            ランダムな値
        """
        raise NotImplementedError('random_value is not implemented!')

    @property
    def name(self) -> str:
        return self._name

    @property
    def var_no(self) -> int:
        return self._var_no

    @property
    def var_index(self) -> int:
        return self._var_index
