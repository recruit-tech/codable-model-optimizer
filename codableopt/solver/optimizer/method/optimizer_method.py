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

from codableopt.solver.optimizer.entity.proposal_to_move import ProposalToMove
from codableopt.solver.optimizer.optimization_state import OptimizationState


class OptimizerMethod(ABC):
    """最適化の手法を定義するクラス。
    """

    def __init__(self, steps: int):
        self._steps = steps

    @abstractmethod
    def name(self) -> str:
        """最適化手法の名前を取得する関数

        Returns:
            最適化手法の名前
        """
        raise NotImplementedError('name is not implemented!')

    @abstractmethod
    def initialize_of_step(self, state: OptimizationState, step: int):
        """ステップの最初に呼び出される関数。

        Args:
            state: 実施中の最適化の情報オブジェクト
            step: ステップ数
        """
        raise NotImplementedError('setup_for_step is not implemented!')

    @abstractmethod
    def propose(self, state: OptimizationState, step: int) -> List[ProposalToMove]:
        """解の遷移を提案する関数。

        Args:
            state: 実施中の最適化の情報オブジェクト
            step: ステップ数

        Returns:
            提案する解の遷移リスト
        """
        raise NotImplementedError('propose_moving is not implemented!')

    @abstractmethod
    def judge(self, state: OptimizationState, step: int) -> bool:
        """解の遷移を判定する関数。

        Args:
            state: 実施中の最適化の情報オブジェクト
            step: ステップ数

        Returns:
            遷移判定結果
        """
        raise NotImplementedError('judge_move is not implemented!')

    @abstractmethod
    def finalize_of_step(self, state: OptimizationState, step: int):
        """ステップの最後に呼び出される関数。

        Args:
            state: 実施中の最適化の情報オブジェクト
            step: ステップ数
        """
        raise NotImplementedError('finalize_of_step is not implemented!')

    @property
    def steps(self) -> int:
        return self._steps
