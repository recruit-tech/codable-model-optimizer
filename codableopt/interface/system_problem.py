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

from typing import List

from codableopt.interface.system_variable import SystemVariable
from codableopt.interface.system_objective import SystemObjective
from codableopt.interface.system_constraint import SystemConstraints


class SystemProblem:
    """最適化問題のクラス。
    """

    def __init__(
            self,
            variables: List[SystemVariable],
            objective: SystemObjective,
            constraints: SystemConstraints):
        """最適化問題のオブジェクト生成関数。

        Args:
            variables: 最適化問題の変数リスト
            objective: 最適化問題の目的関数
            constraints: 最適化問題の制約式集合
        """
        self._variables = variables
        self._objective = objective
        self._constraints = constraints

    @property
    def variables(self) -> List[SystemVariable]:
        return self._variables

    @property
    def objective(self) -> SystemObjective:
        return self._objective

    @property
    def constraints(self) -> SystemConstraints:
        return self._constraints
