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

import numpy as np

from codableopt.solver.formulation.constraint.solver_liner_constraints \
    import SolverLinerConstraints
from codableopt.solver.formulation.constraint.solver_user_define_constraint \
    import SolverUserDefineConstraint


class SolverConstraints:

    def __init__(
            self,
            liner_constraints: Optional[SolverLinerConstraints],
            user_define_constraints: Optional[List[SolverUserDefineConstraint]]):
        self._liner_constraints = liner_constraints
        self._user_define_constraints = user_define_constraints

        self._init_penalty_coefficients = []
        if self._liner_constraints is not None:
            self._init_penalty_coefficients += liner_constraints.init_penalty_coefficients
        if self._user_define_constraints is not None:
            self._init_penalty_coefficients += [x.init_penalty_coefficient
                                                for x in user_define_constraints]

        self._init_penalty_coefficients = np.array(self._init_penalty_coefficients)

    def calc_violation_amounts(
            self,
            var_values,
            cashed_liner_constraint_sums) -> List[np.double]:
        """全ての制約式の制約違反量を計算する関数。

        Args:
            var_values: 各変数の値
            cashed_liner_constraint_sums: 制約式の左項の合計値のキャッシュ（変数の遷移提案前の値）
        Returns:
            全ての制約式の制約違反量
        """
        vio_amounts = []

        if self._liner_constraints is not None:
            vio_amounts += \
                self._liner_constraints.calc_violation_amounts(cashed_liner_constraint_sums)

        if self._user_define_constraints is not None:
            for user_define_constraint in self._user_define_constraints:
                vio_amounts += [user_define_constraint.calc_violation_amount(var_values)]

        return vio_amounts

    @property
    def liner_constraints(self) -> SolverLinerConstraints:
        return self._liner_constraints

    @property
    def user_define_constraints(self) -> List[SolverUserDefineConstraint]:
        return self._user_define_constraints

    @property
    def init_penalty_coefficients(self) -> np.array:
        return self._init_penalty_coefficients
