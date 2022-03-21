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

import numpy as np

from codableopt.solver.formulation.args_map.solver_args_map import SolverArgsMap


class SolverUserDefineConstraint:

    def __init__(self, constraint_function, args_map: SolverArgsMap):
        self._constraint_function = constraint_function
        self._args_map = args_map
        # ペナルティ係数、初期値は1.0で固定とする
        self._init_penalty_coefficient = 1.0

    def calc_violation_amount(self, var_values: np.array):
        # 引数を更新
        self._args_map.update_args(var_values)

        # 目的関数を計算
        return self._constraint_function(**self._args_map.args)

    @property
    def init_penalty_coefficient(self):
        return self._init_penalty_coefficient
