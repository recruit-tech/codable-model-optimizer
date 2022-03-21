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

from codableopt.interface.system_variable import SystemVariable, SystemIntegerVariable, SystemDoubleVariable, SystemCategoryVariable
from codableopt.solver.formulation.variable.solver_variable import SolverVariable
from codableopt.solver.formulation.variable.solver_integer_variable import SolverIntegerVariable
from codableopt.solver.formulation.variable.solver_double_variable import SolverDoubleVariable
from codableopt.solver.formulation.variable.solver_category_variable import SolverCategoryVariable


class SolverVariableFactory:
    """SolverVariableを生成するFactoryクラス。
    """

    def __init__(self):
        self._var_no = 0
        self._var_start_index = 0

    def generate(self, variable: SystemVariable) -> SolverVariable:
        """SolverVariableを生成する関数。

        Args:
            variable: 生成元の変数

        Returns:
            生成したSolverVariable
        """
        if isinstance(variable, SystemIntegerVariable):
            variable = SolverIntegerVariable(
                self._var_no,
                self._var_start_index,
                variable.name,
                variable.lower,
                variable.upper)
        elif isinstance(variable, SystemDoubleVariable):
            variable = SolverDoubleVariable(
                self._var_no,
                self._var_start_index,
                variable.name,
                variable.lower,
                variable.upper)
        elif isinstance(variable, SystemCategoryVariable):
            variable = SolverCategoryVariable(
                self._var_no,
                self._var_start_index,
                variable.name,
                variable.categories)
        else:
            raise NotImplementedError(
                f'Not support variable {variable.__class__}.')

        self._var_no += 1
        self._var_start_index += variable.array_size()

        return variable
