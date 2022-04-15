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

from codableopt.solver.formulation.variable.solver_variable import SolverVariable
from codableopt.solver.formulation.variable.solver_category_variable import SolverCategoryVariable
from codableopt.interface.system_args_map import SystemArgsMap


class SolverArgsMap:
    """Solverための引数マッピング情報のクラス。
    """

    def __init__(
            self,
            args_map: SystemArgsMap,
            solver_variables: List[SolverVariable]):

        # TODO Refactor improve readability
        solver_var_dict = {x.name: x.var_index for x in solver_variables}
        for var in solver_variables:
            if isinstance(var, SolverCategoryVariable):
                array_indexes = [var.var_index + x for x in range(len(var.categories))]
                solver_var_dict[f'{var.name}'] = (array_indexes, var.categories)
                for category_no, category in enumerate(var.categories):
                    solver_var_dict[f'{var.name}:{category}'] = var.var_index + category_no

        self._args = {}
        for parameter_key in args_map.parameter_args_map.keys():
            self._args[parameter_key] = args_map.parameter_args_map[parameter_key]
        # TODO n次元対応
        self._single_vars_of_args = []
        self._multi_vars_of_args = []
        self._2d_array_vars_of_args = []
        self._3d_array_vars_of_args = []
        for var_key in args_map.variable_args_map.keys():
            value = args_map.variable_args_map[var_key]
            if isinstance(value[0], list) and isinstance(value[0][0], list) and \
                    isinstance(value[0][0][0], list):
                state_indexes_and_category_var = \
                    [[[[solver_var_dict[x] for x in y] for y in z] for z in a] for a in value]
                self._3d_array_vars_of_args.append((var_key, state_indexes_and_category_var))
            elif isinstance(value[0], list) and isinstance(value[0][0], list):
                state_indexes_and_category_var = \
                    [[[solver_var_dict[x] for x in y] for y in z] for z in value]
                self._2d_array_vars_of_args.append((var_key, state_indexes_and_category_var))
            elif isinstance(value[0], list):
                state_indexes_and_category_var = [[solver_var_dict[x] for x in y] for y in value]
                self._multi_vars_of_args.append((var_key, state_indexes_and_category_var))
            else:
                state_indexes_and_category_var = [solver_var_dict[x] for x in value]
                self._single_vars_of_args.append((var_key, state_indexes_and_category_var))

        self._single_category_vars_of_args = []
        self._multi_category_vars_of_args = []
        self._2d_array_category_vars_of_args = []
        self._3d_array_category_vars_of_args = []
        for var_key in args_map.category_args_map.keys():
            value = args_map.category_args_map[var_key]
            if isinstance(value[0], list) and isinstance(value[0][0], list) and \
                    isinstance(value[0][0][0], list):
                state_indexes = \
                    [[[[solver_var_dict[x] for x in y][0] for y in z] for z in a] for a in value]
                self._3d_array_category_vars_of_args.append((var_key, state_indexes))
            elif isinstance(value[0], list) and isinstance(value[0][0], list):
                state_indexes = [[[solver_var_dict[x] for x in y][0] for y in z] for z in value]
                self._2d_array_category_vars_of_args.append((var_key, state_indexes))
            elif isinstance(value[0], list):
                state_indexes = [[solver_var_dict[x] for x in y][0] for y in value]
                self._multi_category_vars_of_args.append((var_key, state_indexes))
            else:
                state_indexes = [solver_var_dict[x] for x in value][0]
                self._single_category_vars_of_args.append((var_key, state_indexes))

    def update_previous_args(self, state):
        self._update_args(state, head_name='pre_')

    def update_args(self, state):
        self._update_args(state, head_name='')

    def _update_args(self, state, head_name: str = ''):
        # TODO Refactor improve performance and readability
        for args in self._single_vars_of_args:
            self._args[f'{head_name}{args[0]}'] = sum(state[args[1]])

        for args in self._multi_vars_of_args:
            self._args[f'{head_name}{args[0]}'] = [sum(state[x]) for x in args[1]]

        for args in self._2d_array_vars_of_args:
            self._args[f'{head_name}{args[0]}'] = [[sum(state[x]) for x in y] for y in args[1]]

        for args in self._3d_array_vars_of_args:
            self._args[f'{head_name}{args[0]}'] = \
                [[[sum(state[x]) for x in y] for y in z] for z in args[1]]

        for args in self._single_category_vars_of_args:
            indexes, categories = args[1]
            self._args[f'{head_name}{args[0]}'] = \
                categories[([x for x, y in enumerate(state[indexes]) if y == 1][0])]

        for args in self._multi_category_vars_of_args:
            self._args[f'{head_name}{args[0]}'] = \
                [categories[([x for x, y in enumerate(state[indexes]) if y == 1][0])]
                 for indexes, categories in args[1]]

        for args in self._2d_array_category_vars_of_args:
            self._args[f'{head_name}{args[0]}'] = \
                [[categories[([x for x, y in enumerate(state[indexes]) if y == 1][0])]
                  for indexes, categories in z] for z in args[1]]

        for args in self._3d_array_category_vars_of_args:
            self._args[f'{head_name}{args[0]}'] = \
                [[[categories[([x for x, y in enumerate(state[indexes]) if y == 1][0])]
                   for indexes, categories in z] for z in a] for a in args[1]]

    @property
    def args(self):
        return self._args
