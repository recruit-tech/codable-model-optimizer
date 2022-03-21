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

from typing import Dict, List

import numpy as np

from codableopt.interface.system_args_map import SystemArgsMap


class SystemConstraint:
    """制約式のベースクラス。
    """

    def __init__(self):
        pass


class SystemLinerConstraint(SystemConstraint):
    """線形制約式のクラス。

    Ax + b >= 0 or Ax + b > 0
     A is variable_coefficients
     b is constant
     (if include_equal_to_zero is True) Ax + b >= 0 (else) Ax + b > 0
    """

    def __init__(self,
                 var_coefficients: Dict[str, np.double],
                 constant: np.double,
                 include_equal_to_zero: bool):
        self._var_coefficients = var_coefficients
        self._constant = constant
        self._include_equal_to_zero = include_equal_to_zero
        super().__init__()

    def to_string(self) -> str:
        """線形制約式を文字列に変換。

        Returns:
            線形制約式の文字列
        """
        formula_str = \
            ' '.join([f'{"+" if val >= 0 and no > 0 else ""}{val}*{name}'
                      for no, (name, val)
                      in enumerate(self._var_coefficients.items())])
        formula_str += \
            f' {"+" if self._constant >= 0 else ""} {self._constant}'

        if self._include_equal_to_zero:
            return formula_str + ' >= 0'
        else:
            return formula_str + ' > 0'

    @property
    def var_coefficients(self) -> Dict[str, np.double]:
        return self._var_coefficients

    @property
    def constant(self) -> np.double:
        return self._constant

    @property
    def include_equal_to_zero(self) -> bool:
        return self._include_equal_to_zero


class SystemUserDefineConstraint(SystemConstraint):
    """ユーザ定義の制約式のクラス。（利用非推奨）
    """

    def __init__(self, constraint_function, system_args_map: SystemArgsMap):

        self._constraint_function = constraint_function
        self._args_map = system_args_map
        super().__init__()

    @property
    def constraint_function(self):
        return self._constraint_function

    @property
    def args_map(self):
        return self._args_map


class SystemConstraints:
    """問題に設定されている制約式をまとめたクラス。
    """

    def __init__(self,
                 liner_constraints: List[SystemLinerConstraint],
                 user_define_constraints: List[SystemUserDefineConstraint]):
        self._liner_constraints = liner_constraints
        self._user_define_constraints = user_define_constraints

    @property
    def liner_constraints(self) -> List[SystemLinerConstraint]:
        return self._liner_constraints

    @property
    def user_define_constraints(self) -> List[SystemUserDefineConstraint]:
        return self._user_define_constraints
