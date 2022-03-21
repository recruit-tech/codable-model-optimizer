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

from typing import Optional, List
from abc import ABC, abstractmethod

import numpy as np

from codableopt.interface.system_constraint import SystemLinerConstraint


class SystemVariable(ABC):
    """最適化問題の変数のベースクラス。
    """

    def __init__(self, name: str):
        """最適化問題の変数のオブジェクト生成関数。

        Args:
            name: 変数名
        """
        self._name = name

    @abstractmethod
    def to_constraints_of_range(self) -> List[SystemLinerConstraint]:
        """変数の上界値と下界値を線形制約式に変換する関数。

        Returns:
            上界値と下界値の線形制約式リスト。
        """
        raise NotImplementedError('to_variable_limit_constraints is not implemented!')

    @abstractmethod
    def extract_coefficients(
            self,
            constraint: SystemLinerConstraint) -> List[float]:
        """引数の線形制約式から呼び出し元オブジェクトの変数の係数を取得する関数。

        Args:
            constraint:
                線形制約式のオブジェクト。

        Returns:
            呼び出し元オブジェクトの変数の線形制約式の係数
        """
        raise NotImplementedError('to_constraint_coefficients is not implemented!')

    @property
    def name(self) -> str:
        return self._name


class SystemIntegerVariable(SystemVariable):
    """整数型の変数クラス。
    """

    def __init__(
            self,
            name: str,
            lower: Optional[np.double],
            upper: Optional[np.double]):
        super().__init__(name)
        self._lower = lower
        self._upper = upper

    def to_constraints_of_range(self) -> List[SystemLinerConstraint]:
        constant_list = []
        if self._lower is not None:
            constant = SystemLinerConstraint(
                var_coefficients={self._name: np.double(1.0)},
                constant=np.double(-self._lower),
                include_equal_to_zero=True
            )
            constant_list.append(constant)
        if self._upper is not None:
            constant = SystemLinerConstraint(
                var_coefficients={self._name: np.double(-1.0)},
                constant=self._upper,
                include_equal_to_zero=True
            )
            constant_list.append(constant)

        return constant_list

    def extract_coefficients(
            self,
            constraint: SystemLinerConstraint) -> List[float]:
        if self._name in constraint.var_coefficients.keys():
            return [constraint.var_coefficients[self._name]]
        else:
            return [0.0]

    @property
    def lower(self) -> Optional[np.double]:
        return self._lower

    @property
    def upper(self) -> Optional[np.double]:
        return self._upper


class SystemDoubleVariable(SystemVariable):
    """少数型の変数クラス。
    """

    def __init__(
            self,
            name: str,
            lower: Optional[np.double],
            upper: Optional[np.double]):
        super().__init__(name)
        self._lower = lower
        self._upper = upper

    def to_constraints_of_range(self) -> List[SystemLinerConstraint]:
        constant_list = []
        if self._lower is not None:
            constant = SystemLinerConstraint(
                var_coefficients={self._name: np.double(1.0)},
                constant=np.double(-self._lower),
                include_equal_to_zero=True)
            constant_list.append(constant)
        if self._upper is not None:
            constant = SystemLinerConstraint(
                var_coefficients={self._name: np.double(-1.0)},
                constant=self._upper,
                include_equal_to_zero=True)
            constant_list.append(constant)

        return constant_list

    def extract_coefficients(
            self,
            constraint: SystemLinerConstraint) -> List[float]:
        if self._name in constraint.var_coefficients.keys():
            return [constraint.var_coefficients[self._name]]
        else:
            return [0.0]

    @property
    def lower(self) -> Optional[np.double]:
        return self._lower

    @property
    def upper(self) -> Optional[np.double]:
        return self._upper


class SystemCategoryVariable(SystemVariable):
    """カテゴリ型の変数クラス。
    """

    def __init__(self, name: str, categories: List[str]):
        super().__init__(name)
        self._categories = categories
        self._category_num = len(categories)

    def to_constraints_of_range(self) -> List[SystemLinerConstraint]:
        return []

    def extract_coefficients(
            self,
            constraint: SystemLinerConstraint) -> List[float]:
        return [constraint.var_coefficients[f'{self.name}:{x}']
                if (f'{self._name}:{x}' in constraint.var_coefficients.keys()) else 0.0
                for x in self._categories]

    @property
    def categories(self) -> List[str]:
        return self._categories

    @property
    def category_num(self) -> int:
        return self._category_num
