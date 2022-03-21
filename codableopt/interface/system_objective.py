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

from typing import Dict, Optional, Union, List, Any

import numpy as np

from codableopt.interface.system_args_map import SystemArgsMap


class SystemObjective:
    """目的関数のベースクラス。
    """

    def __init__(self, is_max_problem: bool, exist_delta_function: bool):
        """目的関数を設定するオブジェクト生成関数。

        Args:
            is_max_problem: 最適化問題が最大化問題であるフラグ（Trueの場合は最大化問題、Falseの場合は最小化問題）
            exist_delta_function: 差分計算の利用有無のフラグ
        """
        self._is_max_problem = is_max_problem
        self._exist_delta_function = exist_delta_function

    def calc_objective(self, answer) -> np.double:
        """目的関数の値を計算する関数。

        Args:
            answer: 解答

        Returns:
            目的関数の値
        """
        raise NotImplementedError('You must write calculate_objective_function!')

    def calc_delta_objective(self, answer) -> np.double:
        """目的関数の値を差分計算によって計算する関数。

        Args:
            answer: 解答

        Returns:
            目的関数の値
        """
        raise NotImplementedError(
            'Write calculate_objective_function_by_difference_calculation!')

    @property
    def is_max_problem(self) -> bool:
        return self._is_max_problem

    @property
    def exist_delta_function(self) -> bool:
        return self._exist_delta_function


class SystemUserDefineObjective(SystemObjective):
    """ユーザ定義する目的関数クラス。
    """

    def __init__(
            self,
            is_max_problem: bool,
            objective,
            delta_objective: Optional,
            system_args_map: SystemArgsMap):
        self._objective = objective
        self._delta_objective = delta_objective
        self._system_args_map = system_args_map
        super().__init__(is_max_problem, delta_objective is not None)

    def calc_objective(self, args: Dict[str, Any]) -> np.double:
        return self._objective(**args)

    def calc_delta_objective(self, args: Dict[str, Any]) -> np.double:
        return self._delta_objective(**args)

    @property
    def args_map(self) -> SystemArgsMap:
        return self._system_args_map

    @property
    def variable_args_map(self) -> Dict[str, Union[str, List[str]]]:
        return self._system_args_map.variable_args_map

    @property
    def category_args_map(self) -> Dict[str, Union[str, List[str]]]:
        return self._system_args_map.category_args_map

    @property
    def parameter_args_map(self) -> Dict[str, Union[str, List[str]]]:
        return self._system_args_map.parameter_args_map
