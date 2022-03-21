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

from typing import List, Dict, Union, Any


class SystemArgsMap:
    """引数のマッピング情報クラス。
    """

    def __init__(
            self,
            variable_args_map: Dict[Any, Union[List[str], List[list]]],
            category_args_map: Dict[str, List[str]],
            parameter_args_map: Dict[str, Any]):
        self._variable_args_map = variable_args_map
        self._category_args_map = category_args_map
        self._parameter_args_map = parameter_args_map

    @property
    def variable_args_map(self) -> Dict[Any, Union[List[str], List[list]]]:
        return self._variable_args_map

    @property
    def category_args_map(self) -> Dict[str, List[str]]:
        return self._category_args_map

    @property
    def parameter_args_map(self) -> Dict[str, Any]:
        return self._parameter_args_map
