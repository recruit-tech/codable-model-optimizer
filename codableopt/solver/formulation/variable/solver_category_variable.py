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
import random

import numpy as np

from codableopt.solver.formulation.variable.solver_variable import SolverVariable
from codableopt.solver.optimizer.optimization_state import OptimizationState
from codableopt.solver.optimizer.entity.proposal_to_move import ProposalToMove


class SolverCategoryVariable(SolverVariable):

    def __init__(
            self,
            var_no: int,
            var_index: int,
            name: str,
            categories: List[str]):
        super(SolverCategoryVariable, self).__init__(var_no, var_index, name)
        self._categories = categories
        self._category_num = len(categories)

    @property
    def categories(self) -> List[str]:
        return self._categories

    def array_size(self) -> int:
        return self._category_num

    def propose_low_penalty_move(self, state: OptimizationState) -> List[ProposalToMove]:
        hot_indexes = [index for index
                       in range(self._var_index, self._var_index + self._category_num)
                       if state.var_array[index] == 1]
        if len(hot_indexes) != 1:
            raise ValueError(f'Category Variable is not one hot encoding.')
        else:
            hot_index = hot_indexes[0]

        minimum_penalty_score = None
        best_proposal_list_groups = []

        proposal_to_cold = ProposalToMove(
            var_no=self._var_no,
            var_index=hot_index,
            pre_value=1,
            new_value=0)

        for new_hot_index in range(self._var_index, self._var_index + self._category_num):
            if new_hot_index != hot_index:
                proposal_to_hot = ProposalToMove(
                    var_no=self._var_no,
                    var_index=new_hot_index,
                    pre_value=0,
                    new_value=1)
                # ペナルティスコアで比較
                penalty_score = state.calculate_penalties([proposal_to_cold, proposal_to_hot])

                proposal_list = [proposal_to_cold, proposal_to_hot]
                if minimum_penalty_score is None or minimum_penalty_score > penalty_score:
                    best_proposal_list_groups = [proposal_list]
                    minimum_penalty_score = penalty_score
                elif minimum_penalty_score == penalty_score:
                    best_proposal_list_groups.append(proposal_list)

        return random.choice(best_proposal_list_groups)

    def propose_random_move_with_range(self, var_value_array: np.array, lower: np.double, upper: np.double) \
            -> List[ProposalToMove]:
        raise NotImplementedError('propose_random_move_with_range is not implemented!')

    def propose_random_move(self, var_value_array: np.array) -> List[ProposalToMove]:
        hot_indexes = \
            [index for index
             in range(self._var_index, self._var_index + self._category_num)
             if var_value_array[index] == 1]
        new_hot_indexes = \
            [index for index
             in range(self._var_index, self._var_index + self._category_num)
             if var_value_array[index] != 1]

        if len(hot_indexes) != 1 or len(new_hot_indexes) == 0:
            raise ValueError(f'Category Variable is not one hot encoding.')

        hot_index = hot_indexes[0]
        new_hot_index = random.choice(new_hot_indexes)

        return [ProposalToMove(
                    var_no=self._var_no,
                    var_index=hot_index,
                    pre_value=1,
                    new_value=0),
                ProposalToMove(
                    var_no=self._var_no,
                    var_index=new_hot_index,
                    pre_value=0,
                    new_value=1)]

    def values(self, var_value_array):
        array_indexes = var_value_array[self._var_index:(self._var_index + self.array_size())]
        category_index = [index for index, value in enumerate(array_indexes) if value == 1][0]
        return self._categories[category_index]

    def random_values(self):
        var_value = [0] * self._category_num
        var_value[random.randint(0, self._category_num - 1)] = 1
        return var_value
