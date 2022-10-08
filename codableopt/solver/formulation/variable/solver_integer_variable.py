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

import math
from typing import List

import numpy as np
import sys
from random import randint

from codableopt.solver.formulation.variable.solver_variable import SolverVariable
from codableopt.solver.optimizer.optimization_state import OptimizationState
from codableopt.solver.optimizer.entity.proposal_to_move import ProposalToMove


class SolverIntegerVariable(SolverVariable):

    def __init__(
            self,
            var_no: int,
            var_index: int,
            name: str,
            lower: int,
            upper: int):
        super(SolverIntegerVariable, self).__init__(var_no, var_index, name)
        self._lower = lower
        self._upper = upper

    def array_size(self) -> int:
        return 1

    def propose_low_penalty_move(self, state: OptimizationState) -> List[ProposalToMove]:
        if state.problem.is_no_constraint:
            raise ValueError('propose_low_penalty_move function need constraint to use!')

        prev_value = state.var_array[self._var_index]

        lower, upper = state.problem.constraints.liner_constraints.calc_var_range_in_feasible(
            state.var_array,
            var_index=self._var_index,
            cashed_constraint_sums=state.cashed_liner_constraint_sums,
            is_int=True)

        if upper is None and lower is None:
            raise ValueError('var no={} had no constraint!'.format(self._var_no))
        elif lower is None:
            lower = prev_value - abs(upper - prev_value) - 1
            lower = math.ceil(lower)
        elif upper is None:
            upper = lower + abs(prev_value - lower) + 1
            upper = math.floor(upper)

        if upper < lower:
            if upper == prev_value:
                new_val = lower
            elif lower == prev_value:
                new_val = upper
            else:
                base_lower, base_upper = self._lower, self._upper
                if upper is not None and base_lower is not None \
                        and upper < base_lower:
                    upper = base_lower
                if lower is not None and base_upper is not None \
                        and lower > base_upper:
                    lower = base_upper

                # 満たせない制約がある場合、より制約を満たす方に向かう値を選択する
                # ただし、変数の範囲制約は優先して適用する

                # bad score計算（制約式の違反量）
                proposal_to_upper = ProposalToMove(
                    var_no=self._var_no,
                    var_index=self._var_index,
                    pre_value=prev_value,
                    new_value=upper)
                upper_penalty_scores = state.calculate_penalties([proposal_to_upper])
                penalty_num_of_upper = sum([1 if x > 0 else 0 for x in upper_penalty_scores])

                proposal_to_lower = ProposalToMove(
                    var_no=self._var_no,
                    var_index=self._var_index,
                    pre_value=prev_value,
                    new_value=lower)
                lower_penalty_scores = state.calculate_penalties([proposal_to_lower])
                penalty_num_of_lower = sum([1 if x > 0 else 0 for x in lower_penalty_scores])

                if penalty_num_of_upper > penalty_num_of_lower:
                    new_val = lower
                elif penalty_num_of_upper < penalty_num_of_lower:
                    new_val = upper
                else:
                    if sum(upper_penalty_scores) > sum(lower_penalty_scores):
                        new_val = lower
                    else:
                        new_val = upper

        elif upper > lower:
            if upper == prev_value:
                if randint(1, 2) == 1 or upper - 1 == lower:
                    new_val = lower
                else:
                    new_val = randint(lower, upper - 1)

            elif lower == prev_value:
                if randint(1, 2) == 1 or upper == lower + 1:
                    new_val = upper
                else:
                    new_val = randint(lower + 1, upper)

            else:
                random_num = randint(1, 3)
                if random_num == 1:
                    new_val = lower
                elif random_num == 2:
                    new_val = upper
                else:
                    new_val = randint(lower, upper)
        else:
            # upper == lowerのケース
            new_val = upper

        if new_val > sys.maxsize:
            new_val = sys.maxsize
        if new_val < -sys.maxsize:
            new_val = -sys.maxsize

        if self._lower is not None and new_val < self._lower:
            new_val = self._lower
        elif self._upper is not None and new_val > self._upper:
            new_val = self._upper

        if prev_value == new_val:
            return []

        return [ProposalToMove(
            var_no=self._var_no,
            var_index=self._var_index,
            pre_value=prev_value,
            new_value=new_val)]

    def propose_random_move_with_range(self, var_value_array: np.array, lower: np.double, upper: np.double) \
            -> List[ProposalToMove]:
        prev_value = var_value_array[self._var_index]
        new_val = randint(math.floor(lower), math.ceil(upper))

        return [ProposalToMove(
            var_no=self._var_no,
            var_index=self._var_index,
            pre_value=prev_value,
            new_value=new_val)]

    def propose_random_move(self, var_value_array: np.array) -> List[ProposalToMove]:
        prev_value = var_value_array[self._var_index]
        lower, upper = self._lower, self._upper

        if upper is None and lower is None:
            lower = -np.double(SolverVariable.VALUE_WHEN_NO_LIMIT)
            upper = np.double(SolverVariable.VALUE_WHEN_NO_LIMIT)
        elif lower is None:
            lower = prev_value - abs(upper - prev_value) - 1
            lower = math.ceil(lower)
        elif upper is None:
            upper = lower + abs(prev_value - lower) + 1
            upper = math.floor(upper)

        if upper == prev_value:
            if randint(1, 2) == 1 or upper - 1 == lower:
                new_val = lower
            else:
                new_val = randint(lower, upper - 1)
        elif lower == prev_value:
            if randint(1, 2) == 1 or upper == lower + 1:
                new_val = upper
            else:
                new_val = randint(lower + 1, upper)
        else:
            random_num = randint(1, 3)
            if random_num == 1:
                new_val = lower
            elif random_num == 2:
                new_val = upper
            else:
                new_val = randint(lower, upper)

        if new_val > sys.maxsize:
            new_val = sys.maxsize
        if new_val < -sys.maxsize:
            new_val = -sys.maxsize

        if prev_value == new_val:
            return []

        return [ProposalToMove(
            var_no=self._var_no,
            var_index=self._var_index,
            pre_value=prev_value,
            new_value=new_val)]

    def decode(self, var_value_array):
        return int(var_value_array[self._var_index])

    def random_values(self):
        upper = self._upper
        lower = self._lower
        if lower is None:
            lower = -np.double(SolverVariable.VALUE_WHEN_NO_LIMIT)
        if upper is None:
            upper = np.double(SolverVariable.VALUE_WHEN_NO_LIMIT)

        return [randint(lower, upper)]

    def encode(self, value: int) -> np.array:
        return np.array([value])
