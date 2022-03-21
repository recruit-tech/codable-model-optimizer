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

from typing import Sequence
import numpy as np

from codableopt.solver.optimizer.entity.score_info import ScoreInfo
from codableopt.solver.optimizer.entity.proposal_to_move import ProposalToMove


class OptimizationState:

    def __init__(self, problem, init_var_value_array: np.array):
        self._problem = problem

        self._var_array = init_var_value_array.copy()
        self._best_var_array = self._var_array

        self._proposals = []

        objective_score = self._problem.calc_objective(self._var_array, [])
        self._cashed_objective_score = objective_score

        cashed_liner_constraint_sums = \
            self._problem.constraints.liner_constraints.calc_constraint_sums(self._var_array)
        self._cashed_liner_constraint_sums = cashed_liner_constraint_sums
        self._penalty_coefficients = self._problem.constraints.init_penalty_coefficients.copy()
        # tune前は1に仮置き
        self._constraints_scale = np.array([1.0 for _ in self._penalty_coefficients])

        penalty_scores = self.calculate_penalties([])
        penalty_scores = sum(penalty_scores)

        self._previous_score = ScoreInfo(objective_score, penalty_scores)
        self._current_score = ScoreInfo(objective_score, penalty_scores)
        self._best_score = ScoreInfo(objective_score, penalty_scores)

        self._exist_feasible_answer = False

    def propose(self, proposals: Sequence[ProposalToMove]):
        objective_score = self._problem.calc_objective(self._var_array,
                                                       proposals,
                                                       self._cashed_objective_score)
        penalty_scores = self.calculate_penalties(proposals)
        penalty_score = sum(penalty_scores)

        self._proposals = proposals
        self._current_score.set_scores(objective_score, penalty_score)

    def apply_proposal(self):
        # 解を更新
        for proposal in self._proposals:
            self._var_array[proposal.var_index] = proposal.new_value

        # スコアを更新
        self._previous_score.set_scores(self._current_score.objective_score,
                                        self._current_score.penalty_score)

        # キャッシュを更新
        self._cashed_objective_score = self._current_score.objective_score
        self._problem.apply_proposal_to_liner_constraint_sums(self._proposals,
                                                              self._cashed_liner_constraint_sums)

        is_best = False
        # 実行可能解がはじめて見つかった時
        if not self._exist_feasible_answer and self._current_score.penalty_score == 0:
            self._exist_feasible_answer = True
            is_best = True

        # 実行可能解ではないが、最適性が高まった時
        elif not self._exist_feasible_answer \
                and self._current_score.score < self._best_score.score:
            is_best = True

        # 最適性が高まった時
        elif self._current_score.penalty_score == 0 \
                and self._current_score.score < self._best_score.score:
            is_best = True

        # 最適解を更新
        if is_best:
            self._best_var_array = self._var_array.copy()
            self._best_score.set_scores(self._current_score.objective_score,
                                        self._current_score.penalty_score)

    def cancel_propose(self):
        # スコアを戻す
        self._current_score.set_scores(self._previous_score.objective_score,
                                       self._previous_score.penalty_score)

    def calculate_penalties(self, proposals):
        return self._problem.calc_penalties(
            self._var_array, proposals, self._penalty_coefficients,
            self._cashed_liner_constraint_sums)

    def to_answer(self, var_value_array: np.array):
        return self._problem.to_answer(var_value_array)

    def tune_penalty(self, var_value_arrays, penalty_strength):
        objective_scores = \
            [self._problem.calc_objective(var_array, []) for var_array in var_value_arrays]
        object_diff_score = np.double(max(objective_scores) - min(objective_scores))

        vio_amounts_array = \
            np.array([self._problem.calc_violation_amounts(var_array) for var_array in var_value_arrays])

        if object_diff_score > 0:
            vio_means = vio_amounts_array.mean(axis=0)
            self._constraints_scale = [1.0 if x == 0 else x for x in vio_means]
            self._penalty_coefficients = \
                [1.0 if x == 0 else object_diff_score / x for x in vio_means]
            self._penalty_coefficients = [x * penalty_strength for x in self._penalty_coefficients]

    @property
    def problem(self):
        return self._problem

    @property
    def var_array(self):
        return self._var_array

    @property
    def best_var_array(self):
        return self._best_var_array

    @property
    def previous_score(self):
        return self._previous_score

    @property
    def current_score(self):
        return self._current_score

    @property
    def best_score(self):
        return self._best_score

    @property
    def cashed_objective_score(self):
        return self._cashed_objective_score

    @property
    def cashed_liner_constraint_sums(self):
        return self._cashed_liner_constraint_sums

    @property
    def exist_feasible_answer(self):
        return self._exist_feasible_answer

    @property
    def penalty_coefficients(self):
        return self._penalty_coefficients

    @penalty_coefficients.setter
    def penalty_coefficients(self, penalty_coefficients):
        self._penalty_coefficients = penalty_coefficients

    @property
    def constraints_scale(self):
        return self._constraints_scale

    @constraints_scale.setter
    def constraints_scale(self, constraints_scale):
        self._constraints_scale = constraints_scale
