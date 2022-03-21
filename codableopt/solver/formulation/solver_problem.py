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

from typing import List, Optional
from copy import deepcopy

import numpy as np

from codableopt.interface.system_problem import SystemProblem
from codableopt.interface.system_objective import SystemUserDefineObjective
from codableopt.interface.system_variable import SystemVariable
from codableopt.interface.system_constraint import SystemConstraints, SystemLinerConstraint
from codableopt.solver.formulation.objective.solver_objective import SolverObjective
from codableopt.solver.formulation.constraint.solver_constraints import SolverConstraints
from codableopt.solver.formulation.constraint.solver_liner_constraints \
    import SolverLinerConstraints
from codableopt.solver.formulation.constraint.solver_user_define_constraint \
    import SolverUserDefineConstraint
from codableopt.solver.formulation.variable.solver_variable_factory import SolverVariableFactory
from codableopt.solver.formulation.variable.solver_variable import SolverVariable
from codableopt.solver.formulation.args_map.solver_args_map import SolverArgsMap
from codableopt.solver.optimizer.entity.proposal_to_move import ProposalToMove


class SolverProblem:
    """Solverのための最適化問題クラス。
    SolverProblemは、問題の情報だけを扱い、状態は扱わない。（値オブジェクトである。）
    """

    NOT_MEET_INITIAL_PENALTY = 1.0

    def __init__(self, problem: SystemProblem):
        """OptimizationProblemをSolverで扱うためのオブジェクト生成関数。

        Args:
            problem (SystemProblem): 変換元の最適化問題
        """

        # about variable
        solver_variable_factory = SolverVariableFactory()
        self._variables: List[SolverVariable] = \
            [solver_variable_factory.generate(variable) for variable in problem.variables]

        # about constraint
        self._solver_constraints = SolverProblem._to_solver_constraint(
            problem.constraints, problem.variables, self._variables
        )

        # about objective
        objective = problem.objective
        if isinstance(objective, SystemUserDefineObjective):
            self._solver_objective = SolverObjective(
                objective.calc_objective,
                objective.calc_delta_objective,
                SolverArgsMap(objective.args_map, self._variables),
                objective.is_max_problem,
                objective.exist_delta_function)
        else:
            raise ValueError('objective_function is only support UserDefineObjectiveFunction!')

    def calc_objective(
            self,
            var_value_array: np.array,
            proposals: List[ProposalToMove],
            cashed_objective_score: Optional[np.double] = None) -> np.double:
        """目的関数値を計算する関数。

        Args:
            var_value_array (np.array): 目的関数値を計算するベース解答
            proposals (List[ProposalToMove]): 目的関数値の計算前にベース解答に適用する変数の遷移提案リスト
            cashed_objective_score(Optional[np.double]): キャッシュされている遷移前の目的関数値
        Returns:
            目的関数値
        """
        # 目的関数を計算
        return self._solver_objective.calc_objective(var_value_array, proposals,
                                                     cashed_objective_score)

    def calc_penalties(
            self,
            var_value_array: np.array,
            proposals: List[ProposalToMove],
            penalty_coefficients: np.array,
            cashed_liner_constraint_sums: np.array) -> List[np.double]:
        """制約ごとのペナルティ値を計算する関数。

        Args:
            var_value_array: 変数の値
            proposals: ペナルティ値を計算する前に適用する変数の遷移提案リスト
            penalty_coefficients: ペナルティ係数
            cashed_liner_constraint_sums: 制約式の左項の合計値のキャッシュ（変数の遷移提案前の値）

        Returns:
            ペナルティ値
        """
        self.apply_proposal_to_liner_constraint_sums(proposals, cashed_liner_constraint_sums)
        vio_amounts = self.calc_violation_amounts(var_value_array, cashed_liner_constraint_sums)
        self.cancel_proposal_to_liner_constraint_sums(proposals, cashed_liner_constraint_sums)

        return [x * y for x, y in zip(vio_amounts, penalty_coefficients)]

    def calc_violation_amounts(
            self,
            var_value_array: np.array,
            cashed_liner_constraint_sums=None) -> List[np.double]:
        """全ての制約式の制約違反量を計算する関数。

        Args:
            var_value_array: 各変数の値
            cashed_liner_constraint_sums: 制約式の左項の合計値のキャッシュ（変数の遷移提案前の値）
        Returns:
            全ての制約式の制約違反量
        """
        liner_constraint_sums = cashed_liner_constraint_sums
        if liner_constraint_sums is None:
            liner_constraint_sums = self.calc_liner_constraint_sums(var_value_array)

        return self._solver_constraints.calc_violation_amounts(var_value_array,
                                                               liner_constraint_sums)

    def calc_liner_constraint_sums(self, var_value_array: np.array) -> np.array:
        """全ての制約式の左項の合計値を計算する関数。

        Args:
            var_value_array (np.array): 制約式を計算する解答

        Returns:
            全ての制約式の左項の合計値
        """
        return self._solver_constraints.liner_constraints.calc_constraint_sums(var_value_array)

    def apply_proposal_to_liner_constraint_sums(
            self,
            proposals: List[ProposalToMove],
            cashed_constraint_sums: np.array) -> Optional[np.array]:
        """全ての制約式の左項の合計値に、変数の遷移提案を適用し、値を更新する関数。

        Args:
            proposals: ペナルティ値を計算する前に適用する変数の遷移提案リスト
            cashed_constraint_sums: 制約式の左項の合計値のキャッシュ（変数の遷移提案前の値）

        Returns:
            更新した制約式の左項の合計値
        """
        if self._solver_constraints.liner_constraints is None:
            return None

        return self._solver_constraints.liner_constraints\
            .apply_proposal_to_constraint_sums(proposals, cashed_constraint_sums)

    def cancel_proposal_to_liner_constraint_sums(
            self,
            proposals: List[ProposalToMove],
            cashed_constraint_sums: np.array) -> Optional[np.array]:
        """全ての制約式の左項の合計値に、適用した変数の遷移提案をキャンセルし、値を更新する関数。

        Args:
            proposals: ペナルティ値を計算する前に適用する変数の遷移提案リスト
            cashed_constraint_sums: 制約式の左項の合計値のキャッシュ（変数の遷移提案前の値）

        Returns:
            更新した制約式の左項の合計値
        """
        if self._solver_constraints.liner_constraints is None:
            return None

        return self._solver_constraints.liner_constraints\
            .cancel_proposal_to_constraint_sums(proposals, cashed_constraint_sums)

    def to_answer(self, var_value_array: np.array):
        """呼び出し元の問題オブジェクトに基づき、解答を辞書型に変換して返す関数。

        Args:
            var_value_array (np.array): 変換元の解答

        Returns:
            辞書型の解答、keyが変数名、valueが変数の値
        """
        return {variable.name: variable.values(var_value_array) for variable in self._variables}

    @property
    def variables(self) -> List[SolverVariable]:
        return self._variables

    @property
    def constraints(self) -> SolverConstraints:
        return self._solver_constraints

    @staticmethod
    def _to_solver_constraint(
            constraints: SystemConstraints,
            variables: List[SystemVariable],
            solver_variables: List[SolverVariable]):
        # about liner_constraints
        liner_constraints = deepcopy(constraints.liner_constraints)
        for variable in variables:
            liner_constraints.extend(variable.to_constraints_of_range())
        liner_coefficients = SolverProblem._to_coefficients_of_constraints(
            constraints=liner_constraints,
            variables=variables)
        liner_constraints = SolverLinerConstraints(liner_constraints, liner_coefficients)

        # about user_define_constraints
        user_define_constraints = []
        for sys_user_define_constraint in deepcopy(constraints.user_define_constraints):
            user_define_constraints.append(SolverUserDefineConstraint(
                sys_user_define_constraint.constraint_function,
                SolverArgsMap(sys_user_define_constraint.args_map, solver_variables)
            ))

        return SolverConstraints(liner_constraints, user_define_constraints)

    @staticmethod
    def _to_coefficients_of_constraints(
            constraints: List[SystemLinerConstraint],
            variables: List[SystemVariable]):
        """制約式と変数の上界値下界値の制約を各制約と各変数の係数の2次元配列に変換する関数。

        Args:
            constraints: 変換する制約式のリスト
            variables: 上界値下界値の制約を取り出す変数のリスト

        Returns:
            各制約と各変数の係数の2次元配列
        """
        constraints_coefficients = []
        for constraint in constraints:
            constraint_coefficients = []
            for variable in variables:
                constraint_coefficients.extend(variable.extract_coefficients(constraint))

            constraints_coefficients.append(np.array(constraint_coefficients))

        return np.array(constraints_coefficients)
