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
from random import choice

from codableopt.solver.optimizer.entity.proposal_to_move import ProposalToMove
from codableopt.solver.optimizer.method.optimizer_method import OptimizerMethod
from codableopt.solver.optimizer.optimization_state import OptimizationState


class SampleMethod(OptimizerMethod):

    def __init__(self, steps: int):
        super().__init__(steps)

    def name(self) -> str:
        return 'sample_method'

    def initialize_of_step(self, state: OptimizationState, step: int):
        # ステップ開始時の処理なし
        pass

    def propose(self, state: OptimizationState, step: int) -> List[ProposalToMove]:
        # 変数から1つランダムに選択する
        solver_variable = choice(state.problem.variables)
        # 選択した変数をランダムに移動する解の遷移を提案する
        return solver_variable.propose_random_move(state)

    def judge(self, state: OptimizationState, step: int) -> bool:
        # 遷移前と遷移後のスコアを比較
        delta_energy = state.current_score.score - state.previous_score.score
        # ソルバー内はエネルギーが低い方が最適性が高いことを表している
        # マイナスの場合に解が改善しているため、提案を受け入れる
        return delta_energy < 0

    def finalize_of_step(self, state: OptimizationState, step: int):
        # ステップ終了時の処理なし
        pass
