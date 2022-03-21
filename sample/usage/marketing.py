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

import numpy as np

from codableopt import Problem, Objective, IntVariable, OptSolver, \
    PenaltyAdjustmentMethod


# 顧客数、CM数を設定
CUSTOMER_NUM = 1000
CM_NUM = 100
SELECTED_CM_LIMIT = 5
# 顧客がCMを見る確率を生成
view_rates = np.random.rand(CUSTOMER_NUM, CM_NUM) / 30

# CMの放送有無の変数を定義
cm_times = [IntVariable(name=f'cm_{no}', lower=0, upper=1) for no in range(CM_NUM)]

# 問題を設定
problem = Problem(is_max_problem=True)


# 目的関数として、CMを1度でも見る確率を計算
def calculate_view_rate_sum(var_cm_times, para_non_view_rates):
    selected_cm_noes = \
        [cm_no for cm_no, var_cm_time in enumerate(var_cm_times) if var_cm_time == 1]
    view_rate_per_customers = np.ones(para_non_view_rates.shape[0]) \
                              - np.prod(para_non_view_rates[:, selected_cm_noes], axis=1)
    return np.sum(view_rate_per_customers)


# 目的関数を定義
problem += Objective(objective=calculate_view_rate_sum,
                     args_map={'var_cm_times': cm_times,
                               'para_non_view_rates': np.ones(view_rates.shape) - view_rates})

# CMの選択数の制約式を追加
problem += sum(cm_times) <= SELECTED_CM_LIMIT

print(problem)

# 最適化実施
solver = OptSolver(round_times=2, debug=True, debug_unit_step=1000)
method = PenaltyAdjustmentMethod(steps=50000)
answer, is_feasible = solver.solve(problem, method, n_jobs=-1)

print(f'answer_is_feasible:{is_feasible}')
print(f'selected cm: {[cm_name for cm_name in answer.keys() if answer[cm_name] == 1]}')
