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

import random

from codableopt import Problem, Objective, IntVariable, OptSolver, \
    PenaltyAdjustmentMethod


# アイテム数、最大重量を設定
item_num = 40
max_weight = 1000
# アイテム名を生成
item_names = [f'item_{no}' for no in range(item_num)]
# アイテムのバリューと重量を設定
parameter_item_values = [random.randint(10, 50) for _ in item_names]
parameter_item_weights = [random.randint(20, 40) for _ in item_names]

# アイテムのBool変数を定義
var_item_flags = [IntVariable(name=item_name, lower=0, upper=1) for item_name in item_names]


# 目的関数として、距離を計算する関数を定義
def calculate_total_values(item_flags, item_values):
    return sum([flag * value for flag, value in zip(item_flags, item_values)])


# 問題を設定
problem = Problem(is_max_problem=True)

# 目的関数を定義
problem += Objective(objective=calculate_total_values,
                     args_map={'item_flags': var_item_flags, 'item_values': parameter_item_values})

# 重量制限の制約式を追加
problem += sum([item_flag * weight for item_flag,
                weight in zip(var_item_flags,
                              parameter_item_weights)]) <= max_weight

# 最適化実施
solver = OptSolver(round_times=4, debug=True, debug_unit_step=1000)
method = PenaltyAdjustmentMethod(steps=10000)
answer, is_feasible = solver.solve(problem, method, n_jobs=-1)

print(f'answer_is_feasible:{is_feasible}')
print(f'select_items: 'f'{", ".join([x for x in answer.keys() if answer[x] == 1])}')
