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
from itertools import combinations

from codableopt import Problem, Objective, CategoryVariable, OptSolver, PenaltyAdjustmentMethod


# 距離生成関数
def generate_distances(args_place_names):
    # ポイント間の距離を生成
    generated_distances = {}
    for point_to_point in combinations(['start'] + args_place_names, 2):
        distance_value = random.randint(20, 40)
        generated_distances[point_to_point] = distance_value
        generated_distances[tuple(reversed(point_to_point))] = distance_value
    for x in ['start'] + args_place_names:
        generated_distances[(x, x)] = 0

    return generated_distances


# 単純なTSP問題
# ルート/ポイント数を設定
PLACE_NUM = 30
# 行き先名を生成
destination_names = [f'destination_{no}' for no in range(PLACE_NUM)]
# ポイント名を生成
place_names = [f'P{no}' for no in range(PLACE_NUM)]
# 距離を生成
distances = generate_distances(place_names)

# ルート変数を定義
destinations = [CategoryVariable(name=destination_name, categories=place_names)
                for destination_name in destination_names]

# 問題を設定
problem = Problem(is_max_problem=False)


# 目的関数として、距離を計算する関数を定義
def calc_distance(var_destinations, para_distances):
    return sum([para_distances[(x, y)] for x, y
                in zip(['start'] + var_destinations, var_destinations + ['start'])])


# 目的関数を定義
problem += Objective(objective=calc_distance,
                     args_map={'var_destinations': destinations, 'para_distances': distances})

# 必ず1度以上、全てのポイントに到達する制約式を追加
for place_name in place_names:
    problem += sum([(destination == place_name)
                   for destination in destinations]) >= 1

# 最適化実施
solver = OptSolver(round_times=4, debug=True, debug_unit_step=1000)
method = PenaltyAdjustmentMethod(steps=10000, delta_to_update_penalty_rate=0.9)
answer, is_feasible = solver.solve(problem, method, n_jobs=-1)

print(f'answer_is_feasible:{is_feasible}')
root = ['start'] + [answer[root] for root in destination_names] + ['start']
print(f'root: {" -> ".join(root)}')
