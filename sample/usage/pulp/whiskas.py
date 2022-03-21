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

from codableopt import Problem, Objective, IntVariable, OptSolver, PenaltyAdjustmentMethod


Ingredients = ['CHICKEN', 'BEEF', 'MUTTON', 'RICE', 'WHEAT', 'GEL']
costs = \
    {'CHICKEN': 0.013, 'BEEF': 0.008, 'MUTTON': 0.010, 'RICE': 0.002, 'WHEAT': 0.005, 'GEL': 0.001}
proteinPercent = \
    {'CHICKEN': 0.100, 'BEEF': 0.200, 'MUTTON': 0.150, 'RICE': 0.000, 'WHEAT': 0.040, 'GEL': 0.000}
fatPercent = \
    {'CHICKEN': 0.080, 'BEEF': 0.100, 'MUTTON': 0.110, 'RICE': 0.010, 'WHEAT': 0.010, 'GEL': 0.000}
fibrePercent = \
    {'CHICKEN': 0.001, 'BEEF': 0.005, 'MUTTON': 0.003, 'RICE': 0.100, 'WHEAT': 0.150, 'GEL': 0.000}
saltPercent = \
    {'CHICKEN': 0.002, 'BEEF': 0.005, 'MUTTON': 0.007, 'RICE': 0.002, 'WHEAT': 0.008, 'GEL': 0.000}

# 変数を定義
x = [IntVariable(name=f'x_{ingredient}', lower=0, upper=100) for ingredient in Ingredients]

# 問題を設定
problem = Problem(is_max_problem=False)


def objective_function(var_x, para_costs):
    return np.dot(var_x, para_costs)


# 目的関数を定義
problem += Objective(objective=objective_function,
                     args_map={'var_x': x,
                               'para_costs': [costs[ingredient] for ingredient in Ingredients]})

# 制約式を定義
problem += sum(x) == 100
problem += sum([proteinPercent[ingredient] * x_ for x_, ingredient in zip(x, Ingredients)]) >= 8.0
problem += sum([fatPercent[ingredient] * x_ for x_, ingredient in zip(x, Ingredients)]) >= 6.0
problem += sum([fibrePercent[ingredient] * x_ for x_, ingredient in zip(x, Ingredients)]) <= 2.0
problem += sum([saltPercent[ingredient] * x_ for x_, ingredient in zip(x, Ingredients)]) <= 0.4


# 問題を確認
print(problem)

# ソルバーを生成
solver = OptSolver()

# ソルバー内で使う最適化手法を生成
method = PenaltyAdjustmentMethod(steps=100000)

# 最適化実施
answer, is_feasible = solver.solve(problem, method)
print(f'answer:{answer}, answer_is_feasible:{is_feasible}')
