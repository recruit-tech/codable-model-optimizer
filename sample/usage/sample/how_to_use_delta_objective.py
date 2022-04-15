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

from codableopt import Problem, Objective, IntVariable, DoubleVariable, \
    CategoryVariable, OptSolver, PenaltyAdjustmentMethod

# 変数を定義
x = IntVariable(name='x', lower=0, upper=None)
y = DoubleVariable(name='y', lower=np.double(0.0), upper=None)
z = CategoryVariable(name='z', categories=['a', 'b', 'c'])


# 目的関数に指定する関数
def objective_function(var_x, var_y, var_z, parameters):
    value = parameters['coef_x'] * var_x + parameters['coef_y'] * var_y

    if var_z == 'a':
        value += 10.0
    elif var_z == 'b':
        value += 8.0
    else:
        # var_z == 'c'
        value -= 3.0

    return value


# 差分計算による目的関数に指定する関数
def delta_objective_function(
        pre_var_x,
        pre_var_y,
        pre_var_z,
        var_x,
        var_y,
        var_z,
        parameters):
    delta_value = 0
    if pre_var_x != var_x:
        delta_value += parameters['coef_x'] * (var_x - pre_var_x)

    if pre_var_y != var_y:
        delta_value += parameters['coef_y'] * (var_y - pre_var_y)

    if pre_var_z != var_z:
        if pre_var_z == 'a':
            delta_value -= 10.0
        elif pre_var_z == 'b':
            delta_value -= 8.0
        else:
            # pre_z == 'c'
            delta_value += 3.0

        if var_z == 'a':
            delta_value += 10.0
        elif var_z == 'b':
            delta_value += 8.0
        else:
            # z == 'c'
            delta_value -= 3.0

    return delta_value


# 問題を設定
problem = Problem(is_max_problem=True)

# 目的関数を定義
problem += Objective(objective=objective_function,
                     args_map={'var_x': x, 'var_y': y, 'var_z': z,
                               'parameters': {'coef_x': -3.0, 'coef_y': 4.0}},
                     delta_objective=delta_objective_function)

# 制約式を定義
problem += 2 * x + 4 * y + 2 * (z == 'a') + 3 * (z == ('b', 'c')) <= 8
problem += 2 * x - y + 2 * (z == 'b') > 3

# ソルバーを生成
solver = OptSolver(round_times=2, debug=True, debug_unit_step=1000)

# ソルバー内で使う最適化手法を生成
method = PenaltyAdjustmentMethod(steps=40000)

# 最適化実施
answer, is_feasible = solver.solve(problem, method)
print(f'answer:{answer}, answer_is_feasible:{is_feasible}')
