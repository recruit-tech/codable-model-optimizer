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
    CategoryVariable, OptSolver, PenaltyAdjustmentMethod, UserDefineConstraint

# 変数を定義
x = IntVariable(name='x', lower=np.double(0.0), upper=np.double(2))
y = DoubleVariable(name='y', lower=np.double(0.0), upper=None)
z = CategoryVariable(name='z', categories=['a', 'b', 'c'])


# 目的関数に指定する関数
def objective_function(var_x, var_y, var_z, parameters):
    obj_value = parameters['coef_x'] * var_x + parameters['coef_y'] * var_y

    if var_z == 'a':
        obj_value += 10.0
    elif var_z == 'b':
        obj_value += 8.0
    else:
        # var_z == 'c'
        obj_value -= 3.0

    return obj_value


# 問題を設定
problem = Problem(is_max_problem=True)

# 目的関数を定義
problem += Objective(objective=objective_function,
                     args_map={'var_x': x, 'var_y': y, 'var_z': z,
                               'parameters': {'coef_x': -3.0, 'coef_y': 4.0}})


# 制約式を定義
def udf_constraint_function(var_x, var_y, var_z):
    violation_amount = 2 * var_x + 4 * var_y - 8
    if var_z == 'a':
        violation_amount += 2
    else:
        violation_amount += 3

    if violation_amount <= 0:
        return 0
    else:
        return violation_amount


# 現状、UserDefineConstraintは非推奨の機能
constant = UserDefineConstraint(udf_constraint_function,
                                args_map={'var_x': x, 'var_y': y, 'var_z': z},
                                constraint_name='user_define_constraint')
problem += constant
problem += 2 * x - y + 2 * (z == 'b') > 3

# 問題を確認
print(problem)

# ソルバーを生成
solver = OptSolver(debug=True, debug_unit_step=1000)

# ソルバー内で使う最適化手法を生成
method = PenaltyAdjustmentMethod(steps=40000, proposed_rate_of_random_movement=1)

# 最適化実施
answer, is_feasible = solver.solve(problem, method)
print(f'answer:{answer}, answer_is_feasible:{is_feasible}')
