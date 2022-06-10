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

from codableopt import Problem, Objective, CategoryVariable, OptSolver, PenaltyAdjustmentMethod


# 問題のパラメータ設定
CUSTOMER_NUM = 30
LIMIT_NUM_PER_taxi = 4
TAXI_NUM = 10

customer_names = [f'CUS_{i}' for i in range(CUSTOMER_NUM)]
taxi_names = [f'taxi_{i}' for i in range(TAXI_NUM)]

# 年齢・性別のマッチング
customers_age = [random.choice(['20-30', '30-60', '60-']) for _ in customer_names]
customers_sex = [random.choice(['m', 'f']) for _ in customer_names]

# 顧客の車割り当て変数を作成
x = [CategoryVariable(name=x, categories=taxi_names) for x in customer_names]

# 問題を設定
problem = Problem(is_max_problem=True)


# 目的関数として、距離を計算する関数を定義
def calc_matching_score(var_x, para_taxi_names, para_customers_age, para_customers_sex):
    score = 0
    for para_taxi_name in para_taxi_names:
        customers_in_taxi = [(age, sex) for var_bit_x, age, sex
                            in zip(var_x, para_customers_age, para_customers_sex)
                            if var_bit_x == para_taxi_name]
        num_in_taxi = len(customers_in_taxi)
        if num_in_taxi > 1:
            score += num_in_taxi - len(set([age for age, _ in customers_in_taxi]))
            score += num_in_taxi - len(set([sex for _, sex in customers_in_taxi]))

    return score


# 目的関数を定義
problem += Objective(objective=calc_matching_score,
                     args_map={'var_x': x,
                               'para_taxi_names': taxi_names,
                               'para_customers_age': customers_age,
                               'para_customers_sex': customers_sex})

# 必ず1度以上、全てのポイントに到達する制約式を追加
for taxi_name in taxi_names:
    problem += sum([(bit_x == taxi_name) for bit_x in x]) <= LIMIT_NUM_PER_taxi

# 最適化実施
solver = OptSolver(round_times=4, debug=True, debug_unit_step=1000)
method = PenaltyAdjustmentMethod(steps=10000)
answer, is_feasible = solver.solve(problem, method, n_jobs=-1)

print(f'answer_is_feasible:{is_feasible}')
print(f'answer: {answer}')
