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

import pandas as pd

from sample.usage.problem.matching_problem_generator import MatchingProblemGenerator
from codableopt import Problem, Objective, CategoryVariable, OptSolver, PenaltyAdjustmentMethod


# 最適化問題定義
print('generate Problem')
matching = MatchingProblemGenerator.generate(customer_num=1000, item_num=20)

print('start Optimization')
# 変数を定義
selected_items = \
    [CategoryVariable(name=f'item_for_{customer.name}', categories=matching.item_names)
     for customer in matching.customers]
selected_coupons = \
    [CategoryVariable(name=f'coupon_for_{customer.name}', categories=matching.coupon_names)
     for customer in matching.customers]


# 利益の期待値を計算する関数、目的関数に利用
def calculate_benefit(
        var_selected_items: List[str],
        var_selected_coupons: List[str],
        para_customer_features_df: pd.DataFrame,
        para_item_features_df: pd.DataFrame,
        para_coupon_features_df: pd.DataFrame,
        para_buy_rate_model):

    features_df = pd.concat([
        para_customer_features_df.reset_index(drop=True),
        para_item_features_df.loc[var_selected_items, :].reset_index(drop=True),
        para_coupon_features_df.loc[var_selected_coupons, :].reset_index(drop=True)
    ], axis=1)

    # 目的関数内で機械学習モデルを利用
    buy_rate = \
        [x[1] for x in para_buy_rate_model.predict_proba(features_df.drop(columns='item_cost'))]

    return sum(
        buy_rate *
        (features_df['item_price'] - features_df['item_cost'] - features_df['coupon_down_price']))


# 問題を設定
problem = Problem(is_max_problem=True)

# 目的関数を定義
problem += Objective(objective=calculate_benefit,
                     args_map={
                         'var_selected_items': selected_items,
                         'var_selected_coupons': selected_coupons,
                         'para_customer_features_df': matching.customer_features_df,
                         'para_item_features_df': matching.item_features_df,
                         'para_coupon_features_df': matching.coupon_features_df,
                         'para_buy_rate_model': matching.buy_rate_model
                     })

# 制約式を定義
for item in matching.items:
    # 必ず1人以上のカスタマーにアイテムを表示する制約式を設定
    problem += sum([(x == item.name) for x in selected_items]) >= 1
    # 同じアイテムの最大表示人数を制限する制約式を設定
    problem += sum([(x == item.name) for x in selected_items]) <= matching.max_display_num_per_item

for coupon in matching.coupons:
    # クーポンの最大発行数の制約式を設定
    problem += \
        sum([(x == coupon.name) for x in selected_coupons]) <= matching.max_display_num_per_coupon


# 最適化実施
print('start solve')
solver = OptSolver(round_times=1, debug=True, debug_unit_step=1000,
                   num_to_tune_penalty=100, num_to_select_init_answer=1)
method = PenaltyAdjustmentMethod(steps=40000)
answer, is_feasible = solver.solve(problem, method)

print(f'answer_is_feasible:{is_feasible}')
print(answer)
