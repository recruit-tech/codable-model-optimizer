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
from dataclasses import dataclass
import random

import numpy as np
import pandas as pd
from sklearn import tree


ATTRIBUTE_PATTERN_NUM = 10


@dataclass(frozen=True)
class Customer:
    ATTRIBUTE_A_MASTER = list(range(ATTRIBUTE_PATTERN_NUM))
    ATTRIBUTE_B_MASTER = list(range(ATTRIBUTE_PATTERN_NUM))
    ATTRIBUTE_C_MASTER = list(range(ATTRIBUTE_PATTERN_NUM))
    name: str
    attribute_a: int
    attribute_b: int
    attribute_c: int


@dataclass(frozen=True)
class Item:
    ATTRIBUTE_A_MASTER = list(range(ATTRIBUTE_PATTERN_NUM))
    ATTRIBUTE_B_MASTER = list(range(ATTRIBUTE_PATTERN_NUM))
    ATTRIBUTE_C_MASTER = list(range(ATTRIBUTE_PATTERN_NUM))
    name: str
    price: int
    cost: int
    attribute_a: int
    attribute_b: int
    attribute_c: int


@dataclass(frozen=True)
class Coupon:
    DOWN_PRICE_TYPE_MASTER = [0, 1000, 3000, 5000]
    name: str
    down_price: int


class DataGenerator:

    def __init__(self):
        self._attribute_matching_scores = \
            np.random.randint(0, 2000, (ATTRIBUTE_PATTERN_NUM, ATTRIBUTE_PATTERN_NUM))

        self._customer_attribute_a_scores = np.random.randint(0, 2000, ATTRIBUTE_PATTERN_NUM)
        self._customer_attribute_b_scores = np.random.randint(0, 2000, ATTRIBUTE_PATTERN_NUM)
        self._customer_attribute_c_scores = np.random.randint(0, 2000, ATTRIBUTE_PATTERN_NUM)

        self._item_attribute_a_scores = np.random.randint(0, 2000, ATTRIBUTE_PATTERN_NUM)
        self._item_attribute_b_scores = np.random.randint(0, 2000, ATTRIBUTE_PATTERN_NUM)
        self._item_attribute_c_scores = np.random.randint(0, 2000, ATTRIBUTE_PATTERN_NUM)

    def generate_train_dataset(self, data_num: int) -> pd.DataFrame:
        dataset = []
        all_coupons = self.generate_all_coupons()
        for no in range(data_num):
            customer = self.generate_random_customer(f'customer_{no}')
            item = self.generate_random_item(f'item_{no}')
            coupon = random.choice(all_coupons)
            buy_int_flag = self._simulate_buy_flg(customer, item, coupon)
            dataset.append([customer.attribute_a,
                            customer.attribute_b,
                            customer.attribute_c,
                            item.price,
                            item.attribute_a,
                            item.attribute_b,
                            item.attribute_c,
                            coupon.down_price,
                            buy_int_flag])
        return pd.DataFrame(
            dataset,
            columns=[
                'customer_attribute_a',
                'customer_attribute_b',
                'customer_attribute_c',
                'item_price',
                'item_attribute_a',
                'item_attribute_b',
                'item_attribute_c',
                'coupon_down_price',
                'buy_int_flag'])

    @staticmethod
    def generate_random_customer(customer_name: str) -> Customer:
        return Customer(
            name=customer_name,
            attribute_a=random.choice(Customer.ATTRIBUTE_A_MASTER),
            attribute_b=random.choice(Customer.ATTRIBUTE_B_MASTER),
            attribute_c=random.choice(Customer.ATTRIBUTE_C_MASTER)
        )

    @staticmethod
    def generate_random_item(item_name: str) -> Item:
        price = random.choice(list(range(10000, 30000, 1000)))
        cost = int(price * (1 - 0.1 - random.random() / 5))
        return Item(
            name=item_name,
            price=price,
            cost=cost,
            attribute_a=random.choice(Item.ATTRIBUTE_A_MASTER),
            attribute_b=random.choice(Item.ATTRIBUTE_B_MASTER),
            attribute_c=random.choice(Item.ATTRIBUTE_C_MASTER)
        )

    @staticmethod
    def generate_all_coupons() -> List[Coupon]:
        return [Coupon(name=f'coupon_{x}', down_price=x)
                for x in Coupon.DOWN_PRICE_TYPE_MASTER]

    def _simulate_buy_flg(
            self,
            customer: Customer,
            item: Item,
            coupon: Coupon) -> int:
        value_price = 0
        # value of customer_attribute
        value_price += self._customer_attribute_a_scores[customer.attribute_a]
        value_price += self._customer_attribute_b_scores[customer.attribute_b]
        value_price += self._customer_attribute_c_scores[customer.attribute_c]
        # value of item_attribute
        value_price += self._item_attribute_a_scores[item.attribute_a]
        value_price += self._item_attribute_b_scores[item.attribute_b]
        value_price += self._item_attribute_c_scores[item.attribute_c]
        # value of matching customer and item
        for customer_attribute in [
                customer.attribute_a,
                customer.attribute_b,
                customer.attribute_c]:
            for item_attribute in [
                    item.attribute_a,
                    item.attribute_b,
                    item.attribute_c]:
                value_price += self._attribute_matching_scores[customer_attribute, item_attribute]
        # minus price
        value_price -= (item.price - coupon.down_price)
        # calculate buy_rate
        buy_rate = 1 / (1 + np.exp(- (value_price - 3000) / 5000))
        return 1 if random.random() <= buy_rate else 0


@dataclass(frozen=True)
class MatchingProblem:
    customers: List[Customer]
    items: List[Item]
    coupons: List[Coupon]
    customer_names: List[str]
    item_names: List[str]
    coupon_names: List[str]
    buy_rate_model: tree.DecisionTreeClassifier


class MatchingProblemGenerator:

    def __init__(self):
        pass

    @staticmethod
    def generate(customer_num: int, item_num: int) -> MatchingProblem:
        # 学習モデルの生成
        data_generator = DataGenerator()
        train_df = data_generator.generate_train_dataset(data_num=300000)
        model = tree.DecisionTreeClassifier(max_depth=12)
        model = model.fit(train_df.drop(columns='buy_int_flag'), train_df['buy_int_flag'])

        customers = [data_generator.generate_random_customer(f'customer_{no}')
                     for no in range(customer_num)]
        items = [data_generator.generate_random_item(f'item_{no}')
                 for no in range(item_num)]
        coupons = data_generator.generate_all_coupons()

        customer_names = [customer.name for customer in customers]
        item_names = [item.name for item in items]
        coupon_names = [f'coupon_{coupon.down_price}' for coupon in coupons]

        return MatchingProblem(
            customers=customers,
            items=items,
            coupons=coupons,
            customer_names=customer_names,
            item_names=item_names,
            coupon_names=coupon_names,
            buy_rate_model=model
        )
