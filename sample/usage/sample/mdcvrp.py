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
import math
from typing import List, Tuple, Dict
from dataclasses import dataclass
from itertools import combinations

from pulp import *

from codableopt import Problem, Objective, CategoryVariable, OptSolver, PenaltyAdjustmentMethod


@dataclass
class CVRProblem:
    depot_name: str
    place_names: List[str]
    distances: Dict[Tuple[str, str], int]
    demands: Dict[str, int]
    capacity: int

    def calc_objective(self, answer: Dict[Tuple[str, str], int]):
        obj = 0
        for key in answer.keys():
            obj += self.distances[key] * answer[key]
        return obj

    def is_feasible(self, answer: Dict[Tuple[str, str], int]):
        # ルート整合性チェック
        for place_name in self.place_names:
            car_num_in = sum([answer[x, place_name] for x in self.place_names])
            car_num_out = sum([answer[place_name, x] for x in self.place_names])
            if car_num_in != car_num_out:
                return False

        # 容量制約
        correct_root_sum = 0
        for place_name in self.place_names:
            volume = 0
            if answer[self.depot_name, place_name] == 1:
                correct_root_sum += 1
                while place_name != self.depot_name:
                    volume += self.demands[place_name]
                    next_place_names = [x for x in self.place_names if answer[place_name, x] == 1]
                    if len(next_place_names) != 1:
                        return False
                    place_name = next_place_names[0]
                    correct_root_sum += 1
                if volume > self.capacity:
                    return False

        # サブルートチェック
        if sum(answer.values()) != correct_root_sum:
            return False

        return True

    @staticmethod
    def generate_problem(depot_name: str,
                         place_names: List[str],
                         coordinates: Dict[str, Tuple[int, int]],
                         demands: Dict[str, int],
                         capacity: int):
        # 距離生成関数
        def generate_distances(args_place_names, args_coordinates):
            # ポイント間の距離を生成
            generated_distances = {}
            for point_to_point in combinations(args_place_names, 2):
                coordinate_a = args_coordinates[point_to_point[0]]
                coordinate_b = args_coordinates[point_to_point[1]]
                distance_value = math.sqrt(math.pow(coordinate_a[0] - coordinate_b[0], 2) +
                                           math.pow(coordinate_a[1] - coordinate_b[1], 2))
                generated_distances[point_to_point] = distance_value
                generated_distances[tuple(reversed(point_to_point))] = distance_value
            for args_place_name in args_place_names:
                generated_distances[(args_place_name, args_place_name)] = 0

            return generated_distances

        # 距離を生成
        distances = generate_distances([depot_name] + place_names, coordinates)

        return CVRProblem(
            depot_name=depot_name,
            place_names=[depot_name] + place_names,
            distances=distances,
            demands=demands,
            capacity=capacity
        )


@dataclass
class MultiDepotCVRProblem:
    depot_names: List[str]
    place_names: List[str]
    coordinates: Dict[str, Tuple[int, int]]
    demands: Dict[str, int]
    capacity: int
    depot_capacities: Dict[str, int]

    @staticmethod
    def generate(depot_num: int, place_num: int):
        depot_names = [f'D{no}' for no in range(depot_num)]
        place_names = [f'P{no}' for no in range(place_num)]
        # 座標を生成
        coordinates = {name: (random.randint(1, 1000), random.randint(1, 1000))
                       for name in (depot_names + place_names)}
        # 需要量
        demands = {name: random.randint(10, 50) for name in (depot_names + place_names)}
        # 容量制限
        capacity = 100

        demands_sum = sum(demands.values())
        # depot容量
        depot_capacities = {x: int((0.5 + random.random()) * demands_sum / depot_num)
                            for x in depot_names}
        while sum(depot_capacities.values()) <= demands_sum * 1.15:
            for x in depot_capacities.keys():
                depot_capacities[x] = int(depot_capacities[x] * 1.1)

        return MultiDepotCVRProblem(
            depot_names=depot_names,
            place_names=place_names,
            coordinates=coordinates,
            demands=demands,
            capacity=capacity,
            depot_capacities=depot_capacities
        )

    def to_child_problem(self, depot_no: int, place_noes: List[int]) -> CVRProblem:
        return CVRProblem.generate_problem(
            depot_name=self.depot_names[depot_no],
            place_names=[self.place_names[i] for i in place_noes],
            coordinates=self.coordinates,
            demands=self.demands,
            capacity=self.capacity
        )


class CVRPSolver:

    def __init__(self):
        pass

    @staticmethod
    def solve(problem: CVRProblem):
        # TODO マルチスレッド
        place_num = len(problem.place_names)

        distances = {}
        for place_no_1, place_1 in enumerate(problem.place_names):
            for place_no_2, place_2 in enumerate(problem.place_names):
                distances[(place_no_1, place_no_2)] = problem.distances[(place_1, place_2)]

        lp_problem = LpProblem('TSP', LpMinimize)
        x = pulp.LpVariable.dicts('x',
                                  ((i, j) for i in range(place_num) for j in range(place_num)),
                                  cat='Binary')
        y = pulp.LpVariable.dicts('y',
                                  (i for i in range(place_num)),
                                  lowBound=0, upBound=problem.capacity, cat='Integer')

        lp_problem += pulp.lpSum(distances[(i, j)] * x[i, j]
                                 for i in range(place_num)
                                 for j in range(place_num))

        for i in range(place_num):
            lp_problem += x[i, i] == 0

        for i in range(place_num):
            if i == 0:
                lp_problem += pulp.lpSum(x[i, j] for j in range(place_num)) >= 1
                lp_problem += pulp.lpSum(x[j, i] for j in range(place_num)) >= 1
            else:
                lp_problem += pulp.lpSum(x[i, j] for j in range(place_num)) == 1
                lp_problem += pulp.lpSum(x[j, i] for j in range(place_num)) == 1

        # 容量制限
        lp_problem += y[0] == 0
        for i in range(place_num):
            for j, place_name in enumerate(problem.place_names):
                if i != j and (i != 0 and j != 0):
                    lp_problem += y[j] - y[i] >= -problem.capacity * (1 - x[i, j]) \
                                  + problem.demands[place_name]

        lp_problem.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=60*3))
        return lp_problem.objective.value()


class MultiDepotCVRPSolver:

    def __init__(self):
        pass

    @staticmethod
    def solve(problem: MultiDepotCVRProblem):
        opt_problem = Problem(is_max_problem=False)
        x = [CategoryVariable(name=x, categories=[y for y in problem.depot_names])
             for x in problem.place_names]
        opt_problem += \
            Objective(objective=MultiDepotCVRPSolver.calc_obj,
                      args_map={'var_x': x,
                                'para_problem': problem,
                                'cashed_part_obj': {}})

        # Depotの容量制限の制約式
        for depot_name in problem.depot_names:
            opt_problem += \
                sum([(bit_x == depot_name) * problem.demands[bit_x.name] for bit_x in x]) <= \
                problem.depot_capacities[depot_name]

        # 初期解指定
        init_answer = MultiDepotCVRPSolver.generate_init_answer(problem)
        solver = OptSolver(debug=True, debug_unit_step=100, num_to_tune_penalty=10)
        method = PenaltyAdjustmentMethod(steps=5000)
        dict_answer, is_feasible = solver.solve(opt_problem, method, init_answers=[init_answer])

        return dict_answer, is_feasible

    @staticmethod
    def calc_obj(var_x, para_problem: MultiDepotCVRProblem, cashed_part_obj: Dict[str, float]):
        obj = 0
        for depot_no, depot_name in enumerate(para_problem.depot_names):
            place_noes = [place_no for place_no, val in enumerate(var_x) if val == depot_name]
            if len(place_noes) > 0:
                child_problem = para_problem.to_child_problem(depot_no=depot_no,
                                                              place_noes=place_noes)
                cash_key = depot_name + '_' + '_'.join([str(x) for x in place_noes])
                if cash_key in cashed_part_obj.keys():
                    # キャッシュから子問題の目的関数値を取得
                    part_obj = cashed_part_obj[cash_key]
                else:
                    # 子問題の目的関数値を計算
                    part_obj = CVRPSolver.solve(child_problem)
                    # 差分計算用にキャッシュ
                    cashed_part_obj[cash_key] = part_obj
            else:
                part_obj = 0

            obj += part_obj

        return obj

    @staticmethod
    def generate_init_answer(problem: MultiDepotCVRProblem):
        init_answer = {}
        for place_name in problem.place_names:
            place_coordinate = problem.coordinates[place_name]
            min_distance, nearest_depot_name = None, None
            for depot_name in problem.depot_names:
                depot_coordinate = problem.coordinates[depot_name]
                distance = math.sqrt(math.pow(place_coordinate[0] - depot_coordinate[0], 2) +
                                     math.pow(place_coordinate[1] - depot_coordinate[1], 2))
                if min_distance is None or min_distance > distance:
                    min_distance = distance
                    nearest_depot_name = depot_name
            init_answer[place_name] = nearest_depot_name
        return init_answer


mdcvr_problem = MultiDepotCVRProblem.generate(depot_num=5, place_num=30)
MultiDepotCVRPSolver.solve(mdcvr_problem)
