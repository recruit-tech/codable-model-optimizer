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
import multiprocessing
import colorsys
from pathlib import Path

from pulp import *
from geopy.distance import geodesic
import folium

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
                         coordinates: Dict[str, Tuple[float, float]],
                         demands: Dict[str, int],
                         capacity: int):
        # 距離を計算
        # 分解前に全ての地点間の距離を1度計算すれば、子問題分解時に計算しなくて良いが、
        # 一方で地点数が非常に多い場合は、組み合わせ数が多く、実際に使われない地点間の計算も行われ、
        # 非効率になる場合もあるためこのサンプルコードでは分解後に計算している。
        distances = {}
        depot_and_place_names = [depot_name] + place_names
        for point_to_point in combinations(depot_and_place_names, 2):
            coordinate_a = coordinates[point_to_point[0]]
            coordinate_b = coordinates[point_to_point[1]]
            distance_value = geodesic(coordinate_a, coordinate_b).km
            distances[point_to_point] = distance_value
            distances[tuple(reversed(point_to_point))] = distance_value
        for args_place_name in depot_and_place_names:
            distances[(args_place_name, args_place_name)] = 0

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
    MIN_LONGITUDE = 35.347980
    MAX_LONGITUDE = 35.596265
    MIN_LATITUDE = 139.349233
    MAX_LATITUDE = 139.581319

    @staticmethod
    def generate(depot_num: int, place_num: int):
        depot_names = [f'D{no}' for no in range(depot_num)]
        place_names = [f'P{no}' for no in range(place_num)]

        coordinates = {name: MultiDepotCVRProblem.generate_random_coordinate()
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

    @staticmethod
    def generate_random_coordinate() -> Tuple[float, float]:
        longitude = random.randrange(int(MultiDepotCVRProblem.MIN_LONGITUDE * 1000000),
                                     int(MultiDepotCVRProblem.MAX_LONGITUDE * 1000000), 1)
        latitude = random.randrange(int(MultiDepotCVRProblem.MIN_LATITUDE * 1000000),
                                    int(MultiDepotCVRProblem.MAX_LATITUDE * 1000000), 1)
        return float(longitude) / 1000000, float(latitude) / 1000000

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
    def solve(problem: CVRProblem) -> Tuple[float, List[List[str]]]:
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

        lp_problem.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=60*3,
                                           threads=multiprocessing.cpu_count()))
        roots = []
        for start_place_no in range(place_num):
            if value(x[0, start_place_no]) == 1:
                root = [0, start_place_no]
                current_place_no = start_place_no
                while current_place_no != 0:
                    for next_place_no in range(place_num):
                        if value(x[current_place_no, next_place_no]) == 1:
                            current_place_no = next_place_no
                            root.append(next_place_no)
                            break
                roots.append([problem.place_names[i] for i in root])

        return lp_problem.objective.value(), roots


class MultiDepotCVRPSolver:

    def __init__(self):
        pass

    @staticmethod
    def solve(problem: MultiDepotCVRProblem, steps: int):
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
        method = PenaltyAdjustmentMethod(steps=steps)
        dict_answer, is_feasible = solver.solve(opt_problem, method, init_answers=[init_answer])

        # 実行可能解を見つけられなかった場合
        if not is_feasible:
            return None, None

        return MultiDepotCVRPSolver.calc_obj_and_roots(
            [dict_answer[x] for x in problem.place_names], problem)

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
                    part_obj, _ = CVRPSolver.solve(child_problem)
                    # 差分計算用にキャッシュ
                    cashed_part_obj[cash_key] = part_obj
            else:
                part_obj = 0

            obj += part_obj

        return obj

    @staticmethod
    def calc_obj_and_roots(var_x, para_problem: MultiDepotCVRProblem):
        obj = 0
        roots = []
        for depot_no, depot_name in enumerate(para_problem.depot_names):
            place_noes = [place_no for place_no, val in enumerate(var_x) if val == depot_name]
            if len(place_noes) > 0:
                child_problem = para_problem.to_child_problem(depot_no=depot_no,
                                                              place_noes=place_noes)
                part_obj, part_roots = CVRPSolver.solve(child_problem)
                obj += part_obj
                roots.extend(part_roots)

        return obj, roots

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


class MapGenerator:

    def __init__(self):
        pass

    @staticmethod
    def save_map_html(problem: MultiDepotCVRProblem, roots: List[List[str]], html_path: Path):
        # 解答可視化
        fmap = folium.Map(
            [(MultiDepotCVRProblem.MIN_LONGITUDE + MultiDepotCVRProblem.MAX_LONGITUDE) / 2,
             (MultiDepotCVRProblem.MIN_LATITUDE + MultiDepotCVRProblem.MAX_LATITUDE) / 2],
            zoom_start=11)

        for depot_name in problem.depot_names:
            folium.CircleMarker(
                location=problem.coordinates[depot_name],
                radius=math.sqrt(problem.depot_capacities[depot_name]),
                popup=f'{depot_name}:{problem.depot_capacities[depot_name]}',
                color='#B1221A',
                fill=True,
                fill_color='#B1221A',
            ).add_to(fmap)

        for place_name in problem.place_names:
            folium.CircleMarker(
                location=problem.coordinates[place_name],
                radius=math.sqrt(problem.demands[place_name]),
                popup=f'{place_name}:{problem.demands[place_name]}',
                color='#2f17d0',
                fill=True,
                fill_color='#2f17d0',
            ).add_to(fmap)

        for root, color_code in zip(roots, MapGenerator.generate_color_codes(len(roots))):
            for start_name, end_name in zip(root[:-1], root[1:]):
                point_to_point = (problem.coordinates[start_name], problem.coordinates[end_name])
                fmap.add_child(folium.PolyLine(point_to_point, color=color_code))

        fmap.save(html_path)

    @staticmethod
    def generate_color_codes(color_num: int):
        color_codes = []
        for color_no in range(color_num):
            rgb = colorsys.hsv_to_rgb(1.0 / color_num * color_no, 0.7, 0.7)
            rgb = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
            color_codes.append('#%02x%02x%02x' % rgb)
        return color_codes


mdcvr_problem = MultiDepotCVRProblem.generate(depot_num=8, place_num=40)
answer_objective, answer_roots = MultiDepotCVRPSolver.solve(mdcvr_problem, steps=1000)

if answer_objective is None:
    print('No Answer!')
else:
    print(f'Total Distance {answer_objective} km')
    MapGenerator.save_map_html(mdcvr_problem, answer_roots, Path('mdcvrp_answer.html'))
