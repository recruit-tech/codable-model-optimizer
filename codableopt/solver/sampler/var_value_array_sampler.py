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
import numpy as np

from codableopt.solver.formulation.solver_problem import SolverProblem


class VarValueArraySampler:
    """ランダムに解答を生成するサンプラークラス。
    """

    def __init__(self):
        """ランダムに解答を生成するサンプラーのオブジェクト生成関数。
        """
        pass

    @staticmethod
    def sample(
            solver_problem: SolverProblem,
            generate_num: int,
            choice_num: int):
        """ランダムに解答を生成し、正規化した特徴量の値からユーグリッド距離を基準として、
        解答同士が距離が最大になる組み合わせの解答を選択する関数。

        Args:
            solver_problem (SolverProblem): ソルバー用に変換した最適化問題
            generate_num (int): 解答を生成する数
            choice_num (int): 生成した解答から選択する数
        Returns:
            サンプリングした解答のリスト
        """
        answers = VarValueArraySampler.generate(solver_problem, generate_num)
        return VarValueArraySampler.choice(answers, num=choice_num)

    @staticmethod
    def generate(solver_problem: SolverProblem, generate_num: int) -> np.ndarray:
        """ランダムに解答を生成する関数。

        Args:
            solver_problem (SolverProblem): ソルバー用に変換した最適化問題
            generate_num (int): 解答を生成する数

        Returns:
            生成したランダムな解答ロスト
        """

        return np.array([VarValueArraySampler._generate_random_var_value_array(solver_problem)
                         for _ in range(generate_num)])

    @staticmethod
    def _generate_random_var_value_array(solver_problem: SolverProblem) -> np.ndarray:
        """ランダムに解答を生成する関数。

        Args:
            solver_problem (SolverProblem): ソルバー用に変換した最適化問題

        Returns:
            生成したランダムな解答
        """
        values = []
        for variable in solver_problem.variables:
            values.extend(variable.random_values())

        return np.array(values)

    @staticmethod
    def choice(answers: np.ndarray, num: int) -> List[np.ndarray]:
        """引数の解答から、正規化した特徴量の値からユーグリッド距離を基準として、
        解答同士が距離が最大になる組み合わせの解答を選択する関数。

        Args:
            answers: 選択元の解答群
            num: 選択する解答数

        Returns:
            選択した解答
        """
        # 初期解の各変数の正規化を行う
        min_answers = np.min(answers, axis=0)
        max_answers = np.max(answers, axis=0)
        normalized_sample_answers = \
            np.nan_to_num((answers - min_answers) / (max_answers - min_answers),
                          nan=1, posinf=1, neginf=1)

        # 最初に生成した解を最初の初期解として加える
        init_answers = [answers[0, :]]
        normalized_init_answers = [normalized_sample_answers[0, :]]
        answers = np.delete(answers, 0, axis=0)
        normalized_sample_answers = np.delete(normalized_sample_answers, 0, axis=0)

        # 選択している初期解の中との最初距離が最大の初期解候補を加えていく
        for _ in range(num - 1):
            selected_index_no, max_distance = None, None
            for index_no, (sample_answer, normalized_sample_answer)\
                    in enumerate(zip(answers, normalized_sample_answers)):
                distance = VarValueArraySampler.__calculate_min_distance_from_answers(
                        normalized_sample_answer, normalized_init_answers)
                if max_distance is None or max_distance < distance:
                    max_distance = distance
                    selected_index_no = index_no

            init_answers.append(answers[selected_index_no, :])
            normalized_init_answers.append(normalized_sample_answers[selected_index_no, :])
            answers = np.delete(answers, selected_index_no, axis=0)
            normalized_sample_answers = \
                np.delete(normalized_sample_answers, selected_index_no, axis=0)

        return init_answers

    @staticmethod
    def __calculate_min_distance_from_answers(answer, base_answers):
        """選択済みの解答群の全ての解答に対して、選択候補の解答とのユーグリッド距離を計算し、
        その中の最短距離を計算する関数。

        Args:
            answer: 選択候補の解答
            base_answers: 選択済みの解答群

        Returns:
            最短距離となるユーグリッド距離
        """
        return min([np.linalg.norm(answer - base_answer) for base_answer in base_answers])
