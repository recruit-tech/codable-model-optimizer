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

class ScoreInfo:

    def __init__(self, objective_score: float, penalty_score: float):
        self._score = objective_score + penalty_score
        self._objective_score = objective_score
        self._penalty_score = penalty_score

    def set_scores(self, objective_score: float, penalty_score: float):
        self._score = objective_score + penalty_score
        self._objective_score = objective_score
        self._penalty_score = penalty_score

    @property
    def score(self) -> float:
        return self._score

    @property
    def objective_score(self) -> float:
        return self._objective_score

    @property
    def penalty_score(self) -> float:
        return self._penalty_score
