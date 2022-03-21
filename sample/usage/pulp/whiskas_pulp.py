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

from pulp import *

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

prob = LpProblem('The Whiskas Problem', LpMinimize)
ingredient_vars = LpVariable.dicts('Ingr', Ingredients, 0)

prob += lpSum([costs[i] * ingredient_vars[i] for i in Ingredients])

prob += lpSum([ingredient_vars[i] for i in Ingredients]) == 100
prob += lpSum([proteinPercent[i] * ingredient_vars[i] for i in Ingredients]) >= 8.0
prob += lpSum([fatPercent[i] * ingredient_vars[i] for i in Ingredients]) >= 6.0
prob += lpSum([fibrePercent[i] * ingredient_vars[i] for i in Ingredients]) <= 2.0
prob += lpSum([saltPercent[i] * ingredient_vars[i] for i in Ingredients]) <= 0.4

prob.solve()

for v in prob.variables():
    print(f'{v.name}={v.varValue}')
print(f'obj={value(prob.objective)}')
