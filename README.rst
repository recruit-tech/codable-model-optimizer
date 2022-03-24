.. image:: https://img.shields.io/pypi/v/codableopt.svg
    :target: https://pypi.python.org/pypi/codableopt
    
.. image:: https://readthedocs.org/projects/codable-model-optimizer/badge/?version=latest
    :target: https://codable-model-optimizer.readthedocs.io/ja/latest/?badge=latest
    :alt: Documentation Status


    
=========================
codable-model-optimizer
=========================
Optimization problem meta-heuristics solver for easy modeling.

.. index-start-installation-marker

Installation
================

Use pip
-------

.. code-block:: bash

    $ pip install codableopt
   
Use setup.py
------------

.. code-block:: bash

    # Master branch
    $ git clone https://github.com/recruit-tech/codable-model-optimizer
    $ python3 setup.py install



.. index-end-installation-marker

Example Usage
=================

Sample1
-------------------

.. index-start-sample1

.. code-block:: python

    import numpy as np
    from codableopt import *

    # set problem
    problem = Problem(is_max_problem=True)

    # define variables
    x = IntVariable(name='x', lower=np.double(0), upper=np.double(5))
    y = DoubleVariable(name='y', lower=np.double(0.0), upper=None)
    z = CategoryVariable(name='z', categories=['a', 'b', 'c'])


    # define objective function
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


    # set objective function and its arguments
    problem += Objective(objective=objective_function,
                         args_map={'var_x': x,
                                   'var_y': y,
                                   'var_z': z,
                                   'parameters': {'coef_x': -3.0, 'coef_y': 4.0}})

    # define constraint
    problem += 2 * x + 4 * y + 2 * (z == 'a') + 3 * (z == ('b', 'c')) <= 8
    problem += 2 * x - y + 2 * (z == 'b') > 3

    print(problem)

    solver = OptSolver()

    # generate optimization methods to be used within the solver
    method = PenaltyAdjustmentMethod(steps=40000)

    answer, is_feasible = solver.solve(problem, method)
    print(f'answer:{answer}, answer_is_feasible:{is_feasible}')

.. index-end-sample1

Sample2
-------------------


.. code-block:: python

    import random
    from itertools import combinations

    from codableopt import Problem, Objective, CategoryVariable, OptSolver, PenaltyAdjustmentMethod


    # define distance generating function
    def generate_distances(args_place_names):
        generated_distances = {}
        for point_to_point in combinations(['start'] + args_place_names, 2):
            distance_value = random.randint(20, 40)
            generated_distances[point_to_point] = distance_value
            generated_distances[tuple(reversed(point_to_point))] = distance_value
        for x in ['start'] + args_place_names:
            generated_distances[(x, x)] = 0

        return generated_distances


    # generate TSP problem
    PLACE_NUM = 30
    destination_names = [f'destination_{no}' for no in range(PLACE_NUM)]
    place_names = [f'P{no}' for no in range(PLACE_NUM)]
    distances = generate_distances(place_names)
    destinations = [CategoryVariable(name=destination_name, categories=place_names)
                    for destination_name in destination_names]

    # set problem
    problem = Problem(is_max_problem=False)


    # define objective function
    def calc_distance(var_destinations, para_distances):
        return sum([para_distances[(x, y)] for x, y in zip(
            ['start'] + var_destinations, var_destinations + ['start'])])


    # set objective function and its arguments
    problem += Objective(objective=calc_distance,
                         args_map={'var_destinations': destinations, 'para_distances': distances})

    # define constraint
    # constraint formula that always reaches all points at least once
    for place_name in place_names:
        problem += sum([(destination == place_name) for destination in destinations]) >= 1

    # optimization implementation
    solver = OptSolver(round_times=4, debug=True, debug_unit_step=1000)
    method = PenaltyAdjustmentMethod(steps=10000, delta_to_update_penalty_rate=0.9)
    answer, is_feasible = solver.solve(problem, method, n_jobs=-1)

    print(f'answer_is_feasible:{is_feasible}')
    root = ["start"] + [answer[root] for root in destination_names] + ["start"]
    print(f'root: {" -> ".join(root)}')
