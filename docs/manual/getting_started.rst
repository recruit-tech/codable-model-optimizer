=====================
Getting Started
=====================

.. include:: ../../README.rst
  :start-after: index-start-installation-marker
  :end-before: index-end-installation-marker

Basic Usage
============

1. **問題を設定**

問題オブジェクトを生成する際に、最大化または最小化問題のどちらかを指定をする必要があります。is_max_problemが、Trueの場合は最大化問題、Falseの場合は最小化問題となります。

>>> from codableopt import Problem
>>> problem = Problem(is_max_problem=True)


2. **変数を定義**

利用する変数を定義します。生成した変数オブジェクトは、制約式や目的関数の引数に利用することができます。

>>> from codableopt import IntVariable, DoubleVariable, CategoryVariable
>>> x = IntVariable(name='x', lower=np.double(0), upper=np.double(5))
>>> y = DoubleVariable(name='y', lower=np.double(0.0), upper=None)
>>> z = CategoryVariable(name='z', categories=['a', 'b', 'c'])

変数は、内包表記やfor文によってまとめて定義することもできます。

>>> from codableopt import IntVariable
>>> x = [IntVariable(name=f'x_{no}', lower=None, upper=None) for no in range(100)]


3. **目的関数を設定**

目的関数を問題に設定します。目的関数は、Objectiveオブジェクトを問題オブジェクトに加えることによって、設定できます。Objectiveオブジェクトを生成時には、「目的関数を計算するPython関数」と「引数のマッピング情報」を引数に設定します。
「引数のマッピング情報」は、Dict型で設定し、keyは目的関数の引数名、valueは変数オブジェクトまたは定数やPythonオブジェクトなどを指定します。なお、引数にマッピングした変数オブジェクトは、目的関数を計算するPython関数内では、最適化計算中の変数の値に変換されてから、引数に渡されます。

>>> def objective_function(var_x, var_y, var_z, parameters):
>>>     obj_value = parameters['coef_x'] * var_x + parameters['coef_y'] * var_y
>>>     if var_z == 'a':
>>>         obj_value += 10.0
>>>     elif var_z == 'b':
>>>         obj_value += 8.0
>>>     else:
>>>         # var_z == 'c'
>>>         obj_value -= 3.0
>>>
>>>     return obj_value
>>>
>>> problem += Objective(objective=objective_function,
>>>                      args_map={'var_x': x, 'var_y': y, 'var_z': z,
>>>                                'parameters': {'coef_x': -3.0, 'coef_y': 4.0}})

「引数のマッピング情報」には、変数リストを渡すこともできます。

>>> from codableopt import IntVariable
>>> x = [IntVariable(name=f'x_{no}', lower=None, upper=None) for no in range(100)]
>>>
>>> problem += Objective(objective=objective_function, args_map={'var_x': x}})


4. **制約式を定義**

制約式を問題に設定します。制約は、制約式オブジェクトを問題オブジェクトに加えることによって、設定できます。制約式オブジェクトは、変数オブジェクトと不等式を組み合わせることによって生成できます。不等式には、<,<=,>,>=,==が利用できます。また、1次式の制約式しか利用できません。

>>> constant = 2 * x + 4 * y + 2 * (z == 'a') + 3 * (z == ('b', 'c')) <= 8
>>> problem += constant

5. **最適化計算を実行**

ソルバーオブジェクトと最適化手法オブジェクトを生成し、ソルバーオブジェクトに問題オブジェクトと最適化手法オブジェクトを渡し、最適化計算を行います。ソルバーは、得られた最も良い解と得られた解が制約を全て満たすかの判定フラグを返します。

>>> solver = OptSolver(round_times=2)
>>> method = PenaltyAdjustmentMethod(steps=40000)
>>> answer, is_feasible = solver.solve(problem, method)
>>> print(f'answer:{answer}')
answer:{'x': 0, 'y': 1.5, 'z': 'a'}
>>> print(f'answer_is_feasible:{is_feasible}')
answer_is_feasible:True

Variable
============

整数・連続値・カテゴリの3種類の変数を提供しています。各変数は、目的関数に渡す引数や制約式に利用します。どの種類の変数も共通で、変数名を設定することができます。変数名は、最適化の解を返す際に利用されます。

IntVariable
--------------

IntVariableは、整数型の変数です。lowerには下界値、upperには上界値を設定します。なお、境界値は可能な値として設定されます。また、Noneを設定した場合は、下界値/上界値が設定されます。IntVariableは、目的関数に渡す引数と制約式のどちらにも利用できます。

.. code-block:: python

    from codableopt import IntVariable
    x = IntVariable(name='x', lower=0, upper=None)

DoubleVariable
------------------

DoubleVariableは、連続値型の変数です。lowerには下界値、upperには上界値を設定します。なお、境界値は可能な値として設定されます。また、Noneを設定した場合は、下界値/上界値が設定されます。DoubleVariableは、目的関数に渡す引数と制約式のどちらにも利用できます。

.. code-block:: python

    from codableopt import DoubleVariable
    x = DoubleVariable(name='x', lower=None, upper=2.3)

CategoryVariable
-------------------

CategoryVariableは、カテゴリ型の変数です。categoriesには、取り得るカテゴリ値を設定します。CategoryVariableは、目的関数に渡すことはできるが、制約式に利用することはできません。カテゴリ値を制約式に利用したい場合は、CategoryCaseVariableを利用する必要があります。

.. code-block:: python

    from codableopt import CategoryVariable
    x = CategoryVariable(name='x', categories=['a', 'b', 'c'])


CategoryCaseVariableは、カテゴリ型の変数と等式を組み合わせることで生成できます。Tupleを利用すれば、比較する値を複数設定でき、いずれかと等しい場合は1、それ以外の場合は0となります。CategoryCaseVariableは、目的関数に渡す引数と制約式のどちらにも利用できます。

.. code-block:: python

    # xが'a'の時は1、'b'または'c'の時は0となるCategoryCaseVariable
    x_a = x == 'a'
    # xがb'または'c'の時は1、'a'の時は0となるCategoryCaseVariable
    x_bc = x == ('b', 'c')
