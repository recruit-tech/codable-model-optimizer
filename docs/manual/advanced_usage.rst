=====================
Advanced Usage
=====================

Delta Objective Function
===============================================

目的関数の計算は、関数によっては非常に計算コストが高くなります。しかし、差分計算を用いることで目的関数の計算コストを下げることができます。本ソルバーでも、目的関数の差分計算関数を設定することができます。差分計算は、Objectiveオブジェクト生成時の引数にdelta_objectiveを設定することで利用できます。なお、差分計算関数の引数には、目的関数と同様の引数に加えて、遷移前の変数の値が元の変数名の前にpre_をつけた名前で渡されます。

.. code-block:: python

    x = IntVariable(name='x', lower=np.double(0.0), upper=None)
    y = DoubleVariable(name='y', lower=np.double(0.0), upper=None)

    # 目的関数
    def obj_fun(var_x, var_y):
        return 3 * var_x + 5 * var_y

    # 目的関数の差分計算用の関数
    def delta_obj_fun(pre_var_x, pre_var_y, var_x, var_y, parameters):
        delta_value = 0
        if pre_var_x != var_x:
            delta_value += parameters['coef_x'] * (var_x - pre_var_x)
        if pre_var_y != var_y:
            delta_value += parameters['coef_y'] * (var_y - pre_var_y)
        return delta_value

    # 目的関数を定義
    problem += Objective(objective=obj_fun,
                         delta_objective=delta_obj_fun,
                         args_map={'var_x': x, 'var_y': y,
                                   'parameters': {'coef_x': 3.0, 'coef_y': 2.0}})

Custom Optimization Method
==============================

本ソルバーは、共通アルゴリズム上で最適化手法をカスタマイズして利用することはできます。最適化手法をカスタマイズする場合は、本ソルバーが提供しているOptimizerMethodを継承して実装することで実現することができます。本ソルバーが提供しているペナルティ係数調整手法もその枠組み上で実装されています。

OptimizerMethod
-----------------

.. autoclass:: codableopt.solver.optimizer.method.optimizer_method.OptimizerMethod
    :members:

Sample Code
-----------------

.. code-block:: python

    from typing import List
    from random import choice

    from codableopt.solver.optimizer.entity.proposal_to_move import ProposalToMove
    from codableopt.solver.optimizer.method.optimizer_method import OptimizerMethod
    from codableopt.solver.optimizer.optimization_state import OptimizationState


    class SampleMethod(OptimizerMethod):

        def __init__(self, steps: int):
            super().__init__(steps)

        def name(self) -> str:
            return 'sample_method'

        def initialize_of_step(self, state: OptimizationState, step: int):
            # ステップ開始時の処理なし
            pass

        def propose(self, state: OptimizationState, step: int) -> List[ProposalToMove]:
            # 変数から1つランダムに選択する
            solver_variable = choice(state.problem.variables)
            # 選択した変数をランダムに移動する解の遷移を提案する
            return solver_variable.propose_random_move(state)

        def judge(self, state: OptimizationState, step: int) -> bool:
            # 遷移前と遷移後のスコアを比較
            delta_energy = state.current_score.score - state.previous_score.score
            # ソルバー内はエネルギーが低い方が最適性が高いことを表している
            # マイナスの場合に解が改善しているため、提案を受け入れる
            return delta_energy < 0

        def finalize_of_step(self, state: OptimizationState, step: int):
            # ステップ終了時の処理なし
            pass


[deprecation] User Define Constraint
===============================================

非推奨ではありますが、本ソルバーでは、制約式を関数として渡すこともできます。関数の返り値には制約違反量を設定します。引数は、目的関数同様にargs_mapを設定することで指定できます。ただし、デフォルトで提供しているmethodでは、User Define Constraintを利用している最適化問題は実用に耐えうる最適化精度を実現できません。そのため現状では利用することは推奨していません。

Sample Code
-----------------

.. code-block:: python

    # 制約式を定義
    def udf_constraint_function(var_x, var_y, var_z):
        violation_amount = 2 * var_x + 4 * var_y - 8
        if var_z == 'a':
            violation_amount += 2
        else:
            violation_amount += 3

        if violation_amount <= 0:
            return 0
        else:
            return violation_amount


    constant = UserDefineConstraint(udf_constraint_function,
                                    args_map={'var_x': x, 'var_y': y, 'var_z': z},
                                    constraint_name='user_define_constraint')
    problem += constant
