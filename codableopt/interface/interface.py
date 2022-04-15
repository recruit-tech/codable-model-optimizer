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

from typing import Sequence, Dict, List, Union, Optional, Tuple, Any
from collections.abc import Callable
import re
import copy
from enum import Enum
from functools import reduce

import numpy as np

from codableopt.interface.system_problem import SystemProblem
from codableopt.interface.system_variable \
    import SystemVariable, SystemIntegerVariable, SystemDoubleVariable, SystemCategoryVariable
from codableopt.interface.system_constraint \
    import SystemLinerConstraint, SystemUserDefineConstraint, SystemConstraints
from codableopt.interface.system_objective import SystemUserDefineObjective
from codableopt.interface.system_args_map import SystemArgsMap


class Sign(Enum):
    LOWER = '<'
    LOWER_EQUAL = '<='
    GREATER = '>'
    GREATER_EQUAL = '>='
    EQUAL = '=='


class Term:

    def __init__(self):
        pass


class Variable(Term):

    illegal_chars = '-+[] ->/'
    expression = re.compile(f'[{re.escape(illegal_chars)}]')

    def __init__(self, name: str):
        super(Variable, self).__init__()
        self._coefficient: np.double = np.double(1.0)
        self.__name = str(name).translate(Variable.illegal_chars)

    def to_system_variable(self) -> SystemVariable:
        raise NotImplementedError('not implemented to_system_variable')

    @property
    def name(self) -> str:
        return self.__name

    @property
    def coefficient(self) -> np.double:
        return self._coefficient

    @coefficient.setter
    def coefficient(self, coefficient: np.double):
        self._coefficient = coefficient


class Formula:

    def __init__(self, number_variables: Sequence):
        self._number_variable_dict: Dict[str, NumberVariable] = {}
        self._constant: Constant = Constant(value=np.double(0.0))
        for num_var in number_variables:
            self.__add_term(num_var)

    def __add__(self, other):
        new_f = copy.deepcopy(self)

        if isinstance(other, (NumberVariable, Constant)):
            new_f.__add_term(copy.deepcopy(other))
        elif isinstance(other, (int, float, np.double)):
            new_f.__add_term(Constant(other))
        elif isinstance(other, Formula):
            for num_var in other._number_variable_dict.values():
                new_f.__add_term(num_var)
            new_f.__add_term(other._constant)
        else:
            raise ValueError(f'Not support {type(other)}')

        return new_f

    def __sub__(self, other):
        new_f = copy.deepcopy(self)

        if isinstance(other, (NumberVariable, Constant)):
            tmp_other = copy.deepcopy(other)
            tmp_other.coefficient = -tmp_other.coefficient
            new_f.__add_term(tmp_other)

        elif isinstance(other, (int, float, np.double)):
            tmp_constant = Constant(-other)
            new_f.__add_term(tmp_constant)

        elif isinstance(other, Formula):
            for num_var in other._number_variable_dict.values():
                tmp_num_var = copy.deepcopy(num_var)
                tmp_num_var.coefficient = -tmp_num_var.coefficient
                new_f.__add_term(tmp_num_var)

            tmp_constant = copy.deepcopy(other._constant)
            tmp_constant.value = -tmp_constant.value
            new_f.__add_term(tmp_constant)
        else:
            raise ValueError(f'Not support {type(other)}')

        return new_f

    def __mul__(self, other):
        if isinstance(other, (int, float, np.double)):
            new_f = copy.deepcopy(self)

            for num_var in new_f._number_variable_dict.values():
                num_var.coefficient *= other
            new_f._constant *= other

            return new_f
        else:
            raise ValueError(f'Not support {type(other)}')

    def __truediv__(self, other):

        if isinstance(other, (int, float, np.double)):
            new_f = copy.deepcopy(self)

            for num_var in new_f._number_variable_dict.values():
                num_var.coefficient /= other
            new_f._constant /= other

            return new_f
        else:
            raise ValueError(f'Not support {type(other)}')

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __lt__(self, other):
        return self.__to_constraint(other, sign=Sign.LOWER)

    def __le__(self, other):
        return self.__to_constraint(other, sign=Sign.LOWER_EQUAL)

    def __gt__(self, other):
        return self.__to_constraint(other, sign=Sign.GREATER)

    def __ge__(self, other):
        return self.__to_constraint(other, sign=Sign.GREATER_EQUAL)

    def __eq__(self, other):
        return self.__to_constraint(other, sign=Sign.EQUAL)

    def __str__(self):
        num_values = self._number_variable_dict.values()
        return ' '.join([x.to_formulated_str(no == 0) for no, x in enumerate(num_values)])\
               + ('' if len(num_values) == 0 else ' ')\
               + self._constant.to_formulated_str(len(num_values) == 0)

    def __add_term(self, term: Term):
        # case Constant
        if isinstance(term, Constant):
            self._constant.value += term.value
        # case Variable
        elif isinstance(term, (IntVariable, DoubleVariable, CategoryCaseVariable)):
            if term.name in self._number_variable_dict.keys():
                term_b = self._number_variable_dict[term.name]

                if sum([isinstance(term, x) and isinstance(term_b, x)
                        for x in [IntVariable, DoubleVariable, CategoryCaseVariable]]) == 0:
                    raise ValueError(f'variables({term.name}) of different types are exist!')

                term_b.coefficient += term.coefficient
                # check delete
                if term_b.coefficient == 0:
                    del term_b
            else:
                self._number_variable_dict[term.name] = term
        else:
            raise ValueError(f'Not support {type(term)}')

    def __to_constraint(self, other, sign: Sign):
        left_formula = copy.deepcopy(self)

        if isinstance(other, (NumberVariable, Constant)):
            right_formula = Formula(number_variables=[copy.deepcopy(other)])
        elif isinstance(other, (int, float, np.double)):
            right_formula = Formula(number_variables=[Constant(other)])
        elif isinstance(other, Formula):
            right_formula = copy.deepcopy(other)
        else:
            raise ValueError(f'Not support {type(other)}')

        return LinerConstraint(
            left_formula=left_formula,
            right_formula=right_formula,
            sign=sign)

    @property
    def number_variables(self):
        return list(self._number_variable_dict.values())

    @property
    def constant(self):
        return self._constant


class LinerConstraint:

    def __init__(
            self,
            left_formula: Formula,
            right_formula: Formula,
            sign: Sign):
        self._left_formula = left_formula
        self._right_formula = right_formula
        self._sign = sign

    def __add__(self, other):
        self._right_formula += other
        return self

    def __sub__(self, other):
        self._right_formula -= other
        return self

    def __mul__(self, other):
        self._right_formula *= other
        return self

    def __truediv__(self, other):
        self._right_formula /= other
        return self

    def __str__(self):
        return f'{self._left_formula} {self._sign.value} {self._right_formula}'

    def to_system_liner_constraints(self) -> List[SystemLinerConstraint]:
        if self._sign in [Sign.LOWER, Sign.LOWER_EQUAL]:
            sum_formulas = [self._right_formula - self._left_formula]
        elif self._sign in [Sign.GREATER, Sign.GREATER_EQUAL]:
            sum_formulas = [self._left_formula - self._right_formula]
        elif self._sign == Sign.EQUAL:
            sum_formulas = [self._left_formula - self._right_formula,
                            self._right_formula - self._left_formula]
        else:
            raise ValueError(f'Not support sign:{self._sign}')

        constraints = []
        for sum_formula in sum_formulas:
            var_coefficients = {}
            for var in sum_formula.number_variables:
                for var_name, coefficient in var.to_names_and_coefficients():
                    var_coefficients[var_name] = coefficient

            include_equal_to_zero = \
                self._sign in [Sign.LOWER_EQUAL, Sign.GREATER_EQUAL, Sign.EQUAL]
            constant = SystemLinerConstraint(
                var_coefficients=var_coefficients,
                constant=sum_formula.constant.value,
                include_equal_to_zero=include_equal_to_zero)
            constraints.append(constant)

        return constraints

    @property
    def number_variables(self):
        return self._left_formula.number_variables + self._right_formula.number_variables


class UserDefineConstraint:

    def __init__(
            self,
            constraint_function: Callable,
            args_map: Dict[str, Any],
            constraint_name: Optional[str] = None):
        """ユーザ定義の制約式オブジェクトの生成。（非推奨）

        Args:
            constraint_function: 制約式の違反量を計算するPython関数
            args_map: 目的関数の引数のマッピング情報（key:引数名、value:引数に渡す値）
            constraint_name: 制約式の名前
        """
        if not callable(constraint_function):
            raise ValueError(f'constraint_function must be callable!')

        self._constraint_function = copy.deepcopy(constraint_function)
        self._args_map = args_map
        self._constraint_name = constraint_name

    def __str__(self):
        return f'user_define_constraint ' \
               f'{"" if self._constraint_name is None else self._constraint_name}'

    def to_system_user_define_constraint(
            self,
            system_args_map: SystemArgsMap) -> SystemUserDefineConstraint:
        return SystemUserDefineConstraint(
            constraint_function=self._constraint_function,
            system_args_map=system_args_map
        )

    @property
    def constraint_function(self):
        return self._constraint_function

    @property
    def args_map(self):
        return self._args_map

    @property
    def variables(self):
        return self._args_map.values()

    @property
    def number_variables(self):
        return [x for x in self._args_map.values() if isinstance(x, NumberVariable)]


class Constant(Term):

    def __init__(self, value: Union[int, float, np.double]):
        super(Constant, self).__init__()
        self._value = np.double(value)

    def __neg__(self):
        constant = copy.deepcopy(self)
        constant.value *= -1
        return constant

    def __add__(self, other) -> Formula:
        if isinstance(other, (NumberVariable, Constant)):
            return Formula([copy.deepcopy(self), copy.deepcopy(other)])
        elif isinstance(other, (int, float, np.double)):
            return Formula([copy.deepcopy(self), Constant(other)])
        elif isinstance(other, Formula):
            new_f = copy.deepcopy(other)
            new_f += self
            return new_f
        else:
            raise ValueError(f'Not support {type(other)}')

    def __sub__(self, other) -> Formula:
        if isinstance(other, (NumberVariable, Constant)):
            new_other = copy.deepcopy(other)
            new_other *= -1
            return Formula([copy.deepcopy(self), new_other])
        elif isinstance(other, (int, float, np.double)):
            new_other = copy.deepcopy(Constant(other))
            new_other *= -1
            return Formula([copy.deepcopy(self), new_other])
        elif isinstance(other, Formula):
            new_f = copy.deepcopy(other)
            new_f -= self
            return new_f
        else:
            raise ValueError(f'Not support {type(other)}')

    def __mul__(self, other):
        if isinstance(other, (int, float, np.double)):
            new_c = copy.deepcopy(self)
            new_c._value *= other
            return new_c
        else:
            raise ValueError(f'Not support {type(other)}')

    def __truediv__(self, other):
        if isinstance(other, (int, float, np.double)):
            new_c = copy.deepcopy(self)
            new_c._value /= other
            return new_c
        else:
            raise ValueError(f'Not support {type(other)}')

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __lt__(self, other):
        return self.__to_constraint(other, sign=Sign.LOWER)

    def __le__(self, other):
        return self.__to_constraint(other, sign=Sign.LOWER_EQUAL)

    def __gt__(self, other):
        return self.__to_constraint(other, sign=Sign.GREATER)

    def __ge__(self, other):
        return self.__to_constraint(other, sign=Sign.GREATER_EQUAL)

    def __eq__(self, other):
        return self.__to_constraint(other, sign=Sign.EQUAL)

    def to_formulated_str(self, is_first_term: bool = False):
        if self._value == 0:
            return ''
        elif self._value > 0:
            return f'{"" if is_first_term else "+ "}{self._value}'
        else:
            return f'- {abs(self._value)}'

    def __to_constraint(self, other, sign: Sign):
        left_formula = Formula([copy.deepcopy(self)])

        if isinstance(other, (NumberVariable, Constant)):
            right_formula = Formula([copy.deepcopy(other)])
        elif isinstance(other, Formula):
            right_formula = copy.deepcopy(other)
        else:
            raise ValueError(f'Not support {type(other)}')

        return LinerConstraint(
            left_formula=left_formula,
            right_formula=right_formula,
            sign=sign)

    @property
    def value(self) -> np.double:
        return self._value

    @value.setter
    def value(self, value):
        self._value = value


class NumberVariable(Variable):

    def __init__(self, name: str):
        super(NumberVariable, self).__init__(name)

    def __neg__(self):
        num_var = copy.deepcopy(self)
        num_var.coefficient *= -1
        return num_var

    def __add__(self, other) -> Formula:
        if isinstance(other, (NumberVariable, Constant)):
            return Formula([copy.deepcopy(self), copy.deepcopy(other)])
        elif isinstance(other, (int, float, np.double)):
            return Formula([copy.deepcopy(self), Constant(other)])
        elif isinstance(other, Formula):
            new_f = copy.deepcopy(other)
            new_f += self
            return new_f
        else:
            raise ValueError(f'Not support {type(other)}')

    def __sub__(self, other) -> Formula:
        if isinstance(other, (NumberVariable, Constant)):
            return Formula([copy.deepcopy(self), -copy.deepcopy(other)])
        elif isinstance(other, (int, float, np.double)):
            return Formula([copy.deepcopy(self), -Constant(other)])
        elif isinstance(other, Formula):
            new_f = copy.deepcopy(other)
            new_f -= self
            return new_f
        else:
            raise ValueError(f'Not support {type(other)}')

    def __mul__(self, other):
        if isinstance(other, (int, float, np.double)):
            new_num = copy.deepcopy(self)
            new_num._coefficient *= other
            return new_num
        else:
            raise ValueError(f'Not support {type(other)}')

    def __truediv__(self, other):
        if isinstance(other, (int, float, np.double)):
            new_num = copy.deepcopy(self)
            new_num._coefficient /= other
            return new_num
        else:
            raise ValueError(f'Not support {type(other)}')

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __lt__(self, other):
        return self.__to_liner_constraint(other, sign=Sign.LOWER)

    def __le__(self, other):
        return self.__to_liner_constraint(other, sign=Sign.LOWER_EQUAL)

    def __gt__(self, other):
        return self.__to_liner_constraint(other, sign=Sign.GREATER)

    def __ge__(self, other):
        return self.__to_liner_constraint(other, sign=Sign.GREATER_EQUAL)

    def __eq__(self, other):
        return self.__to_liner_constraint(other, sign=Sign.EQUAL)

    def to_formulated_str(self, is_first_term: bool = False):
        coefficient = self.coefficient

        if coefficient == 0:
            return '0'
        elif coefficient > 0:
            return f'{"" if is_first_term else "+ "}' \
                   f'{"" if coefficient == 1 else coefficient}' \
                   f'{self._to_formulated_variable_name_str()}'
        else:
            return f'- {"" if abs(coefficient) == 1 else abs(coefficient)}' \
                   f'{self._to_formulated_variable_name_str()}'

    def equals(self, number_variable) -> bool:
        raise NotImplementedError('not implemented equals')

    def to_system_variable(self) -> SystemVariable:
        raise NotImplementedError('not implemented to_system_variable')

    def to_names_and_coefficients(self) -> List[Tuple[str, np.double]]:
        raise NotImplementedError('not implemented to_tuple_of_system_variable')

    def _to_formulated_variable_name_str(self):
        raise NotImplementedError(
            '_to_formulated_variable_name_str is not implemented!')

    def __to_liner_constraint(self, other, sign: Sign):
        left_formula = Formula(number_variables=[copy.deepcopy(self)])

        if isinstance(other, (NumberVariable, Constant)):
            right_formula = Formula(number_variables=[copy.deepcopy(other)])
        elif isinstance(other, (int, float, np.double)):
            right_formula = Formula(number_variables=[Constant(other)])
        elif isinstance(other, Formula):
            right_formula = copy.deepcopy(other)
        else:
            raise ValueError(f'Not support {type(other)}')

        return LinerConstraint(
            left_formula=left_formula,
            right_formula=right_formula,
            sign=sign)


class IntVariable(NumberVariable):

    def __init__(
            self,
            name: str,
            lower: Optional[int] = None,
            upper: Optional[int] = None):
        """整数変数を生成。

        Args:
            name: 変数名
            lower: 下界値
            upper: 上界値
        """
        super(IntVariable, self).__init__(name)
        self._lower = lower
        self._upper = upper

    def _to_formulated_variable_name_str(self):
        return self.name

    def equals(self, number_variable) -> bool:
        if isinstance(number_variable, IntVariable):
            return self.name == number_variable.name \
                   and self._lower == number_variable._lower \
                   and self._upper == number_variable._upper
        else:
            return False

    def to_system_variable(self) -> SystemIntegerVariable:
        return SystemIntegerVariable(
            name=self.name,
            lower=self._lower,
            upper=self._upper)

    def to_names_and_coefficients(self) -> List[Tuple[str, np.double]]:
        return [(self.name, self.coefficient)]


class DoubleVariable(NumberVariable):

    def __init__(self,
                 name: str,
                 lower: Optional[Union[np.double, int, float]] = None,
                 upper: Optional[Union[np.double, int, float]] = None):
        """連続値変数を生成。

        Args:
            name: 変数名
            lower: 下界値
            upper: 上界値
        """
        super(DoubleVariable, self).__init__(name)
        self._lower = lower
        self._upper = upper

    def _to_formulated_variable_name_str(self):
        return self.name

    def equals(self, number_variable) -> bool:
        if isinstance(number_variable, DoubleVariable):
            return self.name == number_variable.name \
                   and self._lower == number_variable._lower \
                   and self._upper == number_variable._upper
        else:
            return False

    def to_system_variable(self) -> SystemDoubleVariable:
        return SystemDoubleVariable(
            name=self.name,
            lower=self._lower,
            upper=self._upper
        )

    def to_names_and_coefficients(self) -> List[Tuple[str, np.double]]:
        return [(self.name, self.coefficient)]


class CategoryVariable(Variable):

    def __init__(
            self,
            name: str,
            categories: List[Union[str, int, float, np.double]]):
        """カテゴリ変数の生成。

        Args:
            name: 変数名
            categories: カテゴリ値のマスタ
        """
        super(CategoryVariable, self).__init__(name)
        self._categories = sorted(categories)

    def __add__(self, other) -> Formula:
        raise NotImplementedError('__add__ not implemented!')

    def __sub__(self, other) -> Formula:
        raise NotImplementedError('__sub__ not implemented!')

    def __mul__(self, other):
        raise NotImplementedError('__mul__ not implemented!')

    def __truediv__(self, other):
        raise NotImplementedError('__truediv__ not implemented!')

    def __eq__(self,
               other: Union[Sequence[Union[str, int, float, np.double]],
                            Union[str, int, float, np.double]]):
        if isinstance(other, (str, int, float)):
            return CategoryCaseVariable(self, [other], equal_flg=True)
        elif isinstance(other, Sequence):
            return CategoryCaseVariable(self, other, equal_flg=True)
        else:
            raise ValueError('__eq__ support only str or int or float')

    def __ne__(self, other):
        if isinstance(other, Sequence):
            return CategoryCaseVariable(self, other, equal_flg=False)
        elif isinstance(other, (str, int, float)):
            return CategoryCaseVariable(self, [other], equal_flg=False)
        else:
            raise ValueError('__eq__ support only str or int or float')

    def __lt__(self, other):
        raise NotImplementedError('__lt__ not implemented!')

    def __le__(self, other):
        raise NotImplementedError('__le__ not implemented!')

    def __gt__(self, other):
        raise NotImplementedError('__gt__ not implemented!')

    def __ge__(self, other):
        raise NotImplementedError('__ge__ not implemented!')

    @property
    def categories(self) -> List[str]:
        return self._categories

    def equals(self, number_variable) -> bool:
        if isinstance(number_variable, CategoryVariable):
            if self.name != number_variable.name or \
                    len(self._categories) != len(number_variable._categories):
                return False
            for a, b in zip(self._categories, number_variable._categories):
                if a != b:
                    return False
            return True
        else:
            return False

    def to_system_variable(self) -> SystemCategoryVariable:
        return SystemCategoryVariable(
            name=self.name,
            categories=self._categories)


class CategoryCaseVariable(NumberVariable):

    def __init__(
            self,
            category_variable: CategoryVariable,
            categories_in_case: Sequence[Union[str, int, float, np.double]],
            equal_flg: bool):
        sorted_cases = sorted(categories_in_case)
        name = f'{category_variable.name}_is_' \
               + '_or_'.join(['"' + str(x) + '"' for x in sorted_cases])
        super(CategoryCaseVariable, self).__init__(name)

        not_exist_categories = [x for x in sorted_cases
                                if (x not in category_variable.categories)]
        if len(not_exist_categories) > 0:
            raise ValueError(
                f'{",".join(not_exist_categories)}'
                f'{"is" if len(not_exist_categories) == 1 else "are"}'
                f' not include in category_variable!')
        if len(sorted_cases) == 0:
            raise ValueError('categories_in_case must be longer than 1!')

        self._category_variable = category_variable
        self._categories_in_case = sorted_cases
        self._equal_flg = equal_flg

    def _to_formulated_variable_name_str(self):
        return f'({self._category_variable.name} ' \
               f'{("==" if self._equal_flg else "!=")} ' \
               + f' {("or" if self._equal_flg else "and")} '\
                   .join([str(x) for x in self._categories_in_case]) + ')'

    @property
    def category_variable(self):
        return self._category_variable

    def equals(self, number_variable) -> bool:
        if not isinstance(number_variable, CategoryCaseVariable):
            return False

        if self.name != number_variable.name:
            return False

        if self._category_variable.equals(number_variable._category_variable):
            return True
        else:
            return False

    def to_system_variable(self) -> SystemVariable:
        raise NotImplementedError('not implemented to_system_variable')

    def to_system_variable_names(self) -> List[str]:
        return [f'{self._category_variable.name}:{category}'
                for category in self._categories_in_case]

    def to_names_and_coefficients(self) -> List[Tuple[str, np.double]]:
        return [(f'{self._category_variable.name}:{x}', self.coefficient)
                for x in self._categories_in_case]


class Objective:

    def __init__(
            self,
            objective: Callable,
            args_map: Dict[str, Any],
            delta_objective: Optional = None):
        """ユーザ定義の目的関数オブジェクトの生成。

        Args:
            objective: 目的関数値を計算するPython関数
            args_map: 目的関数の引数のマッピング情報（key:引数名、value:引数に渡す値）
            delta_objective: 目的関数値を差分計算するPython関数
        """
        if callable(objective):
            self._objective = copy.deepcopy(objective)
            self._args_map = args_map
            self._delta_objective = delta_objective
            self._objective_is_function = True
        elif isinstance(objective, (NumberVariable, Constant)):
            raise ValueError('Not support objective type is formulate')
        elif isinstance(objective, Formula):
            raise ValueError('Not support objective type is formulate')
        else:
            raise ValueError(f'Not support {type(objective)}')

    def __str__(self):
        if self._objective_is_function:
            return 'user_define_function'
        else:
            return str(self._objective)

    def to_system_objective(
            self,
            is_max_problem: bool,
            system_args_map: SystemArgsMap) -> SystemUserDefineObjective:
        if self._objective_is_function:
            return SystemUserDefineObjective(
                is_max_problem=is_max_problem,
                objective=self._objective,
                system_args_map=system_args_map,
                delta_objective=self._delta_objective)
        else:
            raise Exception('Not support objective type is formulate')

    @property
    def objective(self) -> Callable:
        return self._objective

    @property
    def args_map(self):
        return self._args_map

    @property
    def delta_objective(self):
        return self._delta_objective

    @property
    def objective_is_function(self):
        return self._objective_is_function

    @property
    def variables(self):
        if isinstance(self._objective, Formula):
            return self._objective.number_variables
        else:
            # TODO n次元対応
            variables = []
            for x in self._args_map.values():
                if isinstance(x, Variable):
                    variables.append(x)
                elif isinstance(x, list) and isinstance(x[0], Variable):
                    variables.extend(x)
                elif isinstance(x, list) and isinstance(x[0], list) and \
                        isinstance(x[0][0], Variable):
                    variables.extend(x[0])
                elif isinstance(x, list) and isinstance(x[0], list) and \
                        isinstance(x[0][0], list) and isinstance(x[0][0][0], Variable):
                    variables.extend(x[0][0])
            return variables

    @property
    def number_variables(self):
        if isinstance(self._objective, Formula):
            return self._objective.number_variables
        else:
            return [x for x in self._args_map.values() if isinstance(x, NumberVariable)]


class Problem:

    def __init__(self, is_max_problem: bool = True):
        """最適化問題を生成

        Args:
            is_max_problem: 最大化問題フラグ（True:最大化問題、False:最小化問題）
        """
        self._is_max_problem = is_max_problem
        self._objective: Optional[Objective] = None
        self._user_define_constraints: List[UserDefineConstraint] = []
        self._liner_constraints: List[LinerConstraint] = []

    def __add__(self, other):
        if isinstance(other, Objective):
            self._objective = other
            return self
        elif isinstance(other, UserDefineConstraint):
            self._user_define_constraints.append(other)
            return self
        elif isinstance(other, LinerConstraint):
            self._liner_constraints.append(other)
            return self
        elif isinstance(other, (NumberVariable, Constant)):
            # support only user define function at now
            # self._objective = Objective(Formula(number_variables=[copy.deepcopy(other)]))
            # return self
            raise ValueError(f'Not support {type(other)}')
        elif isinstance(other, Formula):
            # support only user define function at now
            # self._objective = Objective(other)
            # return self
            raise ValueError(f'Not support {type(other)}')
        else:
            raise ValueError(f'Not support {type(other)}')

    def __str__(self):
        return f'Objective:\n' \
               f'  UserDefineFunction\n' \
               f'subject to.\n' \
               + '\n'.join([f'  {x}' for x in self._user_define_constraints]) \
               + '\n'.join([f'  {x}' for x in self._liner_constraints])

    def compile(self):
        variables = self.__convert_to_system_variables(self._objective,
                                                       self._user_define_constraints,
                                                       self._liner_constraints)

        system_variables = [x.to_system_variable() for x in variables]
        if len(self._liner_constraints) == 0:
            system_liner_constraints = []
        else:
            system_liner_constraints = reduce(
                lambda a, b: a + b,
                [cons.to_system_liner_constraints() for cons in self._liner_constraints])

        system_user_define_constraints = [
            x.to_system_user_define_constraint(Problem.__convert_to_system_args_map(x.args_map))
            for x in self._user_define_constraints]
        system_constraints = SystemConstraints(
            liner_constraints=system_liner_constraints,
            user_define_constraints=system_user_define_constraints
        )
        system_objective = self._objective.to_system_objective(
            is_max_problem=self._is_max_problem,
            system_args_map=Problem.__convert_to_system_args_map(self._objective.args_map)
        )

        return SystemProblem(
            variables=system_variables,
            objective=system_objective,
            constraints=system_constraints)

    @staticmethod
    def __convert_to_system_variables(objective, user_define_constraints, liner_constraints):
        # 目的関数内の数値変数取得
        variables = objective.variables

        # 制約式内の数値変数取得
        for constraint in user_define_constraints:
            variables.extend(constraint.variables)
        for constraint in liner_constraints:
            # liner_constraintのvariablesは、number_variablesしかない
            variables.extend(constraint.number_variables)

        # 問題に利用する数値変数を集約
        variables_dict = {}
        for var in variables:
            if var.name in variables_dict.keys():
                if not variables_dict[var.name].equals(var):
                    raise Exception(f'Wrong variable(name:{var.name}) exist!'
                                    f' (same name, but different parameter)')
            else:
                variables_dict[var.name] = var

        var_dict = {}
        for var in variables_dict.values():
            if isinstance(var, (IntVariable, DoubleVariable)):
                var_dict[var.name] = var
            elif isinstance(var, CategoryCaseVariable):
                cat_var = var.category_variable
                if cat_var.name in var_dict.keys():
                    if not var_dict[cat_var.name].equals(cat_var):
                        raise Exception(f'Wrong variable(name:{cat_var.name}) exist!'
                                        f' (same name, but different parameter)')
                else:
                    var_dict[cat_var.name] = cat_var
            elif isinstance(var, CategoryVariable):
                var_dict[var.name] = var
            else:
                raise Exception(f'Not support number_variable_type:{var}')

        return var_dict.values()

    @staticmethod
    def __convert_to_system_args_map(args_map):
        variable_args_map = {}
        category_args_map = {}
        parameter_args_map = {}

        # TODO n次元対応
        for arg_name in args_map.keys():
            arg_value = args_map[arg_name]
            if isinstance(arg_value, NumberVariable):
                variable_args_map[arg_name] = [x for x, _ in arg_value.to_names_and_coefficients()]

            elif isinstance(arg_value, list) and isinstance(arg_value[0], NumberVariable):
                var_names = [[x[0] for x in y.to_names_and_coefficients()] for y in arg_value]
                variable_args_map[arg_name] = var_names

            elif isinstance(arg_value, list) and isinstance(arg_value[0], list) and \
                    isinstance(arg_value[0][0], NumberVariable):
                var_names = \
                    [[[x[0] for x in y.to_names_and_coefficients()] for y in z] for z in arg_value]
                variable_args_map[arg_name] = var_names

            elif isinstance(arg_value, list) and isinstance(arg_value[0], list) and \
                    isinstance(arg_value[0][0], list) and \
                    isinstance(arg_value[0][0][0], NumberVariable):
                var_names = [
                    [[[x[0] for x in y.to_names_and_coefficients()] for y in z] for z in a]
                    for a in arg_value]
                variable_args_map[arg_name] = var_names

            elif isinstance(arg_value, CategoryVariable):
                category_args_map[arg_name] = [arg_value.name]

            elif isinstance(arg_value, list) and isinstance(arg_value[0], CategoryVariable):
                var_names = [[y.name] for y in arg_value]
                category_args_map[arg_name] = var_names

            elif isinstance(arg_value, list) and isinstance(arg_value[0], list) and \
                    isinstance(arg_value[0][0], CategoryVariable):
                var_names = [[[y.name] for y in z] for z in arg_value]
                category_args_map[arg_name] = var_names

            elif isinstance(arg_value, list) and \
                    isinstance(arg_value[0], list) and \
                    isinstance(arg_value[0][0], list) and \
                    isinstance(arg_value[0][0][0], CategoryVariable):
                var_names = [[[[y.name] for y in z] for z in a] for a in arg_value]
                category_args_map[arg_name] = var_names

            else:
                parameter_args_map[arg_name] = args_map[arg_name]

        return SystemArgsMap(
            variable_args_map=variable_args_map,
            category_args_map=category_args_map,
            parameter_args_map=parameter_args_map
        )
