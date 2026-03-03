"""
Soon docs, here if needed.
"""

import logging
import operator
from typing import Callable, List, Optional, Tuple, Union

import pandas


LOGGER = logging.getLogger(__name__)
# NOTE: works best for vectorized operators, like the following 3
SUPPORTED_OPERATORS = {"*": operator.mul, "+": operator.add, "-": operator.sub}


Number = Union[int, float, pandas.Series]
Operator = Callable[[Number, Number], Number]
Operators = dict[str, Operator]
ParsedRole = Tuple[Operator, List[str]]


class ParseError(Exception):
    """Exception raised for errors in the `role` parsing process."""

    pass


def parse_role(
    role: str, valid_operators: Operators = SUPPORTED_OPERATORS
) -> ParsedRole:
    """
    Parses `role` against `valid_operators`.
    """
    allowed_symbols = valid_operators.keys()
    operator_symbol: Optional[str] = None
    column_names = []
    for symbol in allowed_symbols:
        column_names = [name.strip() for name in role.split(symbol)]
        if len(column_names) == 2:
            operator_symbol = symbol
            break
    if operator_symbol is None:
        joined = list(allowed_symbols)
        msg = f"{role=} is missing operator (like {joined})"
        raise ParseError(msg)
    operator_fn = valid_operators[operator_symbol]
    return operator_fn, column_names


def is_column_name_valid(column_name: str) -> bool:
    """
    Checks if `column_name` is valid.
    """
    return all(char.isalpha() or char == "_" for char in column_name)


def add_virtual_column(
    df: pandas.DataFrame, role: str, new_column: str
) -> pandas.DataFrame:
    """
    Copies `df` adding a `new_column` as a result of `role` operation.

    Example:
    >>> fruits_sales = pandas.DataFrame({
        'name': ['banana', 'apple'],
        'quantity': [10, 3],
        'price': [10, 1]
    })
    >>> add_virtual_column(fruits_sales, "quantity * price", "total")
    name quantity price price_total
    0 banana 10 10 100
    1 apple 3 1 3

    Validations:
    - Column labels must consist only of letters and underscores (_).
    - The function must support basic operations: addition (+), subtraction (-),
    and multiplication (*).
    - If the role or any column label is incorrect, the function should return an
    empty DataFrame.
    """
    # Validation
    try:
        operator_fn, role_columns = parse_role(role)
    except ParseError as error:
        LOGGER.error(error)
        return pandas.DataFrame()
    data_frame_columns = df.columns.tolist()
    all_column_names = [new_column, *data_frame_columns, *role_columns]
    for column_name in all_column_names:
        if not is_column_name_valid(column_name):
            msg = "Not all column names valid, found %s"
            LOGGER.error(msg, all_column_names)
            return pandas.DataFrame()
    if not set(role_columns).issubset(set(data_frame_columns)):
        msg = "Not all columns %s belongs to %s"
        LOGGER.error(msg, role_columns, data_frame_columns)
        return pandas.DataFrame()
    # Actual code
    values = operator_fn(df[role_columns[0]], df[role_columns[1]])
    new_df = df.assign(**{new_column: values})
    return new_df
