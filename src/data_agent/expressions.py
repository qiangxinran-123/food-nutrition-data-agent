from __future__ import annotations

import ast
import operator
from typing import Any, Callable

import numpy as np
import pandas as pd


class ExpressionError(ValueError):
    pass


def _as_series(value: Any, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value
    return pd.Series(value, index=index)


def _col(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        raise ExpressionError(f"unknown column in expression: {name!r}")
    return frame[name]


def _elementwise_min(*values: Any) -> Any:
    result = values[0]
    for value in values[1:]:
        result = np.minimum(result, value)
    return result


def _elementwise_max(*values: Any) -> Any:
    result = values[0]
    for value in values[1:]:
        result = np.maximum(result, value)
    return result


_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
}

_COMPARISON_OPERATORS: dict[type[ast.cmpop], Callable[[Any, Any], Any]] = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


def _evaluate(node: ast.AST, frame: pd.DataFrame) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (str, int, float, bool)) or node.value is None:
            return node.value
        raise ExpressionError(f"unsupported constant: {node.value!r}")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExpressionError("only named functions are allowed")
        if node.keywords:
            raise ExpressionError("keyword arguments are not allowed")
        function = node.func.id
        arguments = [_evaluate(argument, frame) for argument in node.args]
        if function == "col":
            if len(arguments) != 1 or not isinstance(arguments[0], str):
                raise ExpressionError("col() expects one string column name")
            return _col(frame, arguments[0])
        if function == "abs":
            if len(arguments) != 1:
                raise ExpressionError("abs() expects one argument")
            return abs(arguments[0])
        if function == "isfinite":
            if len(arguments) != 1:
                raise ExpressionError("isfinite() expects one argument")
            return np.isfinite(arguments[0])
        if function == "min":
            if not arguments:
                raise ExpressionError("min() expects at least one argument")
            return _elementwise_min(*arguments)
        if function == "max":
            if not arguments:
                raise ExpressionError("max() expects at least one argument")
            return _elementwise_max(*arguments)
        raise ExpressionError(f"unsupported function: {function!r}")

    if isinstance(node, ast.BinOp):
        operation = _BINARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise ExpressionError(f"unsupported binary operator: {type(node.op).__name__}")
        return operation(_evaluate(node.left, frame), _evaluate(node.right, frame))

    if isinstance(node, ast.BoolOp):
        values = [_evaluate(value, frame) for value in node.values]
        result = values[0]
        if isinstance(node.op, ast.And):
            for value in values[1:]:
                result = result & value
        elif isinstance(node.op, ast.Or):
            for value in values[1:]:
                result = result | value
        else:
            raise ExpressionError("unsupported boolean operator")
        return result

    if isinstance(node, ast.UnaryOp):
        value = _evaluate(node.operand, frame)
        if isinstance(node.op, ast.Not):
            return ~value
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return +value
        raise ExpressionError(f"unsupported unary operator: {type(node.op).__name__}")

    if isinstance(node, ast.Compare):
        left = _evaluate(node.left, frame)
        result: Any = None
        for comparison, comparator in zip(node.ops, node.comparators):
            operation = _COMPARISON_OPERATORS.get(type(comparison))
            if operation is None:
                raise ExpressionError(
                    f"unsupported comparison operator: {type(comparison).__name__}"
                )
            right = _evaluate(comparator, frame)
            current = operation(left, right)
            result = current if result is None else result & current
            left = right
        return result

    raise ExpressionError(f"unsupported expression element: {type(node).__name__}")


def evaluate_expression(expression: str, frame: pd.DataFrame) -> pd.Series:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ExpressionError(f"invalid expression syntax: {error.msg}") from error
    result = _evaluate(tree.body, frame)
    return _as_series(result, frame.index)


def evaluate_boolean_expression(expression: str, frame: pd.DataFrame) -> pd.Series:
    result = evaluate_expression(expression, frame)
    return result.fillna(False).astype(bool)


__all__ = [
    "ExpressionError",
    "evaluate_boolean_expression",
    "evaluate_expression",
]
