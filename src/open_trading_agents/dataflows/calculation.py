from __future__ import annotations

import ast
import json
import math
import operator
import re
import statistics
from collections.abc import Callable
from typing import Any

Number = int | float
Value = Number | list[Number]
MAX_MAGNITUDE = 1e100
MAX_EXPRESSION_CHARS = 2000
MAX_NODES = 300
MAX_LIST_ITEMS = 1000
MAX_VARIABLE_ITEMS = 5000
FUNCTIONS = frozenset(
    {"abs", "sum", "min", "max", "round", "len", "mean", "median", "stdev", "pstdev", "sqrt", "log", "log10", "floor", "ceil"}
)
CONSTANTS = {"pi": math.pi, "e": math.e}
BINARY: dict[type[ast.operator], Callable[[Number, Number], Number]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}


def _number(value: Any) -> Number:
    """Validate a finite real number within the calculator's magnitude limit.

    Args:
        value: Input or intermediate numeric value.

    Returns:
        The validated number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Only real numbers and flat numeric lists are supported")
    if abs(value) > MAX_MAGNITUDE or not math.isfinite(value):
        raise ValueError("Numbers must be finite with absolute value at most 1e100")
    return value


def _sequence(value: Value) -> list[Number]:
    """Validate one flat numeric sequence used by aggregate functions.

    Args:
        value: Evaluated expression value.

    Returns:
        A bounded numeric list.
    """
    if not isinstance(value, list) or len(value) > MAX_LIST_ITEMS:
        raise ValueError("This function requires a numeric list of at most 1000 items")
    return [_number(item) for item in value]


def _call(name: str, arguments: list[Value]) -> Number:
    """Apply a small arithmetic/statistics function without dynamic name resolution.

    Args:
        name: Explicitly allowed function name.
        arguments: Already evaluated numeric inputs.

    Returns:
        A finite, bounded numeric result.
    """
    if name in {"sum", "len", "mean", "median", "stdev", "pstdev"}:
        if len(arguments) != 1:
            raise ValueError(f"{name} takes exactly one numeric list")
        values = _sequence(arguments[0])
        functions = {
            "sum": sum,
            "len": len,
            "mean": statistics.mean,
            "median": statistics.median,
            "stdev": statistics.stdev,
            "pstdev": statistics.pstdev,
        }
        return _number(functions[name](values))
    if name in {"min", "max"}:
        values = (
            _sequence(arguments[0]) if len(arguments) == 1 and isinstance(arguments[0], list) else [_number(value) for value in arguments]
        )
        return _number(min(values) if name == "min" else max(values))
    if name == "round":
        if not 1 <= len(arguments) <= 2:
            raise ValueError("round takes a value and an optional integer digit count")
        digits = arguments[1] if len(arguments) == 2 else 0
        if isinstance(digits, bool) or not isinstance(digits, int) or not -100 <= digits <= 100:
            raise ValueError("round digit count must be an integer between -100 and 100")
        return _number(round(_number(arguments[0]), digits))
    if name == "log":
        if not 1 <= len(arguments) <= 2:
            raise ValueError("log takes a positive value and an optional base")
        values = [_number(value) for value in arguments]
        return _number(math.log(*values))
    unary = {"abs": abs, "sqrt": math.sqrt, "log10": math.log10, "floor": math.floor, "ceil": math.ceil}
    if name not in unary or len(arguments) != 1:
        raise ValueError("Use an allowed function with the documented numeric arguments")
    return _number(unary[name](_number(arguments[0])))


def _evaluate(node: ast.AST, variables: dict[str, Value], depth: int = 0) -> Value:
    """Evaluate explicitly supported arithmetic nodes without Python execution.

    Args:
        node: Parsed expression node.
        variables: Validated numeric inputs.
        depth: Current expression nesting depth.

    Returns:
        A bounded numeric value or list.
    """
    if depth > 32:
        raise ValueError("Expression nesting exceeds 32 levels")
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, variables, depth + 1)
    if isinstance(node, ast.Constant):
        return _number(node.value)
    if isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        raise ValueError(f"Unknown numeric variable: {node.id}")
    if isinstance(node, (ast.List, ast.Tuple)):
        if len(node.elts) > MAX_LIST_ITEMS:
            raise ValueError("Numeric lists may contain at most 1000 items")
        return [_number(_evaluate(item, variables, depth + 1)) for item in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _number(_evaluate(node.operand, variables, depth + 1))
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _number(_evaluate(node.left, variables, depth + 1))
        right = _number(_evaluate(node.right, variables, depth + 1))
        if isinstance(node.op, ast.Pow):
            if abs(right) > 1000:
                raise ValueError("Exponent magnitude must not exceed 1000")
            if abs(left) > 1 and right > 0 and right * math.log10(abs(left)) > 100:
                raise ValueError("Power result exceeds the numeric magnitude limit")
            return _number(left**right)
        if function := BINARY.get(type(node.op)):
            return _number(function(left, right))
        raise ValueError("Unsupported arithmetic operator")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS:
        if node.keywords or len(node.args) > 32:
            raise ValueError("Functions accept at most 32 positional arguments and no keywords")
        return _call(node.func.id, [_evaluate(argument, variables, depth + 1) for argument in node.args])
    if isinstance(node, ast.Subscript):
        values = _sequence(_evaluate(node.value, variables, depth + 1))
        index = _evaluate(node.slice, variables, depth + 1)
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("List indices must be integers; slices are unsupported")
        return _number(values[index])
    raise ValueError("Only arithmetic, numeric lists, indexing, variables, and approved functions are allowed")


def calculate(expression: str, variables: dict[str, Any] | None = None) -> str:
    """Compute bounded arithmetic and statistics while retaining the exact supplied inputs.

    Args:
        expression: Arithmetic with +, -, *, /, //, **, %, numeric lists, or approved functions.
        variables: Up to 64 named finite numbers or numeric lists.

    Returns:
        JSON containing inputs and the computed result, or a clear calculation error.
    """
    try:
        if not isinstance(expression, str) or not expression.strip() or len(expression) > MAX_EXPRESSION_CHARS:
            raise ValueError("expression must contain between 1 and 2000 characters")
        if variables is not None and (not isinstance(variables, dict) or len(variables) > 64):
            raise ValueError("variables must be a mapping with at most 64 names")
        values: dict[str, Value] = {}
        item_count = 0
        for name, value in (variables or {}).items():
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", name) or name in FUNCTIONS | CONSTANTS.keys():
                raise ValueError("Variable names must be short identifiers and cannot replace functions or constants")
            values[name] = _sequence(value) if isinstance(value, list) else _number(value)
            item_count += len(value) if isinstance(value, list) else 1
            if item_count > MAX_VARIABLE_ITEMS:
                raise ValueError("variables may contain at most 5000 total numeric items")
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
            raise ValueError("Expression exceeds the 300-node calculation limit")
        result = _evaluate(tree, values)
        return json.dumps(
            {
                "expression": expression,
                "variables": values,
                "result": result,
                "numeric_note": "Uses finite real arithmetic and Python floating-point precision. Round explicitly where required.",
            },
            allow_nan=False,
        )
    except (ValueError, TypeError, SyntaxError, ArithmeticError, IndexError, RecursionError) as exc:
        shown = expression[:MAX_EXPRESSION_CHARS] if isinstance(expression, str) else ""
        return json.dumps({"expression": shown, "error": f"CALCULATION_ERROR: {exc}"}, allow_nan=False)
