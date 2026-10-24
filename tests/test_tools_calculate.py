from __future__ import annotations

import json
from typing import Any

import pytest

from open_trading_agents.dataflows.calculation import calculate
from open_trading_agents.tools import ToolContext, create_server


@pytest.mark.parametrize(
    ("expression", "variables", "expected"),
    [
        ("round((current / previous - 1) * 100, 2)", {"current": 110, "previous": 100}, 10.0),
        ("floor(equity * risk_fraction / (entry - stop))", {"equity": 100000, "risk_fraction": 0.01, "entry": 150, "stop": 143}, 142),
        ("round(mean(returns), 4)", {"returns": [0.02, -0.01, 0.05]}, 0.02),
        ("median([1, 3, 5, 7])", None, 4.0),
        ("stdev([1, 2, 3])", None, 1.0),
        ("round(pstdev([1, 2, 3]), 4)", None, 0.8165),
        ("sqrt(81) + log(100, 10) + log10(1000)", None, 14.0),
        ("abs(-3) + max(2, 5) - min([2, 4])", None, 6),
        ("sum(prices) / len(prices)", {"prices": [1, 2, 3, 4]}, 2.5),
        ("prices[-1] - prices[0]", {"prices": [100, 105, 110]}, 10),
        ("2**10 % 100", None, 24),
    ],
)
def test_numeric_verification_returns_only_the_result(expression: str, variables: dict[str, Any] | None, expected: float) -> None:
    """Compute real finance and statistics examples instead of relying on model arithmetic."""
    result = json.loads(calculate(expression, variables))
    assert result["result"] == pytest.approx(expected)
    assert set(result) == {"result"}


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo unsafe')",
        "open('/etc/passwd').read()",
        "(1).__class__",
        "[x for x in [1,2]]",
        "(lambda: 1)()",
        "sum.__globals__",
        "2**1000000000",
        "10**1000",
        "1e309",
        "float('nan')",
        "[1]*1000000000",
        "True + 1",
        "'text' * 10000",
        "1 << 10000000",
        "sqrt(-1)",
        "1/0",
        "round(1, 1000000000)",
    ],
)
def test_calculator_rejects_execution_nonfinite_values_and_unbounded_work(expression: str) -> None:
    """Reject code execution and resource amplification before any such operation can occur."""
    result = json.loads(calculate(expression))
    assert "CALCULATION_ERROR" in result["error"] and "result" not in result


@pytest.mark.parametrize(
    "variables",
    [
        {"value": float("nan")},
        {"value": float("inf")},
        {"value": True},
        {"value": [[1]]},
        {"value": [1] * 1001},
        {"sum": 1},
        {"__secret": 1},
    ],
)
def test_calculator_validates_variable_values_and_names(variables: dict[str, Any]) -> None:
    """Reject non-numeric input and reserved names even when the expression is harmless."""
    assert "CALCULATION_ERROR" in json.loads(calculate("1 + 1", variables))["error"]


def test_expression_size_and_nesting_are_bounded() -> None:
    """Bound parse/evaluation work independently of numeric magnitude limits."""
    assert "CALCULATION_ERROR" in json.loads(calculate("1+" * 1001 + "1"))["error"]
    assert "CALCULATION_ERROR" in json.loads(calculate("1+" * 150 + "1"))["error"]
    assert "CALCULATION_ERROR" in json.loads(calculate("-" * 40 + "1"))["error"]


@pytest.mark.asyncio
async def test_calculator_is_available_as_a_selected_mcp_tool() -> None:
    """Allow deterministic verification for roles that have no financial-data tools."""
    server = create_server(ToolContext("NVDA", "2024-03-15"), ("calculate",))
    listed = await server.list_tools()
    assert [tool.name for tool in listed] == ["calculate"]
    result = await server.call_tool("calculate", {"expression": "round(1500/7, 2)"})
    assert not result.is_error
    payload = json.loads(result.content[0].text)
    assert payload["result"] == 214.29
