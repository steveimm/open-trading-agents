import pytest

from open_trading_agents.analysis.decisions import parse_decision


def test_numeric_fields_do_not_turn_percentages_into_prices() -> None:
    """Keep usable decisions while discarding unsupported absolute levels."""
    result = parse_decision("trader", '{"action":"Buy","reasoning":"Evidence","entry_price":"$1,234.50","stop_loss":"15%"}')
    assert result.entry_price == 1234.5
    assert result.stop_loss is None


def test_rating_quoted_in_thesis_cannot_replace_actual_rating() -> None:
    """Parse the explicit structured rating rather than matching report prose."""
    result = parse_decision("portfolio_manager", '''```json
{"rating":"Underweight","executive_summary":"Reduce exposure","investment_thesis":"The bull proposed Buy."}
```''')
    assert result.rating == "Underweight"
    with pytest.raises(ValueError):
        parse_decision("portfolio_manager", "The bull says Buy, but the output was interrupted.")
