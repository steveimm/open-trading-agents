from __future__ import annotations

import json
import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

Rating = Literal["Buy", "Overweight", "Hold", "Underweight", "Sell"]


class ResearchPlan(BaseModel):
    recommendation: Rating
    rationale: str
    strategic_actions: str


class NumericDecision(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    @field_validator("*", mode="before")
    @classmethod
    def normalize_prices(cls, value: Any, info: Any) -> Any:
        """Normalize optional price placeholders while rejecting ambiguous levels.

        Args:
            value: Raw schema field value.
            info: Pydantic field information.

        Returns:
            A positive numeric price, None, or the untouched non-price field.
        """
        if info.field_name not in ("entry_price", "stop_loss", "price_target") or value is None:
            return value
        try:
            number = float(str(value).strip().lstrip("$€£¥").replace(",", ""))
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) and number > 0 else None


class TraderProposal(NumericDecision):
    action: Literal["Buy", "Hold", "Sell"]
    reasoning: str
    entry_price: float | None = None
    stop_loss: float | None = None
    position_sizing: str | None = None


class PortfolioDecision(NumericDecision):
    rating: Rating
    executive_summary: str
    investment_thesis: str
    price_target: float | None = None
    time_horizon: str | None = None


class ScreenedPost(BaseModel):
    index: int = Field(ge=0)
    relevance: float = Field(ge=0, le=1)
    stance: Literal["bullish", "bearish", "neutral", "unclear"]
    confidence: float = Field(ge=0, le=1)


class SocialScreen(BaseModel):
    items: list[ScreenedPost]


SCHEMAS: dict[str, type[BaseModel]] = {
    "research_manager": ResearchPlan,
    "trader": TraderProposal,
    "portfolio_manager": PortfolioDecision,
    "screening": SocialScreen,
}


def decision_format(role: str) -> str:
    """Describe the exact final-response schema validated for a structured role.

    Args:
        role: Key of the decision or screening role.

    Returns:
        Final-response instructions followed by the authoritative JSON schema.
    """
    return "Your final response must be exactly one JSON object matching the schema below, with no prose or Markdown fences outside it. Keep string fields as text, not nested objects or arrays. Put supporting evidence within the specified text fields. Use null only for fields whose schema allows it.\nJSON schema:\n" + json.dumps(SCHEMAS[role].model_json_schema(), ensure_ascii=False)


def decision_error(error: ValueError) -> str:
    """Summarize a validation failure without replaying the rejected response.

    Args:
        error: JSON syntax or schema validation failure.

    Returns:
        Concise field-specific feedback for a single native-session repair.
    """
    if isinstance(error, ValidationError):
        details = error.errors(include_input=False, include_url=False)
        return "; ".join(f"{'.'.join(str(value) for value in item['loc']) or 'response'}: {item['msg']}" for item in details[:8])[:1000]
    if isinstance(error, json.JSONDecodeError):
        return f"Invalid JSON: {error.msg} at line {error.lineno}, column {error.colno}."
    return str(error)[:1000]


def parse_decision(role: str, text: str) -> BaseModel:
    """Validate a complete JSON decision, accepting a surrounding Markdown fence.

    Args:
        role: Decision-making role key.
        text: Settled model output.

    Returns:
        A validated decision.

    Raises:
        ValueError: The text is not a valid decision for the role.
    """
    cleaned = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    return SCHEMAS[role].model_validate(json.loads(cleaned))


def render_decision(decision: BaseModel) -> str:
    """Render validated fields into a readable Markdown report.

    Args:
        decision: A research, trader, or portfolio result.

    Returns:
        Markdown with explicit unavailable optional values.
    """
    return "\n\n".join(f"**{key.replace('_', ' ').title()}**: {value if value is not None else 'not provided'}"
                       for key, value in decision.model_dump().items())
