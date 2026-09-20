"""Token prices for cost estimates. USD per 1M tokens as (input, output).

Verify against https://platform.openai.com/pricing before quoting a dollar figure; token counts
in llm_calls are measured, the dollar amounts are only as current as this table.
"""

PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.0),
    # gpt-4.1-mini and gpt-5.4-mini are deliberately absent: their rates have not been checked
    # against the pricing page, and a wrong number here would quietly corrupt the cost tables.
    # cost_usd returns None for them, and callers report the tokens with the price marked unknown.
}


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    if model not in PRICES:
        return None
    price_in, price_out = PRICES[model]
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000
