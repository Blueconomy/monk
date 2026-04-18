"""
Model pricing data (USD per 1M tokens, as of April 2026).
Update this file when providers change pricing.
"""

# (input_cost_per_1m, output_cost_per_1m)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o":                  (2.50,  10.00),
    "gpt-4o-2024-11-20":       (2.50,  10.00),
    "gpt-4o-mini":             (0.15,   0.60),
    "gpt-4o-mini-2024-07-18":  (0.15,   0.60),
    "gpt-4-turbo":             (10.00, 30.00),
    "gpt-4":                   (30.00, 60.00),
    "gpt-3.5-turbo":           (0.50,  1.50),
    "o1":                      (15.00, 60.00),
    "o1-mini":                 (3.00,  12.00),
    "o3-mini":                 (1.10,   4.40),
    # Anthropic
    "claude-opus-4-6":         (15.00, 75.00),
    "claude-opus-4":           (15.00, 75.00),
    "claude-sonnet-4-6":       (3.00,  15.00),
    "claude-sonnet-4":         (3.00,  15.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-haiku-20251001":  (0.80,  4.00),
    "claude-haiku-4-5-20251001":  (0.80,  4.00),
    "claude-3-opus-20240229":  (15.00, 75.00),
    # Google
    "gemini-1.5-pro":          (1.25,  5.00),
    "gemini-1.5-flash":        (0.075, 0.30),
    "gemini-2.0-flash":        (0.10,  0.40),
}

# Models considered "expensive" — trigger model_overkill checks
EXPENSIVE_MODELS = {
    "gpt-4o", "gpt-4o-2024-11-20", "gpt-4-turbo", "gpt-4",
    "o1", "claude-opus-4-6", "claude-opus-4", "claude-3-opus-20240229",
    "claude-sonnet-4-6", "claude-sonnet-4", "claude-3-5-sonnet-20241022",
    "gemini-1.5-pro",
}

# Suggested cheaper alternatives
CHEAPER_ALTERNATIVE: dict[str, str] = {
    "gpt-4o":                     "gpt-4o-mini",
    "gpt-4o-2024-11-20":          "gpt-4o-mini",
    "gpt-4-turbo":                "gpt-4o-mini",
    "gpt-4":                      "gpt-4o-mini",
    "o1":                         "o3-mini",
    "claude-opus-4-6":            "claude-haiku-4-5-20251001",
    "claude-opus-4":              "claude-haiku-4-5-20251001",
    "claude-3-opus-20240229":     "claude-3-5-haiku-20251001",
    "claude-sonnet-4-6":          "claude-haiku-4-5-20251001",
    "claude-sonnet-4":            "claude-haiku-4-5-20251001",
    "claude-3-5-sonnet-20241022": "claude-3-5-haiku-20251001",
    "gemini-1.5-pro":             "gemini-1.5-flash",
}


def cost_for_call(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return USD cost for a single LLM call."""
    model_lower = model.lower()
    # Try exact match first, then prefix match
    pricing = MODEL_PRICING.get(model_lower)
    if pricing is None:
        for key, val in MODEL_PRICING.items():
            if model_lower.startswith(key) or key.startswith(model_lower):
                pricing = val
                break
    if pricing is None:
        return 0.0
    input_cost, output_cost = pricing
    return (input_tokens / 1_000_000) * input_cost + (output_tokens / 1_000_000) * output_cost


def cost_difference(model: str, alt_model: str, input_tokens: int, output_tokens: int) -> float:
    """Return potential savings (USD) by switching model to alt_model."""
    return cost_for_call(model, input_tokens, output_tokens) - \
           cost_for_call(alt_model, input_tokens, output_tokens)
