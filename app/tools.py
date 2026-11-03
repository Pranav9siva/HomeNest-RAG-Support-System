"""Tiny tool layer to exercise tool-calling and failure recovery."""
import re

_ORDERS = {
    "HN1001": {"status": "Delivered", "delivered_on": "2026-10-02"},
    "HN1002": {"status": "Out for delivery", "eta": "2026-11-05"},
    "HN1003": {"status": "Processing", "eta": "2026-11-09"},
}
ORDER_RE = re.compile(r"\b(HN\d{4})\b", re.I)


class ToolError(Exception):
    pass


def order_status(order_id: str) -> dict:
    o = _ORDERS.get(order_id.upper())
    if not o:
        raise ToolError(f"Order {order_id.upper()} was not found")
    return {"order_id": order_id.upper(), **o}


def maybe_call_tool(question: str):
    """Return (tool_name, result_or_None, error_or_None) when the question needs a tool."""
    low = question.lower()
    if not any(w in low for w in ("order", "delivery", "track")):
        return None
    m = ORDER_RE.search(question)
    if not m:
        return ("order_status", None, "No order id provided. Please share an order id like HN1001.")
    try:
        return ("order_status", order_status(m.group(1)), None)
    except ToolError as e:
        return ("order_status", None, f"{e}. Please double-check the id, or contact support.")
