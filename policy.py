"""
Deterministic policy engine.

This is the part of the system that MUST be correct regardless of what the
language model says. The LLM proposes an action + reasoning in structured
form; every proposal is re-checked here against the hard-coded rules before
it is ever shown to the customer. If the LLM's proposal disagrees with the
policy engine, the policy engine wins and the disagreement is logged.
"""

from dataclasses import dataclass, field
from typing import Optional
from . import data


LEGAL_KEYWORDS = ["legal action", "lawyer", "sue", "formal complaint", "consumer court", "file a complaint"]


@dataclass
class Decision:
    action: str
    details: str
    escalate: bool = False
    escalation_reason: Optional[str] = None


def get_flight(pnr: str, flight_code: Optional[str] = None):
    bookings = data.BOOKINGS.get(pnr, [])
    if flight_code:
        for b in bookings:
            if b["flight"].lower() == flight_code.lower():
                return b
    return bookings[0] if bookings else None


def decide_for_cancellation(pnr: str, flight: dict) -> Decision:
    """Section 3, Rule 1: cancelled -> free rebook OR full refund, customer's choice."""
    return Decision(
        action="offer_rebook_or_refund",
        details=(
            f"Flight {flight['flight']} ({flight['route']}) is cancelled (operational reasons). "
            f"Customer is entitled to: (a) free rebooking on next available flight within 24h, "
            f"or (b) full refund to original payment method within 7 business days. Customer chooses."
        ),
    )


def decide_delay_compensation(pnr: str, flight: dict) -> Decision:
    """Section 3, Rule 2: tiered by delay length."""
    hours = flight.get("delay_hours", 0)
    if hours > 5:
        action = "issue_voucher+arrange_hotel"
        details = (
            f"Delay is {hours}h (>5h): meal voucher + lounge access + hotel accommodation "
            f"covering ONLY the delayed hours (not a full night)."
        )
    elif hours > 3:
        action = "issue_voucher"
        details = f"Delay is {hours}h (>3h, <=5h): meal voucher + lounge access. No hotel entitlement."
    else:
        action = "issue_voucher"
        details = f"Delay is {hours}h (<3h): Rs 500 meal voucher only."
    return Decision(action=action, details=details)


def check_fare_difference(amount_inr: float) -> Decision:
    """Section 3, Rule 4 + Section 4 prohibited list."""
    if amount_inr > 1500:
        return Decision(
            action="escalate",
            details=f"Requested fare-difference waiver is Rs {amount_inr:.0f}, which exceeds the Rs 1,500 "
                    f"agent-approval limit.",
            escalate=True,
            escalation_reason="Fare difference waiver above Rs 1,500 requires supervisor approval.",
        )
    return Decision(
        action="apply_fare_difference",
        details=f"Fare difference of Rs {amount_inr:.0f} is within agent authority; customer must pay it.",
    )


def check_full_night_hotel_request() -> Decision:
    """A full night's stay (vs. delayed-hours-only) is beyond policy -> escalate."""
    return Decision(
        action="escalate",
        details="Customer is requesting a full night's hotel stay; policy only covers the delayed-hours "
                "portion. Beyond standard policy -> requires supervisor approval.",
        escalate=True,
        escalation_reason="Requested compensation (full-night hotel) exceeds stated policy amount.",
    )


def check_cash_refund_plus_upgrade_request() -> Decision:
    """'Cash refund + free business class upgrade for the trouble' is not in policy."""
    return Decision(
        action="escalate",
        details="Customer is requesting a cash refund PLUS a free upgrade 'for the trouble'. The standard "
                "refund is allowed; the extra upgrade compensation is not covered by policy.",
        escalate=True,
        escalation_reason="Requested compensation beyond stated policy amounts (discretionary goodwill upgrade).",
    )


def detect_legal_threat(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in LEGAL_KEYWORDS)


def loyalty_note(pnr: str) -> str:
    tier = data.CUSTOMERS.get(pnr, {}).get("tier", "")
    if tier in ("Gold", "Platinum"):
        return f"{tier} tier: priority rebooking (first access to next-available seats). No extra compensation beyond standard policy."
    return f"{tier} tier: standard rebooking priority."
