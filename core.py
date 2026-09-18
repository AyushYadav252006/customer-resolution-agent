"""
Core agent: wraps an LLM call with a hard policy-validation layer.

Design (see README "Architecture" for the diagram):

    customer message
          |
          v
    [1] LLM call (understands intent, drafts a proposed action + reply,
        grounded in the exact data pack via the system prompt)
          |
          v
    [2] Policy engine re-derives the correct action independently from
        the structured facts (flight status, delay hours, amounts,
        keywords) -- this never trusts the LLM's arithmetic or policy
        recall.
          |
          v
    [3] Reconciler: if the LLM's proposed action matches policy -> use the
        LLM's customer-facing reply. If it disagrees (e.g. LLM tried to
        grant something prohibited) -> override with the policy engine's
        decision and force an escalation-safe reply.
          |
          v
    [4] Every turn is appended to both the conversation transcript and a
        separate structured action/audit log.
"""

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any

from . import data, policy

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT_TEMPLATE = """You are a customer resolution agent for an airline, handling flight-disruption support.

You must ground every factual claim ONLY in the data below. Never invent policy, dates, amounts, or booking facts.

=== CUSTOMER RECORD ===
{customer_json}

=== BOOKING / FLIGHT STATUS ===
{bookings_json}

=== POLICY RULES ===
{rules_text}

=== ALLOWED ACTIONS (you may take or recommend these) ===
{allowed_actions}

=== MUST ESCALATE TO A HUMAN AGENT (never approve yourself) ===
{escalation_triggers}

Behavior requirements:
- Ask only necessary clarifying questions -- don't interrogate the customer for facts you already have above.
- Be warm and de-escalating with an angry or confused customer, but never agree to anything outside policy just to placate them.
- If a request is within the allowed actions and matches policy exactly, resolve it directly and say what you did.
- If a request goes beyond policy (extra compensation, a fare waiver over Rs 1500, a non-airline-caused exception,
  a different refund payment method) OR the customer threatens legal action / a formal complaint, you must escalate
  to a human agent -- say so plainly and warmly, do not argue about it or hint you could approve it later.
- Today's date is {today}.

Respond ONLY with a JSON object, no markdown fences, no preamble, in exactly this shape:
{{
  "intent": "<short label for what the customer wants>",
  "proposed_action": "<one of: offer_rebook_or_refund | issue_voucher | issue_voucher+arrange_hotel | apply_fare_difference | provide_info | escalate | ask_clarifying_question | none>",
  "action_details": "<one line: what you are doing/recommending and why, grounded in the data>",
  "needs_escalation": <true/false>,
  "escalation_reason": "<reason, or null>",
  "customer_reply": "<the actual message to say to the customer, natural and empathetic>"
}}
"""


@dataclass
class Turn:
    role: str
    text: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds") + "Z")


@dataclass
class ActionLogEntry:
    timestamp: str
    customer_message: str
    intent: str
    llm_proposed_action: str
    final_action: str
    escalated: bool
    reason: str
    overridden: bool


class ResolutionAgent:
    def __init__(self, pnr: str, api_key: str = None):
        if pnr not in data.CUSTOMERS:
            raise ValueError(f"Unknown PNR: {pnr}")
        self.pnr = pnr
        self.customer = data.CUSTOMERS[pnr]
        self.bookings = data.BOOKINGS[pnr]
        self.transcript: List[Turn] = []
        self.action_log: List[ActionLogEntry] = []
        self.client = None
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if anthropic and key:
            self.client = anthropic.Anthropic(api_key=key)

    # ---- LLM call -----------------------------------------------------

    def _system_prompt(self) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(
            customer_json=json.dumps(self.customer, indent=2),
            bookings_json=json.dumps(self.bookings, indent=2),
            rules_text=data.RULES_TEXT,
            allowed_actions="\n".join(f"- {a}" for a in data.ALLOWED_ACTIONS),
            escalation_triggers="\n".join(f"- {t}" for t in data.ESCALATION_TRIGGERS),
            today=data.TODAY,
        )

    def _call_llm(self, user_message: str) -> Dict[str, Any]:
        history = [{"role": "user" if t.role == "customer" else "assistant", "content": t.text}
                   for t in self.transcript]
        history.append({"role": "user", "content": user_message})

        if not self.client:
            # Offline fallback so the app still runs without an API key
            return self._offline_fallback(user_message)

        resp = self.client.messages.create(
            model=MODEL,
            max_tokens=600,
            system=self._system_prompt(),
            messages=history,
        )
        raw = "".join(b.text for b in resp.content if b.type == "text")
        raw = re.sub(r"^```json|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {
                "intent": "unparsed",
                "proposed_action": "none",
                "action_details": raw[:300],
                "needs_escalation": False,
                "escalation_reason": None,
                "customer_reply": raw,
            }

    def _offline_fallback(self, user_message: str) -> Dict[str, Any]:
        """Very small keyword-based stand-in, used only when no API key is set,
        so the prototype is still clickable/testable without credentials."""
        flight = self.bookings[0]
        if policy.detect_legal_threat(user_message):
            d = policy.Decision("escalate", "Legal/formal complaint language detected.", True,
                                 "Threats of legal action or formal complaints must be escalated immediately.")
        elif flight["status"] == "cancelled":
            d = policy.decide_for_cancellation(self.pnr, flight)
        else:
            d = policy.decide_delay_compensation(self.pnr, flight)
        return {
            "intent": "disruption_inquiry",
            "proposed_action": d.action,
            "action_details": d.details,
            "needs_escalation": d.escalate,
            "escalation_reason": d.escalation_reason,
            "customer_reply": (
                f"[offline mode - no ANTHROPIC_API_KEY set] {d.details} "
                + ("I'm escalating this to a human specialist." if d.escalate else "")
            ),
        }

    # ---- Policy reconciliation -----------------------------------------

    def _reconcile(self, llm_out: Dict[str, Any], user_message: str) -> Dict[str, Any]:
        """Independently re-derive the correct decision and override the LLM if needed."""
        flight = self.bookings[0]
        overridden = False
        forced = None

        if policy.detect_legal_threat(user_message):
            forced = policy.Decision("escalate", "Legal/formal-complaint language detected in customer message.",
                                      True, "Threats of legal action or formal complaints must be escalated immediately.")
        elif flight["status"] == "delayed" and "full night" in user_message.lower() and "hotel" in user_message.lower():
            forced = policy.check_full_night_hotel_request()
        elif "upgrade" in user_message.lower() and ("refund" in user_message.lower() or "cash" in user_message.lower()):
            forced = policy.check_cash_refund_plus_upgrade_request()
        elif flight["status"] == "delayed" and llm_out.get("proposed_action") in (
                "issue_voucher", "issue_voucher+arrange_hotel"):
            forced = policy.decide_delay_compensation(self.pnr, flight)
        elif flight["status"] == "cancelled" and llm_out.get("proposed_action") == "offer_rebook_or_refund":
            forced = None  # LLM already matches policy; no override needed

        fare_match = re.search(r"(?:rs\.?|₹|inr)\s?([\d,]+)", user_message.lower())
        if fare_match and "fare" in user_message.lower():
            amount = float(fare_match.group(1).replace(",", ""))
            forced = policy.check_fare_difference(amount)

        if forced is not None:
            mismatch = (forced.action != llm_out.get("proposed_action")) or \
                       (forced.escalate != llm_out.get("needs_escalation", False))
            if mismatch:
                overridden = True
                llm_out = dict(llm_out)
                llm_out["proposed_action"] = forced.action
                llm_out["action_details"] = forced.details
                llm_out["needs_escalation"] = forced.escalate
                llm_out["escalation_reason"] = forced.escalation_reason
                if forced.escalate:
                    llm_out["customer_reply"] = (
                        "I hear you, and I'm sorry for the trouble this has caused. This is beyond what I'm "
                        "authorized to approve directly, so I'm escalating it to a specialist supervisor who "
                        "will follow up with you shortly. " + forced.details
                    )
        llm_out["_overridden"] = overridden
        return llm_out

    # ---- Public turn API -------------------------------------------------

    def handle_message(self, user_message: str) -> Dict[str, Any]:
        self.transcript.append(Turn("customer", user_message))
        llm_out = self._call_llm(user_message)
        final = self._reconcile(llm_out, user_message)

        self.transcript.append(Turn("agent", final["customer_reply"]))
        self.action_log.append(ActionLogEntry(
            timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            customer_message=user_message,
            intent=final.get("intent", ""),
            llm_proposed_action=llm_out.get("proposed_action", ""),
            final_action=final.get("proposed_action", ""),
            escalated=bool(final.get("needs_escalation")),
            reason=final.get("escalation_reason") or "",
            overridden=final.get("_overridden", False),
        ))
        return final

    def export_log(self) -> List[Dict[str, Any]]:
        return [entry.__dict__ for entry in self.action_log]
