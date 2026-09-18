# Customer-Facing Resolution Agent — Airline Disruption

Assignment 3 prototype. A support agent that handles cancelled/delayed-flight
conversations, grounded entirely in the supplied data pack, that knows the
difference between "resolve it" and "escalate it."

## Quick start (one command)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens a browser chat UI. Pick a customer (Priya / Arvind / Meher) in the
sidebar, type as that customer, and watch the conversation plus a live
action/audit log build side by side.

Pick a provider in the sidebar:

- **Offline (no key needed)** — deterministic, hand-written replies driven by
  the same policy engine. No signup, no cost. Good for verifying the logic.
- **Google Gemini (free tier)** — get a free key at
  [aistudio.google.com/apikey](https://aistudio.google.com/apikey), paste it
  into the sidebar. Real natural-language conversation at no cost.
- **Anthropic Claude (paid)** — the original design target
  (`claude-sonnet-4-6`); requires a funded Anthropic API key.

All three run through the exact same policy engine and produce the exact
same audit log shape — the provider only changes how naturally the reply is
worded, never what action is actually taken.

Command-line version (prints all three assignment scenarios end-to-end):

```bash
python run_demo_scenarios.py
```

## 1. Architecture & process flow

```
                    ┌─────────────────────┐
                    │   Customer message   │
                    └──────────┬───────────┘
                               v
                 ┌──────────────────────────┐
                 │ 1. LLM call (Claude)      │
                 │  - reads full data pack   │
                 │    (customer, booking,    │
                 │    rules) via system      │
                 │    prompt                 │
                 │  - infers intent          │
                 │  - drafts proposed action │
                 │    + a natural reply      │
                 │  - outputs structured     │
                 │    JSON                   │
                 └──────────┬───────────────┘
                            v
                 ┌──────────────────────────┐
                 │ 2. Policy engine          │
                 │    (pure Python,          │
                 │    deterministic)         │
                 │  - re-derives the correct │
                 │    action independently   │
                 │    from the raw facts     │
                 │    (delay hours, amounts, │
                 │    keywords)              │
                 │  - never trusts the LLM's │
                 │    arithmetic or recall   │
                 └──────────┬───────────────┘
                            v
                 ┌──────────────────────────┐
                 │ 3. Reconciler             │
                 │  - LLM proposal matches   │
                 │    policy?  -> keep reply │
                 │  - mismatch? -> override  │
                 │    with policy decision,  │
                 │    force an escalation-   │
                 │    safe reply, flag the   │
                 │    override in the log    │
                 └──────────┬───────────────┘
                            v
              ┌─────────────┴─────────────┐
              v                           v
     ┌─────────────────┐        ┌───────────────────┐
     │ Conversation log │        │ Structured action /│
     │ (what was said)  │        │ audit log (what was │
     │                  │        │ decided, why, and   │
     │                  │        │ whether escalated)  │
     └─────────────────┘        └───────────────────┘
```

**Why this two-layer design (LLM + policy engine), not LLM-only:** An LLM is
good at understanding messy human language and phrasing an empathetic reply,
but is not reliable for exact threshold math (">3h vs >5h", "is ₹2,000 over
the ₹1,500 cap") or for reliably refusing an out-of-policy ask under
conversational pressure. The policy engine is a small set of pure functions
that re-derive the correct decision from the raw facts every single turn, and
can veto/override the model. This means the *hard constraints from Section 4
of the data pack are enforced in code*, not just requested in a prompt.

## 2. Inputs, sources, and assumptions

**Source of truth:** `agent/data.py` is a direct structured transcription of
the assignment's data pack (customer profiles, booking/transaction table,
service rules, allowed/prohibited actions). No external or invented data is
used anywhere in the system prompt or policy engine.

**Assumptions made (none of these are in the data pack verbatim, but were
needed to make the prototype concrete):**
- "Today" is fixed at 23 Sep 2026 (the date stated for the exercise), so
  delay/cancellation status is treated as current rather than historical.
- The customer identifies themselves by PNR (selected in the UI) rather than
  the agent having to authenticate them mid-conversation — authentication
  flow was out of scope for the data pack given.
- "Escalate" in this prototype means: the agent stops short of approving the
  action, gives the customer a clear, empathetic explanation, and logs the
  reason — it does not simulate an actual human hand-off/ticketing system,
  since none was described in the data pack.
- Currency and amounts are read directly from customer text (e.g. "the fare
  difference is Rs 2000") since the data pack itself doesn't tabulate every
  possible fare-difference amount — only the ₹1,500 approval threshold.
- The three sample prior conversations (Section 5) were used only to match
  tone/style, never as a source of policy or fact, per the data pack's own
  instruction.

## 3. AI tools used and how

- **LLM conversational core (pluggable: Google Gemini `gemini-2.0-flash` or
  Anthropic Claude `claude-sonnet-4-6`)**: whichever provider is selected
  receives the full data pack embedded in its system prompt on every call,
  infers customer intent, decides which of the explicitly allowed actions
  applies (or that escalation is required), and drafts the customer-facing
  reply. It is instructed to output structured JSON so its decision can be
  machine-checked rather than trusted as free text. The provider is
  swappable specifically so the prototype isn't dependent on one paid
  vendor — Gemini's free tier was used for the primary demo.
- **Claude (this conversation, Claude.ai)** was used to design the
  architecture, write the policy-engine guardrail logic, build the Streamlit
  UI, and write this documentation — i.e., as a development/pair-programming
  tool for building the prototype itself, separate from the runtime agent
  described above.
- No other AI services or fine-tuning were used. The offline fallback mode
  (`agent/core.py: _offline_fallback`) is a small hand-written keyword/rule
  matcher — not an AI model — included only so the app is clickable without
  an API key.

## 4. Project layout

```
customer-resolution-agent/
├── agent/
│   ├── data.py     # structured data pack (ground truth)
│   ├── policy.py   # deterministic rule/threshold checks
│   └── core.py     # LLM call + reconciliation + logging
├── app.py                    # Streamlit chat UI + audit log panel
├── run_demo_scenarios.py     # CLI run of all 3 assignment scenarios
├── requirements.txt
└── .env.example
```

## 5. How each scenario is handled (for demo/defense reference)

- **Priya (Gold, cancelled flight):** offers free rebook OR full refund
  (customer's choice) per the cancellation rule. When she additionally asks
  for a cash refund *plus* a free business-class upgrade "for the trouble,"
  the policy engine detects this as compensation beyond the stated policy and
  forces an escalation — even though the LLM alone might be tempted to be
  accommodating.
- **Arvind (Silver, 4h delay):** delay is >3h and ≤5h, so policy is meal
  voucher + lounge access — explicitly *not* a hotel. When he asks for a
  hotel anyway, the agent holds the line and explains why, rather than
  agreeing because he asked twice.
- **Meher (Platinum, 6h delay):** delay is >5h, so the base entitlement is
  voucher + lounge + delayed-hours-only hotel. Her two follow-up asks —a
  full night's stay, and a fare swap with a ₹2,000 difference — each trip a
  separate escalation trigger (exceeds stated policy amount; exceeds the
  ₹1,500 waiver cap) even though she's a Platinum customer, because the
  loyalty tier rule only grants priority rebooking, not extra compensation.

## 6. Known limitations

- Fare amounts are extracted from free text with a simple regex; a
  production system would use a structured "confirm the fare difference"
  step instead of trusting the number the customer states.
- No real authentication, payment, or ticketing integration — actions are
  logged as decisions, not executed against real systems.
- Single-turn "delay hours" are read from the static data pack rather than a
  live flight-status feed.
