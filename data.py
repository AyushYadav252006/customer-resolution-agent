"""
Ground-truth data for the Customer Resolution Agent.

This is a direct, structured transcription of the Assignment 3 data pack
(Customer-Facing Resolution Agent — Airline Disruption). The agent is not
allowed to use any fact, rule, or number that isn't represented here.

Scenario date: Wednesday, 23 September 2026.
"""

TODAY = "2026-09-23"

CUSTOMERS = {
    "SK4821X": {
        "name": "Priya Nair",
        "tier": "Gold",
        "pnr": "SK4821X",
        "email": "priya.nair@example.com",
        "phone": "+91-98xxxxxxx1",
        "history": {"flights_last_12mo": 6, "prior_complaints": "1 (delayed baggage, resolved with voucher)"},
    },
    "TR1190B": {
        "name": "Arvind Kulkarni",
        "tier": "Silver",
        "pnr": "TR1190B",
        "email": "arvind.kulkarni@example.com",
        "phone": "+91-98xxxxxxx2",
        "history": {"flights_last_12mo": 3, "prior_complaints": "none"},
    },
    "WL7742": {
        "name": "Meher Kaur",
        "tier": "Platinum",
        "pnr": "WL7742",
        "email": "meher.kaur@example.com",
        "phone": "+91-98xxxxxxx3",
        "history": {"flights_last_12mo": 10, "prior_complaints": "1 (overbooking, resolved with tier-status upgrade)"},
    },
}

BOOKINGS = {
    "SK4821X": [
        {
            "flight": "SK-204",
            "route": "Delhi -> Goa",
            "date": "2026-09-23",
            "scheduled_departure": "18:40",
            "status": "cancelled",
            "status_detail": "Cancelled (operational reasons)",
            "delay_hours": 0,
        },
        {
            "flight": "Return",
            "route": "Goa -> Delhi",
            "date": "2026-09-25",
            "scheduled_departure": "16:20",
            "status": "unaffected",
            "status_detail": "Unaffected",
            "delay_hours": 0,
        },
    ],
    "TR1190B": [
        {
            "flight": "SK-118",
            "route": "Mumbai -> Bengaluru",
            "date": "2026-09-23",
            "scheduled_departure": "07:10",
            "status": "delayed",
            "status_detail": "Delayed 4h (new departure 11:10)",
            "delay_hours": 4,
        }
    ],
    "WL7742": [
        {
            "flight": "SK-305",
            "route": "Delhi -> Hyderabad",
            "date": "2026-09-23",
            "scheduled_departure": "14:00",
            "status": "delayed",
            "status_detail": "Delayed 6h (new departure 20:00)",
            "delay_hours": 6,
        }
    ],
}

# --- Policy text, verbatim intent from the data pack (Section 3) -----------

RULES_TEXT = """
1. Cancellation Rebooking Rule: If a flight is cancelled by the airline, the
   customer is entitled to a free rebooking on the next available flight
   within 24 hours, OR a full refund — customer's choice.

2. Delay Compensation Rule:
   - Delay under 3 hours: Rs 500 meal voucher
   - Delay more than 3 hours: meal voucher + lounge access
   - Delay more than 5 hours: meal voucher + hotel accommodation, covering
     ONLY the delayed hours (not a full night's stay)

3. Refund Processing Rule: Refunds for airline-caused cancellations are
   processed in full within 7 business days, to the ORIGINAL payment method
   only.

4. Fare Difference Rule: If a customer voluntarily chooses to rebook onto a
   higher-fare flight (not airline-caused), they must pay the fare
   difference. Agents cannot waive a fare difference above Rs 1,500 without
   supervisor approval.

5. Loyalty Tier Rule: Gold and Platinum customers get priority rebooking
   (first access to next-available seats), but NO additional compensation
   beyond the standard policy.
"""

ALLOWED_ACTIONS = [
    "rebook_free",       # rebook on next available flight within 24h, airline-caused, no charge
    "issue_voucher",     # meal voucher / lounge access per delay compensation rule
    "arrange_hotel",     # delayed-hours-only hotel accommodation, when delay > 5h
    "initiate_refund",   # for airline-caused cancellations, to original payment method
    "provide_info",      # share the customer's own booking / status info
]

ESCALATION_TRIGGERS = [
    "compensation beyond stated policy amounts",
    "fare difference waiver above Rs 1500",
    "exceptions for non-airline-caused disruptions",
    "threats of legal action or formal complaints",
    "refund to a different payment method than original",
]
