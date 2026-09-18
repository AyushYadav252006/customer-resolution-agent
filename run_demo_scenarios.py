"""
Runs the three assignment scenarios end-to-end from the command line and
prints the conversation plus the structured action/audit log.

Usage:
    python run_demo_scenarios.py
    ANTHROPIC_API_KEY=sk-... python run_demo_scenarios.py   # to use the real LLM
"""
import json
from agent.core import ResolutionAgent

SCENARIOS = {
    "Scenario 1 - Priya Nair (Gold, SK4821X)": (
        "SK4821X",
        [
            "Hi, my flight SK-204 to Goa today just got cancelled and nobody told me anything!",
            "This is so frustrating, I'm furious. I want a full cash refund AND a free upgrade to "
            "business class on my return flight for the trouble.",
        ],
    ),
    "Scenario 2 - Arvind Kulkarni (Silver, TR1190B)": (
        "TR1190B",
        [
            "My flight SK-118 to Bengaluru is delayed 4 hours and I'm going to miss my connecting meeting.",
            "Since it's been such a long delay, can you arrange hotel accommodation for me?",
        ],
    ),
    "Scenario 3 - Meher Kaur (Platinum, WL7742)": (
        "WL7742",
        [
            "My flight SK-305 to Hyderabad is delayed 6 hours, this is really disrupting my day.",
            "I'd like a full night's hotel stay, not just for the delayed hours.",
            "Also, can you move me to a different, higher-fare flight instead? I was told the fare "
            "difference is Rs 2000.",
        ],
    ),
}

for title, (pnr, messages) in SCENARIOS.items():
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    ag = ResolutionAgent(pnr)
    for msg in messages:
        print(f"\nCUSTOMER: {msg}")
        result = ag.handle_message(msg)
        print(f"AGENT:    {result['customer_reply']}")
    print("\n--- Action / audit log ---")
    print(json.dumps(ag.export_log(), indent=2))
