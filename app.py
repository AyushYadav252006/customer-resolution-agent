import streamlit as st
from agent import data
from agent.core import ResolutionAgent

st.set_page_config(page_title="Airline Resolution Agent", layout="wide")

st.title("Customer-Facing Resolution Agent — Airline Disruption")
st.caption("Prototype for Assignment 3. Grounded only in the supplied data pack.")

pnr_options = {f"{v['name']} ({v['tier']}, {k})": k for k, v in data.CUSTOMERS.items()}

with st.sidebar:
    st.header("Select customer")
    label = st.selectbox("Customer", list(pnr_options.keys()))
    pnr = pnr_options[label]

    st.header("AI provider")
    provider_label = st.selectbox(
        "Provider",
        ["Offline (no key needed)", "Google Gemini (free tier)", "Anthropic Claude (paid)"],
    )
    provider_map = {
        "Offline (no key needed)": "offline",
        "Google Gemini (free tier)": "gemini",
        "Anthropic Claude (paid)": "anthropic",
    }
    provider = provider_map[provider_label]

    api_key = None
    if provider == "gemini":
        api_key = st.text_input("Google API key", type="password",
                                 help="Get a free key at aistudio.google.com/apikey")
    elif provider == "anthropic":
        api_key = st.text_input("Anthropic API key", type="password")

    if st.button("Reset conversation"):
        for k in ("agent", "pnr_loaded", "provider_loaded"):
            st.session_state.pop(k, None)
        st.rerun()

    st.divider()
    st.subheader("Booking on file")
    for b in data.BOOKINGS[pnr]:
        st.write(f"**{b['flight']}** {b['route']} — {b['date']} {b['scheduled_departure']}")
        st.write(f"Status: {b['status_detail']}")

if (
    "agent" not in st.session_state
    or st.session_state.get("pnr_loaded") != pnr
    or st.session_state.get("provider_loaded") != provider
):
    st.session_state.agent = ResolutionAgent(pnr, api_key=api_key or None, provider=provider)
    st.session_state.pnr_loaded = pnr
    st.session_state.provider_loaded = provider

agent: ResolutionAgent = st.session_state.agent

col_chat, col_log = st.columns([2, 1])

with col_chat:
    st.subheader("Conversation")
    for turn in agent.transcript:
        with st.chat_message("user" if turn.role == "customer" else "assistant"):
            st.write(turn.text)

    user_msg = st.chat_input("Type as the customer...")
    if user_msg:
        with st.spinner("Agent is responding..."):
            agent.handle_message(user_msg)
        st.rerun()

with col_log:
    st.subheader("Action & audit log")
    if not agent.action_log:
        st.info("No actions yet.")
    for entry in reversed(agent.export_log()):
        icon = "🚨" if entry["escalated"] else "✅"
        with st.expander(f"{icon} {entry['timestamp']} — {entry['final_action']}"):
            st.write(f"**Customer said:** {entry['customer_message']}")
            st.write(f"**Detected intent:** {entry['intent']}")
            st.write(f"**LLM proposed:** {entry['llm_proposed_action']}")
            st.write(f"**Final action (policy-checked):** {entry['final_action']}")
            if entry["overridden"]:
                st.warning("Policy engine overrode the model's proposed action.")
            if entry["escalated"]:
                st.error(f"Escalated: {entry['reason']}")
