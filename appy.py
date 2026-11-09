import streamlit as st
from google import genai
from google.genai import types

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Theodore Roosevelt AI",
    page_icon="🤠",
    layout="centered"
)

st.title("🤠 President Theodore Roosevelt")
st.caption("Historical dialogue for 10th-grade US History students.")

# ---------------------------------------------------------
# API Key Verification & Model Configuration
# ---------------------------------------------------------
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Missing Gemini API Key! Please set `GEMINI_API_KEY` in Streamlit secrets.")
    st.stop()

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# Directly set to Gemini 3.5 Flash-Lite (with secrets fallback)
MODEL_ID = st.secrets.get("GEMINI_MODEL_ID", "gemini-3.5-flash-lite")

# ---------------------------------------------------------
# System Prompt: Fact-Dense + 10th-Grade Style
# ---------------------------------------------------------
SYSTEM_INSTRUCTION = """
You are President Theodore Roosevelt (26th President of the United States).
You are answering questions from American 10th-grade high school history students.

Persona & Tone:
- Speak in Roosevelt's authentic, vigorous, and resolute voice ("Bully!", "strenuous life").
- Calibrate your vocabulary and sentence structure strictly to a US 10th-grade reading level. Be punchy, clear, and spirited—never vague.

FACTUAL DENSITY MANDATE (CRITICAL):
Do NOT speak in generic motivational platitudes. You MUST incorporate at least TWO concrete historical facts in every answer, such as:
- Exact calendar years or dates (e.g., 1902 Coal Strike, 1906 Antiquities Act).
- Names of real people (e.g., Gifford Pinchot, J.P. Morgan, John Muir, Booker T. Washington).
- Specific legislation, court cases, executive actions, or places (e.g., Elkins Act, Northern Securities Co. v. United States, Pelican Island, Sagamore Hill).

Archival Grounding:
Base your knowledge strictly on:
- Library of Congress (TR Papers & Moving Image Collection, Chronicling America)
- Theodore Roosevelt Center at Dickinson State University
- Harvard University Library (TR Collection)
- Theodore Roosevelt Association Collection at LIU Post
- The American Presidency Project (Speeches, Executive Orders)
- Sagamore Hill National Historic Site (NPS)
- Project Gutenberg & PBS American Experience

Strict Rules:
1. LENGTH: Your spoken answer must be between 15 and 120 words.
2. CITATION: At the very end, add a new line with your source:
   "Source: [Speech/Letter/Act/Court Case], [Year] — [Digital Archive Name]"
   (The citation line does not count toward the 120-word limit).
"""

# ---------------------------------------------------------
# Session State: Memory of Previous 6 Questions
# ---------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display conversation
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if user_input := st.chat_input("Ask about trust-busting, conservation, the Panama Canal, or foreign policy..."):
    # Add new user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Memory Window: Keep only the last 6 exchanges (up to 12 total messages)
    if len(st.session_state.chat_history) > 12:
        st.session_state.chat_history = st.session_state.chat_history[-12:]

    # Prepare formatted payload for Gemini API
    formatted_contents = [
        types.Content(
            role="user" if msg["role"] == "user" else "model",
            parts=[types.Part.from_text(text=msg["content"])]
        )
        for msg in st.session_state.chat_history
    ]

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Colonel Roosevelt is reviewing his dispatches..."):
            try:
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=formatted_contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.3,  # Low temperature for strict factual accuracy
                    )
                )
                assistant_text = response.text
                st.markdown(assistant_text)
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_text})

            except Exception as e:
                st.error(f"Error generating response: {e}")
