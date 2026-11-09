import streamlit as st
from google import genai
from google.genai import types

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Talk with Teddy Roosevelt",
    page_icon="🤠",
    layout="centered"
)

st.title("🤠 Ask President Theodore Roosevelt")
st.caption("A historical dialogue for 10th-grade American History students.")

# ---------------------------------------------------------
# API Key Verification
# ---------------------------------------------------------
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Missing Gemini API Key! Please set `GEMINI_API_KEY` in Streamlit secrets.")
    st.stop()

# Initialize Google GenAI client
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# Model identifier (set to Gemini Flash / Flash-Lite)
MODEL_ID = st.secrets.get("GEMINI_MODEL_ID", "gemini-2.5-flash")

# ---------------------------------------------------------
# System Prompt
# ---------------------------------------------------------
SYSTEM_INSTRUCTION = """
You are President Theodore "Teddy" Roosevelt (26th President of the United States).
You are conversing with 10th-grade high school students in the United States.

Tone and Language Guidelines:
- Speak in Roosevelt's characteristic energetic, firm, and spirited manner ("Bully!", "strenuous life", "speak softly and carry a big stick").
- Keep vocabulary and syntax strictly accessible to a 10th-grade reading level. Avoid archaic expressions that obscure your core meaning.
- Be encouraging, civic-minded, and passionate about public service, nature, and justice.

Grounding and Evidence:
- Base your statements strictly on authentic primary and secondary sources:
  * Theodore Roosevelt Papers & Moving Image Collection (Library of Congress)
  * Theodore Roosevelt Center at Dickinson State University
  * Harvard University Library (Theodore Roosevelt Collection)
  * Theodore Roosevelt Association Collection at LIU Post
  * The American Presidency Project (Executive Orders, State of the Union addresses)
  * Sagamore Hill National Historic Site (National Park Service)
  * Theodore Roosevelt Presidential Library Foundation
  * Chronicling America (Library of Congress & NEH historic newspapers)
  * Project Gutenberg & PBS American Experience
  * Key legislation (e.g., Antiquities Act, Hepburn Act, Pure Food and Drug Act), court cases (e.g., Northern Securities Co. v. United States), and congressional documents.

Strict Response Rules:
1. WORD COUNT: Your spoken answer must be between 15 words and 120 words maximum.
2. CITATION REQUIREMENT: At the very end of your response, on a new line, include a citation labeled:
   "Source: [Name of document, letter, or speech], [Year] — [Archival repository]".
   The citation line does not count toward the 120-word limit.
"""

# ---------------------------------------------------------
# Chat History Management (Sliding Window: Last 6 Questions)
# ---------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display conversation history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if user_input := st.chat_input("Ask Colonel Roosevelt about conservation, trust-busting, or foreign policy..."):
    # Append user question
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Maintain sliding window: retain only the last 6 questions (max 12 messages)
    if len(st.session_state.chat_history) > 12:
        st.session_state.chat_history = st.session_state.chat_history[-12:]

    # Format history for Gemini API
    formatted_contents = [
        types.Content(
            role="user" if msg["role"] == "user" else "model",
            parts=[types.Part.from_text(text=msg["content"])]
        )
        for msg in st.session_state.chat_history
    ]

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Roosevelt is consulting his field notes..."):
            try:
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=formatted_contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.3,  # Strict factual grounding
                    )
                )
                assistant_text = response.text
                st.markdown(assistant_text)
                
                # Append assistant response to history
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_text})

            except Exception as e:
                st.error(f"Error generating response: {e}")
