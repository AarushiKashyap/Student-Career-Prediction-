import streamlit as st
from google import genai
from dotenv import load_dotenv
import os
import time

# Load API key
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


# Create Gemini client
@st.cache_resource
def get_client():
    return genai.Client(api_key=api_key)


client = get_client()


# Store conversation
if "messages" not in st.session_state:
    st.session_state.messages = []


# Page
st.title("🤖 My GenAI Chatbot")
st.write("Ask me anything!")


# Get question
question = st.chat_input("Ask your question...")


if question:

    # Store user's question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Prepare conversation history
    conversation = []

    for message in st.session_state.messages:
        conversation.append({
            "role": message["role"],
            "parts": [
                {"text": message["content"]}
            ]
        })

    # Try Gemini 3 times
    for attempt in range(3):

        try:
            response = client.models.generate_content(
                model="gemini-3.7-flash",
                contents=conversation
            )

            answer = response.text
            break

        except Exception as e:

            if attempt < 2:
                time.sleep(3 * (attempt + 1))
            else:
                answer = "Gemini is temporarily unavailable. Please try again."


    # Store AI answer
    st.session_state.messages.append({
        "role": "model",
        "content": answer
    })


# Show latest conversation
if st.session_state.messages:

    last_user = None
    last_model = None

    for message in reversed(st.session_state.messages):

        if message["role"] == "model" and last_model is None:
            last_model = message["content"]

        elif message["role"] == "user" and last_user is None:
            last_user = message["content"]

        if last_user and last_model:
            break

    if last_user:
        st.chat_message("user").write(last_user)

    if last_model:
        st.chat_message("assistant").write(last_model)