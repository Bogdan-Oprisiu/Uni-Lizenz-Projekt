import time

import streamlit as st


def force_reload():
    # Inject a small script to reload the browser page
    reload_script = """
    <script>
    window.location.reload();
    </script>
    """
    st.markdown(reload_script, unsafe_allow_html=True)
    st.stop()


st.set_page_config(page_title="Talk to Optimus Prime", layout="wide")
st.title("Talk to Optimus")

# Greeting
st.markdown("""
**Greetings, human! I am Optimus Prime, your friendly AI assistant.**
I'm here to help you with your commands. Let's chat!
""")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display existing messages at the top
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Optimus Prime:** {msg['content']}")

# Build a form so that pressing "Send" triggers exactly one script run
with st.form("chat_form", clear_on_submit=True):
    # Columns for single-line layout: text box and Send button
    col1, col2 = st.columns([8, 1])
    with col1:
        user_input = st.text_input(
            label="",
            placeholder="Type your message here",
            label_visibility="collapsed"
        )
    with col2:
        submitted = st.form_submit_button("Send")

    if submitted and user_input.strip():
        # 1. User message
        st.session_state["messages"].append(
            {"role": "user", "content": user_input}
        )

        # 2. Simulate AI "thinking" with a spinner
        with st.spinner("Optimus Prime is thinking..."):
            time.sleep(2)  # Replace with real logic (e.g. call your backend or LLM)

        # 3. AI response
        system_reply = "This is a placeholder response from Optimus Prime."
        st.session_state["messages"].append(
            {"role": "assistant", "content": system_reply}
        )

        # 4. Force a page refresh so the user instantly sees the new messages
        force_reload()
