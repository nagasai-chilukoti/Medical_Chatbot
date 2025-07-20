import streamlit as st
from llama_cpp import Llama
import os
import urllib.request

# Load GGUF model from Hugging Face if not already downloaded
MODEL_PATH = "BioMistral-7B-GGUF/BioMistral-7B.Q4_K_M.gguf"
MODEL_URL = "https://huggingface.co/TheBloke/BioMistral-7B-GGUF/resolve/main/BioMistral-7B.Q4_K_M.gguf"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading BioMistral model from Hugging Face. Please wait ⏳...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    st.success("Model downloaded successfully! ✅")

# Load GGUF model
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=20,
    n_ctx=2048,
    n_threads=os.cpu_count(),
    chat_format="chatml"
)

# UI Setup
st.set_page_config(page_title="Medical Chatbot (BioMistral)", page_icon="🩺")
st.markdown(
    """
    <h1 style='text-align: center; color: white;'>🩺 Medical Chatbot (Doctor Mistral)</h1>
    <p style='text-align: center;'>Consult with the doctor. Get friendly, clear advice on your health queries.</p>
    """,
    unsafe_allow_html=True
)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "You are a professional and friendly **medical doctor**. "
                "Respond like a doctor giving direct health consultation — you explain symptoms clearly, prescribe advice and speak warmly and professionally."
            )
        }
    ]

# Display chat
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"🧑‍⚕️ **You:** {msg['content']}")
    elif msg["role"] == "assistant":
        st.markdown(f"🩺 **Doctor:** {msg['content']}")

# User input form
with st.form("chat_input", clear_on_submit=True):
    user_input = st.text_input("Ask a medical question...", placeholder="e.g., I feel dizzy and weak")
    submitted = st.form_submit_button("Send")

# Chat processing
if submitted and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Create prompt from history
    conversation = ""
    for msg in st.session_state.messages:
        if msg["role"] == "system":
            continue
        role = "User" if msg["role"] == "user" else "Doctor"
        conversation += f"{role}: {msg['content']}\n"

    prompt = f"""<s>[INST] {st.session_state.messages[0]['content']}

{conversation}
Doctor:"""

    with st.spinner("Doctor is analyzing your condition..."):
        output = llm(prompt, max_tokens=512, stop=["</s>"])
        reply = output["choices"][0]["text"].strip()

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()

# Clear button
if st.button("🧹 Clear Chat"):
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "You are a professional and friendly **medical doctor**. "
                "Respond like a doctor giving direct health consultation — you explain symptoms clearly, prescribe advice and speak warmly and professionally."
            )
        }
    ]
    st.rerun()
