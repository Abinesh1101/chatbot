import streamlit as st
import requests

# API URL
API_URL = "http://127.0.0.1:8000/chat"

# App title
st.set_page_config(page_title="Changi Airport Chatbot")
st.title("🛬 Changi Airport Chatbot")

# User input
question = st.text_input("Ask me anything about Changi Airport or Jewel:", "")

# Send query on button click
if st.button("Ask"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            response = requests.post(API_URL, json={"question": question})
            if response.status_code == 200:
                data = response.json()
                st.markdown("### 🤖 Answer:")
                st.success(data["response"])

                if data["context_sources"]:
                    st.markdown("### 📚 Sources:")
                    for i, source in enumerate(data["context_sources"], 1):
                        score = data["relevance_scores"][i-1]
                        st.markdown(f"{i}. `{source}` (relevance: `{score:.2f}`)")
                
                st.caption(f"⏱️ Response time: {data['response_time']:.2f} seconds")
            else:
                st.error("❌ Failed to get a response from the API.")
