"""Streamlit frontend placeholder for Finance Research Copilot."""
import requests
import streamlit as st

# -----------------------------
# Configuration
# -----------------------------

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Finance Research Copilot",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Finance Research Copilot")

tab1, tab2, tab3 = st.tabs(
    [
        "Company Analysis",
        "Upload Report",
        "Chat with Report",
    ]
)

# -----------------------------
# Company Analysis
# -----------------------------

with tab1:

    st.header("Analyze a Company")

    company = st.text_input(
        "Company Ticker",
        placeholder="AAPL",
    )

    if st.button("Analyze Company"):

        if not company:
            st.warning("Please enter a company ticker.")
        else:

            with st.spinner("Analyzing..."):

                response = requests.post(
                    f"{API_URL}/analyze",
                    json={"company": company},
                )

            if response.status_code == 200:

                result = response.json()

                st.success("Analysis Complete")

                st.markdown(result["analysis"])

            else:

                st.error(response.text)

# -----------------------------
# Upload Report
# -----------------------------

with tab2:

    st.header("Upload Annual Report")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
    )

    if uploaded_file is not None:

        if st.button("Upload Report"):

            with st.spinner("Uploading..."):

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf",
                    )
                }

                response = requests.post(
                    f"{API_URL}/upload",
                    files=files,
                )

            if response.status_code == 200:

                result = response.json()

                st.success(
                    f"Stored {result['chunks']} chunks."
                )

            else:

                st.error(response.text)

# -----------------------------
# Chat
# -----------------------------

with tab3:

    st.header("Chat with Report")

    question = st.text_input(
        "Ask a question",
        placeholder="What are the company's biggest risks?",
    )

    if st.button("Ask"):

        if not question:
            st.warning("Enter a question.")

        else:

            with st.spinner("Thinking..."):

                response = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "question": question,
                    },
                )

            if response.status_code == 200:

                result = response.json()

                st.markdown(result["answer"])

            else:

                st.error(response.text)