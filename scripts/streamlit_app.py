import streamlit as st
import requests
import json
from datetime import datetime
import time

# Configure the page
st.set_page_config(
    page_title="Kubernetes Issue Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .source-box {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .metric-box {
        background-color: #e6f3ff;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🤖 Kubernetes Issue Assistant")
st.markdown("""
    This assistant helps you find answers to your Kubernetes questions by searching through GitHub issues.
    Simply enter your question below and get relevant information from the Kubernetes community.
""")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Create a form for the query
with st.form("query_form"):
    query = st.text_area(
        "Enter your question about Kubernetes:",
        placeholder="e.g., How do I filter Kubernetes pods by labels?",
        height=100
    )
    submit_button = st.form_submit_button("Ask")

# Process the query when submitted
if submit_button and query:
    try:
        # Show a spinner while processing
        with st.spinner("Searching for relevant information..."):
            # Make request to FastAPI backend
            response = requests.post(
                "http://127.0.0.1:8000/query",
                json={"question": query},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "question": query,
                    "answer": result["answer"],
                    "sources": result["sources"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "processing_time": result["processing_time"]
                })
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
    
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend server. Please make sure the FastAPI server is running.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Display chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("Recent Questions and Answers")
    
    for chat in reversed(st.session_state.chat_history):
        with st.expander(f"Q: {chat['question']} ({chat['timestamp']})"):
            # Display the answer
            st.markdown("**Answer:**")
            st.write(chat["answer"])
            
            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Processing Time", f"{chat['processing_time']:.2f} seconds")
            with col2:
                st.metric("Sources Found", len(chat["sources"]))
            
            # Display sources
            st.markdown("**Sources:**")
            for i, source in enumerate(chat["sources"], 1):
                with st.container():
                    st.markdown(f"""
                        <div class="source-box">
                            <strong>Issue #{source['issue_number']}</strong><br>
                            <a href="{source['url']}" target="_blank">{source['url']}</a><br>
                            {source['content']}
                        </div>
                    """, unsafe_allow_html=True)

# Add a sidebar with information
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
        This application uses a RAG (Retrieval-Augmented Generation) system to provide
        accurate answers to Kubernetes-related questions by searching through GitHub issues.
        
        **Features:**
        - Real-time question answering
        - Source document retrieval
        - Performance metrics
        - Chat history
    """)
    
    st.markdown("### How to Use")
    st.markdown("""
        1. Enter your question in the text area
        2. Click 'Ask' to submit
        3. View the answer and relevant sources
        4. Check the chat history for previous questions
    """)
    
    # Add a clear history button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.experimental_rerun()

if __name__ == "__main__":
    # This is needed for running the script directly
    pass 