import streamlit as st
import traceback

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("RAG Chatbot")
st.write("Ask questions about PMKVY Schemes based on the uploaded Government documents.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load the RAG pipeline
@st.cache_resource
def get_rag_pipeline():
    # Importing inside the cached function guarantees it's only loaded once per Streamlit server
    try:
        from RAG import ask_question
        return ask_question
    except Exception as e:
        error_msg = f"Error loading RAG pipeline: {e}\n\n{traceback.format_exc()}"
        return lambda q: (error_msg, [])

ask_question = get_rag_pipeline()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about PMKVY..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Generating answer..."):
        answer, docs = ask_question(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(answer)
        
        # Display context in an expander for debugging later
        if docs:
            with st.expander("View Retrieved Context"):
                for i, doc in enumerate(docs):
                    source = doc.metadata.get("source", "Unknown")
                    st.markdown(f"**Source {i+1} :** `{source}`")
                    st.markdown(doc.page_content)
                    st.markdown("---")
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
