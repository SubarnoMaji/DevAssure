import streamlit as st
import requests

# API URLs
UPLOAD_API_URL = "http://localhost:8001"
AGENT_API_URL = "http://localhost:8002"

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# ============== SIDEBAR - File Upload ==============
with st.sidebar:
    st.header("📁 Document Management")

    # File upload section
    st.subheader("Upload Files")
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=["txt", "pdf", "docx", "csv", "jpg", "jpeg", "png", "bmp", "tiff"],
        accept_multiple_files=True,
        help="Supported: TXT, PDF, DOCX, CSV, Images"
    )

    if uploaded_files:
        if st.button("Upload Files", type="primary", use_container_width=True):
            with st.spinner("Uploading..."):
                for uploaded_file in uploaded_files:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    try:
                        response = requests.post(f"{UPLOAD_API_URL}/upload", files=files)
                        if response.status_code == 200:
                            st.success(f"Uploaded: {uploaded_file.name}")
                        else:
                            try:
                                error_msg = response.json().get('detail', 'Unknown error')
                            except Exception:
                                error_msg = response.text or f"Status: {response.status_code}"
                            st.error(f"Failed: {uploaded_file.name} - {error_msg}")
                    except requests.exceptions.ConnectionError:
                        st.error("Upload API not connected (port 8001)")
                        break
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        break

    st.divider()

    # List existing files
    st.subheader("Uploaded Files")
    if st.button("Refresh", use_container_width=True):
        st.rerun()

    try:
        response = requests.get(f"{UPLOAD_API_URL}/files")
        if response.status_code == 200:
            files = response.json().get("files", [])
            if files:
                for file in files:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(file[:20] + "..." if len(file) > 20 else file)
                    with col2:
                        if st.button("X", key=f"del_{file}", help=f"Delete {file}"):
                            try:
                                del_response = requests.delete(f"{UPLOAD_API_URL}/files/{file}")
                                if del_response.status_code == 200:
                                    st.rerun()
                            except Exception:
                                pass
            else:
                st.info("No files uploaded")
        else:
            st.warning("Could not fetch files")
    except requests.exceptions.ConnectionError:
        st.warning("Upload API offline")
    except Exception:
        st.warning("Error fetching files")

    st.divider()

    # Settings
    st.subheader("Settings")
    use_rag = st.checkbox("Use RAG (Document Search)", value=True)

    st.divider()

    # Clear chat
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ============== MAIN AREA - Chatbot ==============
st.title("🤖 RAG Chatbot")
st.caption("Chat with your documents using AI")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if use_rag:
                    # Use RAG query endpoint
                    response = requests.post(
                        f"{AGENT_API_URL}/query",
                        json={"query": prompt, "use_retrieval": True},
                        timeout=60
                    )
                else:
                    # Use simple query endpoint
                    response = requests.post(
                        f"{AGENT_API_URL}/simple-query",
                        json={"prompt": prompt},
                        timeout=60
                    )

                if response.status_code == 200:
                    result = response.json()
                    if use_rag:
                        answer = result.get("answer", "No response")
                    else:
                        answer = result.get("response", "No response")

                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    try:
                        error_msg = response.json().get('detail', 'Unknown error')
                    except Exception:
                        error_msg = f"Status: {response.status_code}"
                    error_text = f"Error: {error_msg}"
                    st.error(error_text)
                    st.session_state.messages.append({"role": "assistant", "content": error_text})

            except requests.exceptions.ConnectionError:
                error_text = "Agent API not connected. Make sure it's running on port 8002."
                st.error(error_text)
                st.session_state.messages.append({"role": "assistant", "content": error_text})
            except requests.exceptions.Timeout:
                error_text = "Request timed out. Please try again."
                st.error(error_text)
                st.session_state.messages.append({"role": "assistant", "content": error_text})
            except Exception as e:
                error_text = f"Error: {str(e)}"
                st.error(error_text)
                st.session_state.messages.append({"role": "assistant", "content": error_text})
