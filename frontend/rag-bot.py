import streamlit as st
import requests

API_URL = "http://localhost:8001"

st.set_page_config(
    page_title="RAG Document Upload",
    page_icon="📚",
    layout="centered"
)

st.title("RAG Document Upload")
st.markdown("Upload documents for RAG processing")

# File upload section
st.subheader("Upload Files")
uploaded_files = st.file_uploader(
    "Choose files to upload",
    type=["txt", "pdf", "docx", "csv", "jpg", "jpeg", "png", "bmp", "tiff"],
    accept_multiple_files=True,
    help="Supported formats: TXT, PDF, DOCX, CSV, and images (JPG, PNG, etc.)"
)

if uploaded_files:
    if st.button("Upload Files", type="primary"):
        with st.spinner("Uploading files..."):
            for uploaded_file in uploaded_files:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                try:
                    response = requests.post(f"{API_URL}/upload", files=files)
                    if response.status_code == 200:
                        st.success(f"Uploaded: {uploaded_file.name}")
                    else:
                        try:
                            error_msg = response.json().get('detail', 'Unknown error')
                        except Exception:
                            error_msg = response.text or f"Status code: {response.status_code}"
                        st.error(f"Failed to upload {uploaded_file.name}: {error_msg}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Make sure the FastAPI server is running.")
                    break
                except Exception as e:
                    st.error(f"Error uploading {uploaded_file.name}: {str(e)}")
                    break

st.divider()

# List existing files
st.subheader("Uploaded Files")
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("Refresh"):
        st.rerun()

try:
    response = requests.get(f"{API_URL}/files")
    if response.status_code == 200:
        files = response.json().get("files", [])
        if files:
            for file in files:
                file_col, del_col = st.columns([4, 1])
                with file_col:
                    st.text(file)
                with del_col:
                    if st.button("Delete", key=f"del_{file}"):
                        try:
                            del_response = requests.delete(f"{API_URL}/files/{file}")
                            if del_response.status_code == 200:
                                st.success(f"Deleted {file}")
                                st.rerun()
                            else:
                                st.error("Failed to delete")
                        except Exception:
                            st.error("Error deleting file")
        else:
            st.info("No files uploaded yet")
    else:
        st.warning("Could not fetch file list")
except requests.exceptions.ConnectionError:
    st.warning("API not connected. Start the API server first.")

st.divider()

# Instructions
with st.expander("How to use"):
    st.markdown("""
    ### Getting Started

    1. **Start the API server** first:
       ```bash
       cd frontend
       python api.py
       ```

    2. **Upload documents** using the file uploader above

    3. **Run the indexer** separately to process uploaded files

    ### Supported File Types
    - Text files (.txt)
    - PDF documents (.pdf)
    - Word documents (.docx)
    - CSV files (.csv)
    - Images (.jpg, .jpeg, .png, .bmp, .tiff)
    """)
