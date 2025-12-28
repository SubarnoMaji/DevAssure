import logging
import os
import time
import hashlib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from utils.parser import (
    TextParser, PDFParser, DocxParser, ImageOCRParser, CSVParser
)
from utils.vector_store import ChromaVectorStore
from utils.vector_store_config import FOLDER_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SUPPORTED_EXTENSIONS = {"txt", "pdf", "docx", "jpg", "jpeg", "png", "bmp", "tiff", "csv"}


def get_parser_for_file(file_path):
    ext = file_path.lower().split('.')[-1]
    if ext == "txt":
        return TextParser()
    elif ext == "pdf":
        return PDFParser()
    elif ext == "docx":
        return DocxParser()
    elif ext in ["jpg", "jpeg", "png", "bmp", "tiff"]:
        return ImageOCRParser()
    elif ext == "csv":
        return CSVParser()
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def generate_doc_id(file_path, index):
    """Generate a unique document ID based on file path and chunk index."""
    file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
    return f"{os.path.basename(file_path)}_{file_hash}_{index}"


def is_supported_file(file_path):
    """Check if file has a supported extension."""
    ext = file_path.lower().split('.')[-1]
    return ext in SUPPORTED_EXTENSIONS and not os.path.basename(file_path).startswith(".")


def index_file(file_path, store):
    """Index a single file into the vector store."""
    if not is_supported_file(file_path):
        return 0

    try:
        parser = get_parser_for_file(file_path)
        if isinstance(parser, CSVParser):
            docs = parser.parse(file_path, delimiter=',', has_header=True)
        else:
            docs = parser.parse(file_path)

        if not docs:
            logging.warning(f"No content extracted from {file_path}")
            return 0

        contents = [doc['content'] for doc in docs]
        ids = [generate_doc_id(file_path, i) for i in range(len(docs))]

        # Add source file to metadata for tracking
        metadatas = []
        for doc in docs:
            meta = doc.get('metadata', {})
            meta['source'] = file_path
            metadatas.append(meta)

        store.add_documents(ids, contents, metadatas)
        logging.info(f"Indexed {len(docs)} chunks from: {os.path.basename(file_path)}")
        return len(docs)

    except Exception as e:
        logging.error(f"Error indexing {file_path}: {e}")
        return 0


def remove_file_from_index(file_path, store):
    """Remove all documents associated with a file from the vector store."""
    try:
        doc_ids = store.get_documents_by_source(file_path)
        if doc_ids:
            store.delete_documents(doc_ids)
            logging.info(f"Removed {len(doc_ids)} chunks for: {os.path.basename(file_path)}")
            return len(doc_ids)
        return 0
    except Exception as e:
        logging.error(f"Error removing {file_path} from index: {e}")
        return 0


class FileWatchHandler(FileSystemEventHandler):
    """Handler for file system events."""

    def __init__(self, store):
        self.store = store
        super().__init__()

    def on_created(self, event):
        if event.is_directory:
            return
        file_path = event.src_path
        if is_supported_file(file_path):
            # Small delay to ensure file is fully written
            time.sleep(0.5)
            logging.info(f"New file detected: {os.path.basename(file_path)}")
            index_file(file_path, self.store)

    def on_deleted(self, event):
        if event.is_directory:
            return
        file_path = event.src_path
        if is_supported_file(file_path):
            logging.info(f"File deleted: {os.path.basename(file_path)}")
            remove_file_from_index(file_path, self.store)

    def on_modified(self, event):
        if event.is_directory:
            return
        file_path = event.src_path
        if is_supported_file(file_path):
            # Small delay to ensure file is fully written
            time.sleep(0.5)
            logging.info(f"File modified: {os.path.basename(file_path)}")
            # Remove old entries and re-index
            remove_file_from_index(file_path, self.store)
            index_file(file_path, self.store)


def initial_index(source_folder, store):
    """Perform initial indexing of all files in the folder."""
    if not os.path.isdir(source_folder):
        logging.error(f"The configured path {source_folder} is not a valid directory.")
        return 0

    files = [
        os.path.join(source_folder, fname)
        for fname in os.listdir(source_folder)
        if is_supported_file(os.path.join(source_folder, fname))
    ]

    total_docs = 0
    for file_path in files:
        total_docs += index_file(file_path, store)

    return total_docs


def main():
    source_folder = os.path.abspath(FOLDER_PATH)

    # Ensure folder exists
    os.makedirs(source_folder, exist_ok=True)

    logging.info(f"Initializing vector store...")
    store = ChromaVectorStore()

    logging.info(f"Performing initial indexing of: {source_folder}")
    total_docs = initial_index(source_folder, store)
    logging.info(f"Initial indexing complete. Total documents: {total_docs}")

    # Set up file watcher
    event_handler = FileWatchHandler(store)
    observer = Observer()
    observer.schedule(event_handler, source_folder, recursive=False)
    observer.start()

    logging.info(f"Watching folder for changes: {source_folder}")
    logging.info("Press Ctrl+C to stop...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Stopping file watcher...")
        observer.stop()

    observer.join()
    logging.info("Indexer stopped.")


if __name__ == "__main__":
    main()
