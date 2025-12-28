import logging
import os
import time
import hashlib
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from utils.chunker import FileChunker
from utils.vector_store import ChromaVectorStore
from utils.vector_store_config import FOLDER_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SUPPORTED_EXTENSIONS = {"txt", "pdf", "docx", "jpg", "jpeg", "png", "bmp", "tiff", "csv"}

_chunker = None

def get_chunker(chunk_size=1000, chunk_overlap=200):
    """Get or create the chunker instance."""
    global _chunker
    if _chunker is None:
        _chunker = FileChunker(
            data_folder=FOLDER_PATH,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    return _chunker


def generate_doc_id(file_path, index):
    """Generate a unique document ID based on file path and chunk index."""
    file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
    return f"{os.path.basename(file_path)}_{file_hash}_{index}"


def is_supported_file(file_path):
    """Check if file has a supported extension."""
    ext = file_path.lower().split('.')[-1]
    return ext in SUPPORTED_EXTENSIONS and not os.path.basename(file_path).startswith(".")


def index_file(file_path, store, chunker=None):
    """Index a single file into the vector store using chunker."""
    if not is_supported_file(file_path):
        return 0

    try:
        # Use provided chunker or get default one
        if chunker is None:
            chunker = get_chunker()
        
        # Parse and chunk the file
        file_path_obj = Path(file_path)
        chunked_docs = chunker.parse_file(file_path_obj)

        if not chunked_docs:
            logging.warning(f"No content extracted from {file_path}")
            return 0

        # Extract contents, generate IDs, and prepare metadatas
        contents = [doc['content'] for doc in chunked_docs]
        ids = [generate_doc_id(file_path, i) for i in range(len(chunked_docs))]

        # Add source file path to metadata for tracking (chunker already provides chunk_size, filename, type)
        metadatas = []
        for doc in chunked_docs:
            meta = doc.get('metadata', {}).copy()
            meta['source'] = file_path  # Add source path for tracking/deletion
            metadatas.append(meta)

        store.add_documents(ids, contents, metadatas)
        logging.info(f"Indexed {len(chunked_docs)} chunks from: {os.path.basename(file_path)}")
        return len(chunked_docs)

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

    def __init__(self, store, chunker=None):
        self.store = store
        self.chunker = chunker
        super().__init__()

    def on_created(self, event):
        if event.is_directory:
            return
        file_path = event.src_path
        if is_supported_file(file_path):
            # Small delay to ensure file is fully written
            time.sleep(0.5)
            logging.info(f"New file detected: {os.path.basename(file_path)}")
            index_file(file_path, self.store, self.chunker)
            doc_count = self.store.get_number_of_documents()
            logging.info(f"Documents in collection: {doc_count}")

    def on_deleted(self, event):
        if event.is_directory:
            return
        file_path = event.src_path
        if is_supported_file(file_path):
            logging.info(f"File deleted: {os.path.basename(file_path)}")
            remove_file_from_index(file_path, self.store)
            doc_count = self.store.get_number_of_documents()
            logging.info(f"Documents in collection: {doc_count}")

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
            index_file(file_path, self.store, self.chunker)
            doc_count = self.store.get_number_of_documents()
            logging.info(f"Documents in collection: {doc_count}")


def initial_index(source_folder, store, chunker=None):
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
        total_docs += index_file(file_path, store, chunker)

    return total_docs


def main():
    source_folder = os.path.abspath(FOLDER_PATH)

    # Ensure folder exists
    os.makedirs(source_folder, exist_ok=True)

    logging.info(f"Initializing vector store...")
    store = ChromaVectorStore()
    
    # Initialize chunker
    logging.info(f"Initializing chunker...")
    chunker = get_chunker(chunk_size=1000, chunk_overlap=200)

    logging.info(f"Performing initial indexing of: {source_folder}")
    total_docs = initial_index(source_folder, store, chunker)
    logging.info(f"Initial indexing complete. Total documents: {total_docs}")

    # Set up file watcher
    event_handler = FileWatchHandler(store, chunker)
    observer = Observer()
    observer.schedule(event_handler, source_folder, recursive=False)
    observer.start()

    logging.info(f"Watching folder for changes: {source_folder}")
    logging.info("Press Ctrl+C to stop...")
    
    # Display initial document count
    doc_count = store.get_number_of_documents()
    logging.info(f"Current documents in collection: {doc_count}")

    try:
        last_display_time = time.time()
        display_interval = 10  # Display count every 10 seconds when idle
        
        while True:
            time.sleep(1)
            
            # Display document count periodically when idle
            current_time = time.time()
            if current_time - last_display_time >= display_interval:
                doc_count = store.get_number_of_documents()
                logging.info(f"Idle state - Documents in collection: {doc_count}")
                last_display_time = current_time
                
    except KeyboardInterrupt:
        logging.info("Stopping file watcher...")
        observer.stop()

    observer.join()
    
    # Display final document count
    final_doc_count = store.get_number_of_documents()
    logging.info(f"Final documents in collection: {final_doc_count}")
    logging.info("Indexer stopped.")


if __name__ == "__main__":
    main()
