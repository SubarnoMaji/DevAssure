import os
import logging
from pathlib import Path
from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .parser import (
    TextParser,
    PDFParser,
    DocxParser,
    ImageOCRParser,
    CSVParser,
    URLParser
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class FileChunker:
    PARSER_MAP = {
        '.txt': TextParser,
        '.pdf': PDFParser,
        '.docx': DocxParser,
        '.doc': DocxParser,
        '.jpg': ImageOCRParser,
        '.jpeg': ImageOCRParser,
        '.png': ImageOCRParser,
        '.gif': ImageOCRParser,
        '.bmp': ImageOCRParser,
        '.tiff': ImageOCRParser,
        '.csv': CSVParser,
    }

    def __init__(
        self,
        data_folder: str = "datafolder",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        self.data_folder = Path(data_folder)
        if not self.data_folder.exists():
            logger.warning(f"Data folder '{data_folder}' does not exist. Creating it...")
            self.data_folder.mkdir(parents=True, exist_ok=True)

        self.parsers = {}
        self._initialize_parsers()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )

    def _initialize_parsers(self):
        for ext, parser_class in self.PARSER_MAP.items():
            try:
                self.parsers[ext] = parser_class()
            except ImportError as e:
                logger.warning(f"Parser for {ext} not available: {e}")
                self.parsers[ext] = None

    def _get_file_extension(self, file_path: Path) -> str:
        return file_path.suffix.lower()

    def _get_parser(self, file_path: Path) -> Optional[object]:
        ext = self._get_file_extension(file_path)
        return self.parsers.get(ext)

    def _is_supported_file(self, file_path: Path) -> bool:
        ext = self._get_file_extension(file_path)
        return ext in self.PARSER_MAP and self.parsers.get(ext) is not None

    def _chunk_document(self, doc: Dict[str, str], file_path: Path) -> List[Dict[str, str]]:
        content = doc.get('content', '')
        if not content.strip():
            return []

        chunks = self.text_splitter.split_text(content)

        original_metadata = doc.get('metadata', {})
        file_type = original_metadata.get('type', self._get_file_extension(file_path).lstrip('.'))

        chunked_docs = []
        for chunk_text in chunks:
            chunk_metadata = {
                'chunk_size': len(chunk_text),
                'filename': file_path.name,
                'type': file_type
            }

            chunked_docs.append({
                'content': chunk_text,
                'metadata': chunk_metadata
            })

        return chunked_docs

    def parse_file(self, file_path: Path) -> List[Dict[str, str]]:
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []

        parser = self._get_parser(file_path)
        if not parser:
            ext = self._get_file_extension(file_path)
            logger.warning(f"No parser available for file type: {ext} ({file_path.name})")
            return []

        try:
            logger.info(f"Parsing file: {file_path.name}")
            documents = parser.parse(str(file_path))
            logger.info(f"Successfully parsed {len(documents)} document(s) from {file_path.name}")

            all_chunks = []
            for doc in documents:
                chunks = self._chunk_document(doc, file_path)
                all_chunks.extend(chunks)

            logger.info(f"Split into {len(all_chunks)} chunk(s) from {file_path.name}")
            return all_chunks
        except Exception as e:
            logger.error(f"Error parsing {file_path.name}: {e}")
            return []

    def parse_folder(self) -> Dict[str, List[Dict[str, str]]]:

        results = {}

        if not self.data_folder.exists():
            logger.error(f"Data folder does not exist: {self.data_folder}")
            return results

        logger.info(f"Scanning data folder: {self.data_folder}")

        files = [f for f in self.data_folder.iterdir() if f.is_file()]

        if not files:
            logger.warning(f"No files found in {self.data_folder}")
            return results

        logger.info(f"Found {len(files)} file(s) in data folder")

        for file_path in files:
            if self._is_supported_file(file_path):
                documents = self.parse_file(file_path)
                if documents:
                    results[file_path.name] = documents
            else:
                ext = self._get_file_extension(file_path)
                logger.info(f"Skipping unsupported file type: {file_path.name} (extension: {ext})")

        return results


def main():
    chunker = FileChunker(data_folder="datafolder")
    results = chunker.parse_folder()
    print(results)


if __name__ == "__main__":
    main()
