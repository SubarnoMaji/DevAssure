import os
import csv
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urlparse

import PyPDF2
from docx import Document
from PIL import Image
import pytesseract
import requests
from bs4 import BeautifulSoup


class BaseParser(ABC):
    """Base class for all parsers."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def parse(self, source: str) -> List[Dict[str, str]]:
        """
        Parse the source and return a list of document dictionaries.
        
        Args:
            source: Path to file or URL string
            
        Returns:
            List of dicts with 'content' and optionally 'metadata' keys
        """
        pass
    
    def _create_document(self, content: str, metadata: Optional[Dict] = None) -> Dict[str, str]:
        """Helper method to create a standardized document dict."""
        doc = {"content": content}
        if metadata:
            doc["metadata"] = metadata
        return doc


class TextParser(BaseParser):
    """Parser for plain text files."""
    
    def parse(self, source: str) -> List[Dict[str, str]]:
        """
        Parse a plain text file.
        
        Args:
            source: Path to the text file
            
        Returns:
            List containing a single document dict
        """
        if not os.path.exists(source):
            raise FileNotFoundError(f"Text file not found: {source}")
        
        try:
            with open(source, 'r', encoding='utf-8') as f:
                content = f.read()
            
            metadata = {
                "source": source,
                "type": "text",
                "filename": os.path.basename(source)
            }
            
            return [self._create_document(content, metadata)]
        except Exception as e:
            self.logger.error(f"Error parsing text file {source}: {e}")
            raise


class PDFParser(BaseParser):
    """Parser for PDF files."""
    
    def __init__(self):
        super().__init__()
    
    def parse(self, source: str) -> List[Dict[str, str]]:
        """
        Parse a PDF file and extract text from all pages.
        
        Args:
            source: Path to the PDF file
            
        Returns:
            List of document dicts, one per page
        """
        if not os.path.exists(source):
            raise FileNotFoundError(f"PDF file not found: {source}")
        
        documents = []
        
        try:
            with open(source, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages, start=1):
                    content = page.extract_text()
                    
                    if content.strip():  # Only add non-empty pages
                        metadata = {
                            "source": source,
                            "type": "pdf",
                            "filename": os.path.basename(source),
                            "page": page_num,
                            "total_pages": total_pages
                        }
                        documents.append(self._create_document(content, metadata))
            
            self.logger.info(f"Parsed {len(documents)} pages from PDF: {source}")
            return documents
        except Exception as e:
            self.logger.error(f"Error parsing PDF {source}: {e}")
            raise


class DocxParser(BaseParser):
    """Parser for Microsoft Word (.docx) files."""
    
    def __init__(self):
        super().__init__()
    
    def parse(self, source: str) -> List[Dict[str, str]]:
        """
        Parse a DOCX file and extract text from all paragraphs.
        
        Args:
            source: Path to the DOCX file
            
        Returns:
            List containing a single document dict with all text
        """
        if not os.path.exists(source):
            raise FileNotFoundError(f"DOCX file not found: {source}")
        
        try:
            doc = Document(source)
            
            # Extract text from all paragraphs
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            content = "\n\n".join(paragraphs)
            
            # Extract text from tables if any
            table_texts = []
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells)
                    if row_text.strip():
                        table_texts.append(row_text)
            
            if table_texts:
                content += "\n\n" + "\n".join(table_texts)
            
            metadata = {
                "source": source,
                "type": "docx",
                "filename": os.path.basename(source),
                "paragraph_count": len(paragraphs),
                "table_count": len(doc.tables)
            }
            
            return [self._create_document(content, metadata)]
        except Exception as e:
            self.logger.error(f"Error parsing DOCX file {source}: {e}")
            raise


class ImageOCRParser(BaseParser):
    """Parser for images using OCR (Optical Character Recognition)."""
    
    def __init__(self, tesseract_cmd: Optional[str] = None):
        super().__init__()
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    
    def parse(self, source: str) -> List[Dict[str, str]]:
        """
        Parse an image file and extract text using OCR.
        
        Args:
            source: Path to the image file
            
        Returns:
            List containing a single document dict with extracted text
        """
        if not os.path.exists(source):
            raise FileNotFoundError(f"Image file not found: {source}")
        
        try:
            image = Image.open(source)
            
            # Perform OCR
            content = pytesseract.image_to_string(image)
            
            metadata = {
                "source": source,
                "type": "image_ocr",
                "filename": os.path.basename(source),
                "image_format": image.format,
                "image_size": f"{image.width}x{image.height}"
            }
            
            return [self._create_document(content, metadata)]
        except Exception as e:
            self.logger.error(f"Error parsing image {source}: {e}")
            raise


class CSVParser(BaseParser):
    """Parser for CSV files."""
    
    def parse(self, source: str, delimiter: str = ',', has_header: bool = True) -> List[Dict[str, str]]:
        """
        Parse a CSV file and convert rows to text documents.
        
        Args:
            source: Path to the CSV file
            delimiter: CSV delimiter character
            has_header: Whether the CSV has a header row
            
        Returns:
            List of document dicts, one per row
        """
        if not os.path.exists(source):
            raise FileNotFoundError(f"CSV file not found: {source}")
        
        documents = []
        
        try:
            with open(source, 'r', encoding='utf-8') as file:
                csv_reader = csv.DictReader(file, delimiter=delimiter) if has_header else csv.reader(file, delimiter=delimiter)
                
                for row_num, row in enumerate(csv_reader, start=1):
                    if has_header:
                        # Convert dict row to readable text
                        content = "\n".join(f"{key}: {value}" for key, value in row.items() if value)
                    else:
                        # Convert list row to text
                        content = ", ".join(str(cell) for cell in row)
                    
                    metadata = {
                        "source": source,
                        "type": "csv",
                        "filename": os.path.basename(source),
                        "row": row_num
                    }
                    documents.append(self._create_document(content, metadata))
            
            self.logger.info(f"Parsed {len(documents)} rows from CSV: {source}")
            return documents
        except Exception as e:
            self.logger.error(f"Error parsing CSV {source}: {e}")
            raise


class URLParser(BaseParser):
    """Parser for web URLs (optional)."""
    
    def __init__(self, timeout: int = 10):
        super().__init__()
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def parse(self, source: str) -> List[Dict[str, str]]:
        """
        Parse a URL and extract text content from the webpage.
        
        Args:
            source: URL string
            
        Returns:
            List containing a single document dict with webpage content
        """
        try:
            # Validate URL
            parsed = urlparse(source)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError(f"Invalid URL: {source}")
            
            # Fetch the webpage
            response = self.session.get(source, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            content = soup.get_text(separator='\n', strip=True)
            
            # Get title if available
            title = soup.find('title')
            title_text = title.get_text() if title else None
            
            metadata = {
                "source": source,
                "type": "url",
                "url": source,
                "title": title_text,
                "status_code": response.status_code
            }
            
            return [self._create_document(content, metadata)]
        except Exception as e:
            self.logger.error(f"Error parsing URL {source}: {e}")
            raise

