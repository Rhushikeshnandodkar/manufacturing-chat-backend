"""Utilities for PDF text extraction."""
import PyPDF2
from django.core.files.uploadedfile import UploadedFile


def extract_text_from_pdf(file: UploadedFile) -> str:
    """Extract text from an uploaded PDF file."""
    text_parts = []
    try:
        file.seek(0)
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n\n".join(text_parts).strip()
    except Exception:
        return ""
