from pathlib import Path

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MENU_PDF = PROJECT_ROOT / "data" / "B_menu_trattoria_nonna_rosa.pdf"
ADDITIONAL_INFO = PROJECT_ROOT / "data" / "informazione_aggiuntive.txt"


def load_documents(
    pdf_path: Path = MENU_PDF,
    additional_info_path: Path = ADDITIONAL_INFO,
) -> list[str]:
    """Load menu pages and supplementary information as searchable documents."""
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Menu PDF not found: {pdf_path}")
    if not additional_info_path.is_file():
        raise FileNotFoundError(
            f"Additional information file not found: {additional_info_path}"
        )

    reader = PdfReader(pdf_path)
    documents = [
        text.strip()
        for page in reader.pages
        if (text := page.extract_text())
        and text.strip()
    ]
    if not documents:
        raise ValueError(f"No extractable text found in menu PDF: {pdf_path}")

    additional_text = additional_info_path.read_text(encoding="utf-8").strip()
    if not additional_text:
        raise ValueError(
            f"Additional information file is empty: {additional_info_path}"
        )

    documents.append(additional_text)
    return documents