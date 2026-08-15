import io

from fastapi import HTTPException, UploadFile
from pypdf import PdfReader


MAX_CV_FILE_SIZE = 10 * 1024 * 1024

ALLOWED_PDF_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
}


async def read_and_validate_pdf(cv: UploadFile) -> bytes:
    """
    Validate the uploaded CV and return its raw bytes.

    Validation layers:
    1. File extension
    2. Content-Type
    3. Non-empty file
    4. Maximum file size
    5. Actual PDF parsing happens in extract_pdf_text()
    """

    filename = (cv.filename or "").strip().lower()

    if not filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF CV files are supported.",
        )

    content_type = (cv.content_type or "").strip().lower()

    if (
        content_type
        and content_type not in ALLOWED_PDF_CONTENT_TYPES
    ):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not recognized as a PDF.",
        )

    file_bytes = await cv.read()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded CV is empty.",
        )

    if len(file_bytes) > MAX_CV_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="CV file exceeds 10 MB.",
        )

    return file_bytes


def extract_pdf_text(file_bytes: bytes) -> str:
    """
    Extract readable text from a PDF.

    PdfReader also acts as the final validation layer:
    a renamed non-PDF file will fail here.
    """

    try:
        reader = PdfReader(
            io.BytesIO(file_bytes)
        )

        pages = []

        for page in reader.pages:
            text = page.extract_text()

            if text:
                pages.append(text)

        return "\n".join(pages).strip()

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not read the PDF file: "
                f"{type(error).__name__}"
            ),
        ) from error


def get_interrupt_payload(result: dict):
    """
    Extract the Human-in-the-Loop payload returned
    by LangGraph interrupt().
    """

    interrupts = result.get(
        "__interrupt__",
        [],
    )

    if not interrupts:
        return None

    current_interrupt = interrupts[0]

    return getattr(
        current_interrupt,
        "value",
        current_interrupt,
    )
