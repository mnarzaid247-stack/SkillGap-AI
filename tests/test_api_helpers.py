import io

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from api_helpers import (
    MAX_CV_FILE_SIZE,
    extract_pdf_text,
    read_and_validate_pdf,
)


def make_upload(
    filename: str,
    content: bytes,
    content_type: str = "application/pdf",
):
    return UploadFile(
        filename=filename,
        file=io.BytesIO(content),
        headers={
            "content-type": content_type,
        },
    )


@pytest.mark.asyncio
async def test_non_pdf_extension_is_rejected():
    upload = make_upload(
        filename="cv.txt",
        content=b"test",
        content_type="text/plain",
    )

    with pytest.raises(HTTPException) as exc:
        await read_and_validate_pdf(upload)

    assert exc.value.status_code == 400
    assert (
        exc.value.detail
        == "Only PDF CV files are supported."
    )


@pytest.mark.asyncio
async def test_invalid_content_type_is_rejected():
    upload = make_upload(
        filename="cv.pdf",
        content=b"test",
        content_type="text/plain",
    )

    with pytest.raises(HTTPException) as exc:
        await read_and_validate_pdf(upload)

    assert exc.value.status_code == 400
    assert (
        exc.value.detail
        == "Uploaded file is not recognized as a PDF."
    )


@pytest.mark.asyncio
async def test_empty_pdf_is_rejected():
    upload = make_upload(
        filename="cv.pdf",
        content=b"",
    )

    with pytest.raises(HTTPException) as exc:
        await read_and_validate_pdf(upload)

    assert exc.value.status_code == 400
    assert exc.value.detail == "Uploaded CV is empty."


@pytest.mark.asyncio
async def test_oversized_pdf_is_rejected():
    upload = make_upload(
        filename="cv.pdf",
        content=b"A" * (
            MAX_CV_FILE_SIZE + 1
        ),
    )

    with pytest.raises(HTTPException) as exc:
        await read_and_validate_pdf(upload)

    assert exc.value.status_code == 400
    assert exc.value.detail == "CV file exceeds 10 MB."


@pytest.mark.asyncio
async def test_valid_pdf_metadata_passes():
    content = b"%PDF-1.4 fake content for upload validation"

    upload = make_upload(
        filename="resume.pdf",
        content=content,
    )

    result = await read_and_validate_pdf(upload)

    assert result == content


def test_fake_pdf_cannot_be_parsed():
    fake_pdf = (
        b"%PDF-1.4 this is not actually "
        b"a valid PDF document"
    )

    with pytest.raises(HTTPException) as exc:
        extract_pdf_text(fake_pdf)

    assert exc.value.status_code == 400
    assert "Could not read the PDF file" in exc.value.detail