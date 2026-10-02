"""Bounded attachments: recorded audio, spoken answers, and uploaded documents."""

import base64
import binascii
from pathlib import Path

from openai import OpenAI
from pypdf import PdfReader
from pypdf.errors import PyPdfError


def decode_media(value, maximum):
    if not isinstance(value, str) or not value or len(value) > 4 * ((maximum + 2) // 3):
        raise ValueError("전송할 파일이 비어 있거나 너무 큽니다.")
    try:
        data = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError("파일 형식을 확인할 수 없습니다.") from None
    if len(data) > maximum:
        raise ValueError("전송할 파일이 너무 큽니다.")
    return data


def transcribe(key, encoded, mime):
    data = decode_media(encoded, 6_000_000)
    mime = mime.split(";")[0]
    formats = {
        "audio/webm": ("webm", data.startswith(b"\x1aE\xdf\xa3")),
        "audio/ogg": ("ogg", data.startswith(b"OggS")),
        "audio/mp4": ("mp4", data[4:8] == b"ftyp"),
        "audio/wav": ("wav", data.startswith(b"RIFF") and data[8:12] == b"WAVE"),
    }
    if mime not in formats or not formats[mime][1]:
        raise ValueError("지원하지 않는 녹음 형식입니다. Chrome에서 다시 시도해 주세요.")
    with OpenAI(api_key=key, timeout=45, max_retries=1) as client:
        result = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe", language="ko",
            file=(f"voice.{formats[mime][0]}", data, mime))
    text = result.text.strip()
    if not text:
        raise ValueError("말소리를 인식하지 못했어요. 다시 말씀해 주세요.")
    if len(text) > 6000:
        raise ValueError("인식된 문장이 너무 깁니다. 나누어서 말씀해 주세요.")
    return text


def speak(key, text):
    # ponytail: 한 번에 1,200자까지 읽는다. 긴 문서 낭독이 필요하면 문단별 스트리밍으로 확장.
    with OpenAI(api_key=key, timeout=45, max_retries=1) as client:
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts", voice="cedar", input=text[:1200],
            instructions="차분하고 명료한 한국어 개인 비서 목소리로 자연스럽게 읽으세요.",
            response_format="mp3")
        return base64.b64encode(response.content).decode()


DOCUMENT_TYPES = {".txt", ".md", ".csv", ".pdf"}


def read_document(path):
    """Read an uploaded document as plain text. Returns (name, text).

    Gradio writes the upload to a temp file; this app never copies it anywhere else.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in DOCUMENT_TYPES:
        raise ValueError("txt, md, csv, pdf 파일만 읽을 수 있어요.")
    if not path.is_file() or path.stat().st_size > 5_000_000:
        raise ValueError("5MB 이하의 파일을 올려 주세요.")
    if suffix == ".pdf":
        try:
            reader = PdfReader(str(path))
            # ponytail: 50쪽까지. 더 긴 문서는 앞부분만 필요한 경우가 대부분이라 잘라 쓴다.
            pages = reader.pages[:50]
            text = "\n".join(page.extract_text() or "" for page in pages)
        except (PyPdfError, ValueError, OSError):
            raise ValueError("PDF를 읽지 못했어요. 암호가 걸렸거나 손상된 파일일 수 있어요.") from None
    else:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            raise ValueError("텍스트 파일은 UTF-8로 저장해 주세요.") from None
    text = "\n".join(line.rstrip() for line in text.splitlines() if line.strip()).strip()
    if not text:
        raise ValueError("파일에서 글자를 찾지 못했어요. 스캔한 이미지 PDF는 읽을 수 없어요.")
    # ponytail: 2만 자까지 전달. 더 긴 문서가 흔해지면 청크 분할과 검색을 붙인다.
    return path.name, text[:20000]
