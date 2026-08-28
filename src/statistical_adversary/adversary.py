from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from google import genai
    from google.genai import types as genai_types

    _GENAI_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - phụ thuộc môi trường cài đặt
    genai = None  # type: ignore
    genai_types = None  # type: ignore
    _GENAI_IMPORT_ERROR = exc

_FALLBACK_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
]

# Ngân sách token MẶC ĐỊNH cho câu trả lời.
# Xem thêm: https://github.com/googleapis/python-genai/issues/2062
_DEFAULT_MAX_OUTPUT_TOKENS = 2048
_RETRY_MAX_OUTPUT_TOKENS = 4096

_THINKING_LEVEL_MODEL_PATTERN = re.compile(r"^gemini-3")

def _load_system_instruction() -> str:
    file_path = Path(__file__).parent / "prompts" / "system_instruction.md"
    return file_path.read_text(encoding="utf-8")


SYSTEM_INSTRUCTION = _load_system_instruction()

OPENING_USER_PROMPT_TEMPLATE = """\
Analyze this simulation:

Name: {ten}
Method: {mo_ta}
Params: {tham_so}
Results: {ket_qua}

Provide your critiques and a challenge question.
"""

class AdversaryError(RuntimeError):
    """Lỗi khi gọi Gemini API hoặc khi module chưa được cấu hình."""

def _get_api_key() -> str | None:
    # Ưu tiên st.secrets (deploy trên Streamlit Cloud), fallback về biến môi trường (chạy local / Docker)
    try:
        import streamlit as st  # import trễ để module này dùng được ngoài Streamlit

        if hasattr(st, "secrets"):
            try:
                if "GEMINI_API_KEY" in st.secrets:
                    return str(st.secrets["GEMINI_API_KEY"])
            except Exception:
                pass
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")

def is_configured() -> bool:
    return _GENAI_IMPORT_ERROR is None and bool(_get_api_key())

def configuration_hint() -> str:
    if _GENAI_IMPORT_ERROR is not None:
        return (
            "Thiếu thư viện `google-genai`. Chạy `pip install google-genai` "
            "(đã có trong requirements.txt/requirements-web.txt)."
        )
    return (
        "Chưa tìm thấy `GEMINI_API_KEY`. Thêm vào "
        "`.streamlit/secrets.toml` (local, xem "
        "`.streamlit/secrets.toml.example`) hoặc mục **Settings > Secrets** "
        "trên Streamlit Community Cloud."
    )

def _get_client():
    if _GENAI_IMPORT_ERROR is not None:
        raise AdversaryError(configuration_hint()) from _GENAI_IMPORT_ERROR
    api_key = _get_api_key()
    if not api_key:
        raise AdversaryError(configuration_hint())
    return genai.Client(api_key=api_key)

def _get_model_candidates(preferred: str | None) -> list[str]:
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)

    override = None
    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            try:
                if "GEMINI_MODEL" in st.secrets:
                    override = str(st.secrets["GEMINI_MODEL"])
            except Exception:
                pass
    except Exception:
        pass
    if not override:
        override = os.environ.get("GEMINI_MODEL")
    if override:
        candidates.append(override)

    for m in _FALLBACK_MODELS:
        candidates.append(m)

    # loại trùng, giữ thứ tự ưu tiên
    seen: set[str] = set()
    ordered: list[str] = []
    for m in candidates:
        if m and m not in seen:
            seen.add(m)
            ordered.append(m)
    return ordered

def _is_model_not_found_error(exc: Exception) -> bool:
    text = str(exc)
    return "404" in text or "NOT_FOUND" in text or "is no longer available" in text

def _finish_reason_is_max_tokens(response) -> bool:
    try:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return False
        reason = candidates[0].finish_reason
        return "MAX_TOKENS" in str(reason).upper()
    except Exception:
        return False

def _build_config(base_kwargs: dict, model: str, max_output_tokens: int):
    kwargs = dict(base_kwargs)
    kwargs["max_output_tokens"] = max_output_tokens
    if _THINKING_LEVEL_MODEL_PATTERN.match(model):
        kwargs["thinking_config"] = genai_types.ThinkingConfig(thinking_level="low")
    return genai_types.GenerateContentConfig(**kwargs)

def _generate_with_fallback(client, contents, base_config_kwargs: dict, preferred_model: str | None):
    candidates = _get_model_candidates(preferred_model)
    base_max_tokens = base_config_kwargs.get("max_output_tokens", _DEFAULT_MAX_OUTPUT_TOKENS)
    last_exc: Exception | None = None

    for model in candidates:
        for max_output_tokens in (base_max_tokens, _RETRY_MAX_OUTPUT_TOKENS):
            config = _build_config(base_config_kwargs, model, max_output_tokens)
            try:
                response = client.models.generate_content(model=model, contents=contents, config=config)
            except Exception as exc:  # noqa: BLE001 - cần bắt rộng để fallback
                last_exc = exc
                if _is_model_not_found_error(exc):
                    break  # model này không tồn tại -> bỏ qua, sang model kế tiếp luôn
                raise AdversaryError(f"Gọi Gemini API thất bại: {exc}") from exc

            if _finish_reason_is_max_tokens(response):
                last_exc = AdversaryError(
                    f"Model {model} bị cắt ngang (hết ngân sách token) ở "
                    f"max_output_tokens={max_output_tokens}."
                )
                continue  # thử lại model NÀY với ngân sách lớn hơn

            return response

    raise AdversaryError(
        f"Gọi Gemini API thất bại: đã thử {len(candidates)} model "
        f"({', '.join(candidates)}) nhưng đều không khả dụng hoặc liên tục "
        f"bị cắt ngang do hết ngân sách token. Lỗi cuối cùng: {last_exc}"
    ) from last_exc


def _to_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)

@dataclass
class AdversaryTurn:
    role: str  # "adversary" hoặc "user"
    text: str


def critique(
    tran_phap_name: str,
    mo_ta: str,
    params: dict,
    ket_qua: dict,
    *,
    model: str | None = None,
) -> str:
    client = _get_client()
    prompt = OPENING_USER_PROMPT_TEMPLATE.format(
        ten=tran_phap_name,
        mo_ta=mo_ta,
        tham_so=_to_json(params),
        ket_qua=_to_json(ket_qua),
    )
    base_config_kwargs = {
        "system_instruction": SYSTEM_INSTRUCTION,
        "max_output_tokens": _DEFAULT_MAX_OUTPUT_TOKENS,
    }
    response = _generate_with_fallback(client, prompt, base_config_kwargs, model)

    text = getattr(response, "text", None)
    if not text:
        raise AdversaryError("Gemini API trả về phản hồi rỗng.")
    return text


def respond_to_defense(
    history: list[dict],
    defense_text: str,
    *,
    model: str | None = None,
) -> str:
    client = _get_client()

    contents = []
    for turn in history:
        role = "model" if turn["role"] == "adversary" else "user"
        contents.append(
            genai_types.Content(role=role, parts=[genai_types.Part(text=turn["text"])])
        )
    contents.append(
        genai_types.Content(role="user", parts=[genai_types.Part(text=defense_text)])
    )

    base_config_kwargs = {
        "system_instruction": SYSTEM_INSTRUCTION,
        "max_output_tokens": _DEFAULT_MAX_OUTPUT_TOKENS,
    }
    response = _generate_with_fallback(client, contents, base_config_kwargs, model)

    text = getattr(response, "text", None)
    if not text:
        raise AdversaryError("Gemini API trả về phản hồi rỗng.")
    return text
