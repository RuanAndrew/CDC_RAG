"""Extração auditável da Lei 8.078/1990 (CDC) a partir de PDFs.

O extrator identifica a seção correspondente à Lei 8.078 em compilações que também
contêm decretos e portarias. Se não reconhecer uma seção formal da lei, mantém o PDF
inteiro para preservar compatibilidade com PDFs que contêm somente o CDC.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any

MAX_ARTICLE_GAP = 5
FIRST_ARTICLE_MAX = 5
EDGE_LINES = 3

ARTICLE_RE = re.compile(
    r"^\s*Art(?:igo)?\.?\s*(?P<num>\d{1,3})"
    r"(?:-(?P<suffix>(?-i:[A-Z])))?\s*(?:[º°]|o\b)?"
    r"(?=[\s.\-–—]|$)(?!\s*,)"
    r"(?!\s*(?:d[aoe]s?|dest[ae]|dess[ae]|c/c|caput)\b)",
    re.IGNORECASE,
)
HEADING_RE = re.compile(
    r"^\s*(?P<kind>T[IÍ]TULO|T[ií]tulo|CAP[IÍ]TULO|Cap[ií]tulo|SE[CÇ][ÃA]O|Se[cç][ãa]o"
    r"|SUBSE[CÇ][ÃA]O|Subse[cç][ãa]o)\s+(?P<num>[IVXLCDM]+|[ÚU]NICO|[Úú]nico)\b"
)
STRUCT_RE = re.compile(
    r"^\s*(?:§|Par[aá]grafo\s+[úu]nico|[IVXLCDM]+\s*[-–—]|[a-z]\))", re.I
)
NOISE_RE = re.compile(
    r"^(?:Este texto n[ãa]o substitui.*|Presid[êe]ncia da Rep[úu]blica|Casa Civil"
    r"|Subchefia para Assuntos Jur[ií]dicos)$", re.I
)
# Não confundir a data editorial "Brasília, 2013" com a assinatura da Lei.
SIGNATURE_RE = re.compile(
    r"^\s*Bras[ií]lia,\s*11\s+de\s+setembro\s+de\s+1990\b", re.I
)
LAW_NUMBER_RE = re.compile(r"\bLEI\s*N[º°O]?\s*8[.,]?\s*078\b", re.I)
PRESIDENT_RE = re.compile(r"\bO\s+PRESIDENTE\s+DA\s+REP[ÚU]BLICA\b", re.I)
HYPHEN_END_RE = re.compile(r"[a-zà-ÿ]-$")
REVOKED_RE = re.compile(r"\((?:Revogad|Vetad)", re.I)
LABEL_STRIP_RE = re.compile(
    r"^\s*(?:Art\.?\s*\d{1,3}(?:-[A-Z])?\s*[º°o]?|§\s*\d+\s*[º°o]?|"
    r"Par[aá]grafo\s+[úu]nico|[IVXLCDM]+\s*[-–—]|[a-z]\))\s*[.\-–—:]?\s*",
    re.I,
)
HEADING_LEVELS = {"TITULO": 1, "CAPITULO": 2, "SECAO": 3, "SUBSECAO": 4}
HEADING_FIELDS = {1: "titulo", 2: "capitulo", 3: "secao", 4: "subsecao"}


def normalize_line(raw_line: str) -> str:
    """Normaliza unicode e espaços, preservando pontuação e acentos."""
    line = unicodedata.normalize("NFC", raw_line).replace("\u00ad", "")
    return re.sub(r"\s+", " ", line).strip()


def _plain(text: str) -> str:
    value = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in value if not unicodedata.combining(ch))


def extract_pdf_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """Extrai texto selecionável de um PDF, preservando a página física de origem."""
    import pymupdf

    pages: list[tuple[int, str]] = []
    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
            if pdf.is_encrypted:
                raise ValueError("O PDF está protegido por senha. Envie uma versão desbloqueada.")
            for page_number, page in enumerate(pdf, start=1):
                pages.append((page_number, page.get_text("text", sort=True)))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Não foi possível ler o PDF: {exc}") from exc

    total_chars = sum(len(text.strip()) for _, text in pages)
    if total_chars < 300:
        raise ValueError(
            "Pouco ou nenhum texto foi extraído. O PDF pode estar digitalizado como imagem. "
            "Use um PDF pesquisável ou faça OCR antes da ingestão."
        )
    return pages


def detect_cdc_page_range(pages: list[tuple[int, str]]) -> tuple[int, int] | None:
    """Detecta o intervalo físico da Lei 8.078 dentro de uma compilação.

    A capa pode conter o ano editorial (por exemplo, 'Brasília, 2013'). Por isso,
    o encerramento exige a assinatura da Lei em 11/09/1990, depois do art. 119.
    Retorna (página inicial, página final), ambas inclusivas, ou None se não reconhecer.
    """
    if not pages:
        return None

    page_lines: list[list[str]] = []
    for _, raw in pages:
        page_lines.append([normalize_line(x) for x in raw.splitlines() if normalize_line(x)])

    start_idx: int | None = None
    for i, lines in enumerate(page_lines):
        blob = " ".join(lines)
        if not (LAW_NUMBER_RE.search(blob) and PRESIDENT_RE.search(blob)):
            continue
        # Exigir que a página do ato formal também contenha o começo do articulado,
        # evitando confundir o sumário (que igualmente menciona a Lei 8.078).
        has_art1 = any(
            (m := ARTICLE_RE.match(line)) is not None and int(m.group("num")) == 1
            for line in lines
        )
        if has_art1:
            start_idx = i
            break

    if start_idx is None:
        # PDFs oficiais podem separar o cabeçalho e o art. 1º por uma quebra de página.
        for i, lines in enumerate(page_lines):
            if not (LAW_NUMBER_RE.search(" ".join(lines)) and PRESIDENT_RE.search(" ".join(lines))):
                continue
            for j in range(i, min(i + 2, len(page_lines))):
                if any(
                    (m := ARTICLE_RE.match(line)) is not None and int(m.group("num")) == 1
                    for line in page_lines[j]
                ):
                    start_idx = i
                    break
            if start_idx is not None:
                break

    if start_idx is None:
        return None

    saw_article_119 = False
    end_idx: int | None = None
    for i in range(start_idx, len(page_lines)):
        for line in page_lines[i]:
            match = ARTICLE_RE.match(line)
            if match and int(match.group("num")) == 119:
                saw_article_119 = True
            if saw_article_119 and SIGNATURE_RE.match(line):
                end_idx = i
                break
        if end_idx is not None:
            break

    if end_idx is None:
        # Fallback conservador: termina após a página do último art. 119 reconhecido.
        last_119 = None
        for i in range(start_idx, len(page_lines)):
            if any(
                (m := ARTICLE_RE.match(line)) is not None and int(m.group("num")) == 119
                for line in page_lines[i]
            ):
                last_119 = i
        if last_119 is not None:
            end_idx = last_119

    if end_idx is None or end_idx < start_idx:
        return None
    return pages[start_idx][0], pages[end_idx][0]


def clean_pages(pages: list[tuple[int, str]]) -> list[tuple[str, int]]:
    """Remove ruído repetido de bordas e recompõe palavras hifenizadas entre linhas."""
    per_page: list[tuple[int, list[str]]] = []
    for page_number, text in pages:
        lines = [line for raw in text.splitlines() if (line := normalize_line(raw))]
        per_page.append((page_number, lines))

    threshold = max(3, math.ceil(0.3 * len(per_page)))
    counts: Counter[str] = Counter()
    for _, lines in per_page:
        edge = lines[:EDGE_LINES] + lines[-EDGE_LINES:]
        counts.update({_plain(re.sub(r"\d+", "#", line.lower())) for line in edge if not _is_structural(line)})
    repeated = {key for key, count in counts.items() if count >= threshold}

    kept: list[tuple[str, int]] = []
    for page_number, lines in per_page:
        for index, line in enumerate(lines):
            on_edge = index < EDGE_LINES or index >= len(lines) - EDGE_LINES
            key = _plain(re.sub(r"\d+", "#", line.lower()))
            if NOISE_RE.match(line):
                continue
            if on_edge and key in repeated:
                continue
            kept.append((line, page_number))
    return _dehyphenate(kept)


def _is_structural(line: str) -> bool:
    return bool(ARTICLE_RE.match(line) or HEADING_RE.match(line) or STRUCT_RE.match(line))


def _dehyphenate(lines: list[tuple[str, int]]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    i = 0
    while i < len(lines):
        text, page = lines[i]
        while (
            HYPHEN_END_RE.search(text)
            and i + 1 < len(lines)
            and re.match(r"[a-zà-ÿ]", lines[i + 1][0])
            and not _is_structural(lines[i + 1][0])
        ):
            text = text[:-1] + lines[i + 1][0]
            i += 1
        out.append((text, page))
        i += 1
    return out


def detect_status(text: str) -> str:
    """Detecta status quando a marca aparece imediatamente após o rótulo da unidade."""
    body = LABEL_STRIP_RE.sub("", text.strip(), count=1).lstrip(" .-–—:")
    lower = body.lower()
    if re.search(r"\(\s*revogad", lower):
        return "revogado"
    if re.search(r"\(\s*vetad", lower):
        return "vetado"
    return "vigente"


def _heading_level(kind: str) -> int:
    plain = _plain(kind).upper()
    return HEADING_LEVELS[plain]


def _valid_next(num: int, suffix: str, last_num: int | None, last_suffix: str) -> bool:
    if last_num is None:
        return num <= FIRST_ARTICLE_MAX
    if num == last_num:
        return bool(suffix) and suffix > last_suffix
    return last_num < num <= last_num + MAX_ARTICLE_GAP


def _accepted_article_lines(lines: list[tuple[str, int]]) -> set[int]:
    candidates: list[tuple[int, int, str]] = []
    for index, (line, _) in enumerate(lines):
        if SIGNATURE_RE.match(line):
            break
        match = ARTICLE_RE.match(line)
        if match and not HEADING_RE.match(line):
            candidates.append((index, int(match.group("num")), match.group("suffix") or ""))

    accepted: set[int] = set()
    last_num: int | None = None
    last_suffix = ""
    for k, (index, num, suffix) in enumerate(candidates):
        ok = _valid_next(num, suffix, last_num, last_suffix)
        if not ok and last_num is not None and num > last_num and k + 1 < len(candidates):
            _, next_num, next_suffix = candidates[k + 1]
            ok = _valid_next(next_num, next_suffix, num, suffix)
        if ok:
            accepted.add(index)
            last_num, last_suffix = num, suffix
    return accepted


def _new_section(num: int, suffix: str, stack: dict[int, str]) -> dict[str, Any]:
    key = f"{num}-{suffix}" if suffix else str(num)
    return {
        "article": f"Art. {key}" if num else "Preâmbulo",
        "article_num": num,
        "article_key": key if num else "0",
        "headings": dict(stack),
        "lines": [],
    }


def _finish(section: dict[str, Any]) -> dict[str, Any]:
    lines = section["lines"]
    text = "\n".join(line for line, _ in lines)
    pages = [page for _, page in lines]
    headings = section.pop("headings")
    section.update(
        {
            "text": text,
            "heading": " > ".join(headings[level] for level in sorted(headings)),
            "page_start": min(pages),
            "page_end": max(pages),
            "status": detect_status(lines[0][0]),
            "has_revoked_parts": bool(REVOKED_RE.search(text)),
        }
    )
    for level, field in HEADING_FIELDS.items():
        section[field] = headings.get(level, "")
    return section


def extract_legal_sections(
    pages: list[tuple[int, str]],
    rejected_out: list[tuple[str, int]] | None = None,
    *,
    isolate_cdc: bool = True,
) -> list[dict[str, Any]]:
    """Extrai artigos e hierarquia legal; isola a Lei 8.078 quando estiver em compilação."""
    selected_pages = pages
    page_range = detect_cdc_page_range(pages) if isolate_cdc else None
    if page_range:
        first, last = page_range
        selected_pages = [(number, text) for number, text in pages if first <= number <= last]

    lines = clean_pages(selected_pages)
    accepted = _accepted_article_lines(lines)
    sections: list[dict[str, Any]] = []
    stack: dict[int, str] = {}
    current: dict[str, Any] | None = _new_section(0, "", stack)
    collecting_level: int | None = None
    collected = 0

    def flush() -> None:
        nonlocal current
        if current and current["lines"]:
            sections.append(_finish(current))
        current = None

    for index, (line, page) in enumerate(lines):
        if SIGNATURE_RE.match(line):
            break

        heading = HEADING_RE.match(line)
        if heading:
            flush()
            level = _heading_level(heading.group("kind"))
            for deeper in [lv for lv in stack if lv >= level]:
                del stack[deeper]
            stack[level] = line
            collecting_level, collected = level, 0
            continue

        article = ARTICLE_RE.match(line)
        if article:
            if index in accepted:
                flush()
                num = int(article.group("num"))
                current = _new_section(num, article.group("suffix") or "", stack)
                current["lines"].append((line, page))
                collecting_level = None
                continue
            if rejected_out is not None:
                rejected_out.append((line, page))

        if current is None:
            if collecting_level is not None and collected < 2 and len(line) <= 120:
                stack[collecting_level] = f"{stack[collecting_level]} {line}"
                collected += 1
            continue
        current["lines"].append((line, page))

    flush()
    return [s for s in sections if s["article_num"] > 0 or len(s["text"]) >= 20]


def sanity_report(sections: list[dict[str, Any]], expected_last: int | None = None) -> dict[str, Any]:
    arts = [s for s in sections if s["article_num"] > 0]
    nums = [s["article_num"] for s in arts]
    keys = [s["article_key"] for s in arts]
    last = expected_last or (max(nums) if nums else 0)

    def order_key(section: dict[str, Any]) -> tuple[int, str]:
        return section["article_num"], section["article_key"].partition("-")[2]

    ordered = [order_key(s) for s in arts]
    return {
        "articles_found": len(set(keys)),
        "expected_last": last,
        "missing": sorted(set(range(1, last + 1)) - set(nums)),
        "duplicates": sorted(k for k, count in Counter(keys).items() if count > 1),
        "out_of_order": [
            arts[i]["article"] for i in range(1, len(arts)) if ordered[i] < ordered[i - 1]
        ],
        "inactive": [s["article"] for s in arts if s["status"] != "vigente"],
        "with_revoked_parts": [s["article"] for s in arts if s["has_revoked_parts"]],
    }
