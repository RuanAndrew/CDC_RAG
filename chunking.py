"""Chunking que respeita a estrutura da lei (artigo > caput/§ > inciso > alínea)."""
from __future__ import annotations

import re
from typing import Any

from extract import REVOKED_RE, detect_status

CHUNK_TOKENS = 350
CHUNK_OVERLAP = 60
OMISSION_MARK = "[...]"

PARA_RE = re.compile(r"^\s*(?P<label>§\s*\d+(?:[º°]|o\b)?|Par[aá]grafo\s+[úu]nico)", re.I)
INCISO_RE = re.compile(r"^\s*(?P<label>[IVXLCDM]+)\s*[-–—]\s*\S")
ALINEA_RE = re.compile(r"^\s*(?P<label>[a-z])\)\s*\S")


# --------------------------------------------------------------------------- tokens
def count_tokens(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def split_text_spans(
    text: str,
    tokenizer: Any,
    chunk_tokens: int = CHUNK_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP,
) -> list[tuple[int, int]]:
    """Janelas de tokens com sobreposição, devolvidas como (início, fim) no texto ORIGINAL.

    Fatiar o original (via offsets) evita o `decode`, que altera espaços ao redor da
    pontuação — inaceitável para texto que será citado como lei.
    """
    if not getattr(tokenizer, "is_fast", True):
        raise ValueError("Use um tokenizador rápido (AutoTokenizer.from_pretrained(..., use_fast=True)).")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [(a, b) for a, b in encoded["offset_mapping"] if b > a]
    if not offsets:
        return []

    overlap = min(overlap_tokens, chunk_tokens // 2)
    stride = max(1, chunk_tokens - overlap)
    spans: list[tuple[int, int]] = []
    for start in range(0, len(offsets), stride):
        end = min(start + chunk_tokens, len(offsets))
        a, b = offsets[start][0], offsets[end - 1][1]
        if text[a:b].strip():
            spans.append((a, b))
        if end == len(offsets):
            break
    return spans


def split_text_by_tokens(
    text: str,
    tokenizer: Any,
    chunk_tokens: int = CHUNK_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP,
) -> list[str]:
    """Versão compatível com a assinatura antiga: devolve só os textos."""
    return [text[a:b].strip() for a, b in split_text_spans(text, tokenizer, chunk_tokens, overlap_tokens)]


# --------------------------------------------------------------------------- estrutura
def _norm_paragraph(label: str) -> str:
    digits = re.search(r"\d+", label)
    return f"§ {digits.group()}º" if digits else "Parágrafo único"


def _unit(kind: str, paragraph: str, inciso: str, line: tuple[str, int]) -> dict[str, Any]:
    return {"kind": kind, "paragraph": paragraph, "inciso": inciso, "lines": [line]}


def _unit_text(unit: dict[str, Any]) -> str:
    return "\n".join(text for text, _ in unit["lines"])


def parse_units(lines: list[tuple[str, int]]) -> list[dict[str, Any]]:
    """Quebra o artigo em unidades: caput, parágrafos, incisos e alíneas."""
    units: list[dict[str, Any]] = []
    paragraph = inciso = ""
    for line in lines:
        text = line[0]
        if not units:
            units.append(_unit("caput", "", "", line))
            continue
        para_m, inc_m, ali_m = PARA_RE.match(text), INCISO_RE.match(text), ALINEA_RE.match(text)
        if para_m:
            paragraph, inciso = _norm_paragraph(para_m.group("label")), ""
            units.append(_unit("paragraph", paragraph, "", line))
        elif inc_m:
            inciso = inc_m.group("label")
            units.append(_unit("inciso", paragraph, inciso, line))
        elif ali_m:
            units.append(_unit("alinea", paragraph, inciso, line))
        else:
            units[-1]["lines"].append(line)  # continuação da unidade anterior
    return units


def _span_pages(lines: list[tuple[str, int]], a: int, b: int) -> tuple[int, int]:
    """Páginas das linhas que se sobrepõem ao intervalo [a, b) do texto da unidade."""
    pages, pos = [], 0
    for text, page in lines:
        start, end = pos, pos + len(text)
        if start < b and end > a:
            pages.append(page)
        pos = end + 1  # "\n"
    return (min(pages), max(pages)) if pages else (lines[0][1], lines[-1][1])


def _scopes(units: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    scopes: list[str] = []
    incisos: list[str] = []
    for unit in units:
        scope = unit["paragraph"] or "caput"
        if scope not in scopes:
            scopes.append(scope)
        if unit["inciso"] and unit["inciso"] not in incisos:
            incisos.append(unit["inciso"])
    return scopes, incisos


# --------------------------------------------------------------------------- chunking
def chunk_section(
    section: dict[str, Any],
    tokenizer: Any,
    max_tokens: int = CHUNK_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Gera os chunks de um artigo.

    1. Artigo que cabe no limite → um único chunk (o caso comum no CDC).
    2. Caso contrário, empacota unidades (caput, §, incisos) sem cortá-las e repete o
       caput (e o § pai, para incisos) no início de cada chunk que não o contenha.
    3. Unidade isolada maior que o limite → janelas de tokens com sobreposição.
    """
    lines = section["lines"]
    units = parse_units(lines)
    pages_all = (section["page_start"], section["page_end"])

    needs_inactive_split = section.get("has_revoked_parts") and section.get("status") == "vigente"
    if count_tokens(tokenizer, section["text"]) <= max_tokens and not needs_inactive_split:
        return [{"prefix": [], "body": section["text"], "units": units, "whole": True,
                 "pages": pages_all}]

    caput_text = _unit_text(units[0])
    caput_prefix = caput_text if count_tokens(tokenizer, caput_text) <= max_tokens // 2 else ""
    para_text = {
        u["paragraph"]: _unit_text(u)
        for u in units
        if u["kind"] == "paragraph" and count_tokens(tokenizer, _unit_text(u)) <= max_tokens // 3
    }

    def prefix_for(unit: dict[str, Any]) -> list[str]:
        parts: list[str] = []
        if unit is not units[0] and caput_prefix:
            parts.append(caput_prefix)
        if unit["kind"] in ("inciso", "alinea") and unit["paragraph"] in para_text:
            parts.append(para_text[unit["paragraph"]])
        return parts

    chunks: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    prefix: list[str] = []
    used = 0

    def flush() -> None:
        nonlocal current
        if current:
            body = "\n".join(_unit_text(u) for u in current)
            pages = [p for u in current for _, p in u["lines"]]
            chunks.append({"prefix": prefix, "body": body, "units": current, "whole": False,
                           "pages": (min(pages), max(pages))})
        current = []

    for unit in units:
        text = _unit_text(unit)
        size = count_tokens(tokenizer, text)
        # Não agrupe um dispositivo vigente com outro explicitamente revogado/vetado.
        if current:
            current_statuses = {detect_status(_unit_text(u)) for u in current}
            next_status = detect_status(text)
            if len(current_statuses) == 1 and next_status not in current_statuses:
                flush()
        if current and used + size > max_tokens:
            flush()
        if not current:
            prefix = prefix_for(unit)
            used = count_tokens(tokenizer, "\n".join(prefix)) if prefix else 0
            if used + size > max_tokens:  # unidade grande demais, mesmo sozinha
                room = max(64, max_tokens - used)
                for a, b in split_text_spans(text, tokenizer, room, overlap_tokens):
                    chunks.append({"prefix": prefix, "body": text[a:b].strip(), "units": [unit],
                                   "whole": False, "pages": _span_pages(unit["lines"], a, b)})
                continue
        current.append(unit)
        used += size
    flush()
    return chunks


def create_rag_chunks(
    sections: list[dict[str, Any]],
    tokenizer: Any,
    source_name: str,
    document_hash: str,
    max_tokens: int = CHUNK_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP,
    *,
    source_version: str = "não informada",
    source_url: str = "",
) -> list[dict[str, Any]]:
    """Gera chunks com IDs estáveis e metadados de citação (compatível com o app)."""
    result: list[dict[str, Any]] = []
    number = 0

    for section in sections:
        # Capa/ementa não é dispositivo legal e não deve competir na recuperação.
        if int(section.get("article_num", 0)) <= 0:
            continue
        pieces = chunk_section(section, tokenizer, max_tokens, overlap_tokens)
        for part_number, piece in enumerate(pieces, start=1):
            number += 1
            scopes, incisos = _scopes(piece["units"])
            article = section["article"]
            if piece["whole"]:
                locator = article
                status = section["status"]
                if status == "vigente" and section.get("has_revoked_parts"):
                    status = "misto"
            else:
                locator = f"{article}, {' e '.join(scopes)}"
                if len(incisos) == 1 and len(piece["units"]) == 1:
                    locator += f", inciso {incisos[0]}"
                statuses = {detect_status(_unit_text(u)) for u in piece["units"]}
                if section["status"] != "vigente":
                    status = section["status"]
                elif len(statuses) == 1:
                    status = next(iter(statuses))
                else:
                    status = "misto"

            body = piece["body"]
            shown = "\n".join(piece["prefix"] + ([OMISSION_MARK] if piece["prefix"] else []) + [body])
            reference = f"Referência: {locator}" + (f" | {section['heading']}" if section["heading"] else "")

            result.append(
                {
                    "id": f"cdc-{document_hash}-{number:05d}",
                    "text": f"{reference}\n\nTrecho legal:\n{shown}",
                    "metadata": {
                        "source": source_name,
                        "source_version": source_version,
                        "source_url": source_url,
                        "document_hash": document_hash,
                        "article": article,
                        "article_num": int(section["article_num"]),
                        "article_key": section["article_key"],
                        "locator": locator,
                        "paragraphs": ", ".join(scopes),
                        "incisos": ", ".join(incisos),
                        "heading": section["heading"],
                        "titulo": section["titulo"],
                        "capitulo": section["capitulo"],
                        "secao": section["secao"],
                        "subsecao": section["subsecao"],
                        "status": status,
                        "has_revoked_parts": bool(REVOKED_RE.search(body)),
                        "page_start": int(piece["pages"][0]),
                        "page_end": int(piece["pages"][1]),
                        "chunk_index": number,
                        "part_in_section": part_number,
                        "scope": "article" if piece["whole"] else "partial",
                    },
                }
            )
    return result
