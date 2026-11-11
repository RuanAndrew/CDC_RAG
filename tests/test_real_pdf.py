from pathlib import Path

from extract import detect_cdc_page_range, extract_legal_sections, extract_pdf_pages, sanity_report
from chunking import create_rag_chunks


class WordTokenizer:
    is_fast = True

    def encode(self, text, add_special_tokens=False):
        import re
        return [match.group() for match in re.finditer(r"\S+", text)]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        import re
        spans = [match.span() for match in re.finditer(r"\S+", text)]
        return {"input_ids": list(range(len(spans))), "offset_mapping": spans}


def _get_sample_pdf() -> Path:
    base = Path(__file__).parents[1] / "sample_data"
    path = base / "cdc_2013_historico.pdf"
    return path if path.exists() else base / "cdc_bolso.pdf"


def test_actual_uploaded_pdf_isolates_only_cdc():
    pdf_path = _get_sample_pdf()
    pages = extract_pdf_pages(pdf_path.read_bytes())
    expected_range = (13, 60) if pdf_path.name == "cdc_2013_historico.pdf" else (14, 76)
    expected_pages = 161 if pdf_path.name == "cdc_2013_historico.pdf" else 125
    expected_articles = 120 if pdf_path.name == "cdc_2013_historico.pdf" else 131
    assert len(pages) == expected_pages
    assert detect_cdc_page_range(pages) == expected_range

    sections = extract_legal_sections(pages)
    articles = [section for section in sections if section["article_num"] > 0]
    keys = {section["article_key"] for section in articles}

    assert len(articles) == expected_articles
    assert "1" in keys and "119" in keys and "42-A" in keys
    assert all(expected_range[0] <= section["page_start"] <= section["page_end"] <= expected_range[1] for section in articles)
    assert not any(section["article_key"] == "2" and "Decreto nº 2.181" in section["text"] for section in articles)

    report = sanity_report(articles, expected_last=119)
    assert report["missing"] == []
    assert report["duplicates"] == []


def test_real_pdf_chunk_metadata_has_version_and_article_key():
    pdf_path = _get_sample_pdf()
    pages = extract_pdf_pages(pdf_path.read_bytes())
    sections = extract_legal_sections(pages)
    chunks = create_rag_chunks(
        sections,
        WordTokenizer(),
        pdf_path.name,
        "abc123",
        source_version="ediçao comemorativa 30 anos",
        source_url="https://example.test/cdc",
    )
    assert chunks
    assert all(chunk["metadata"]["article_key"] != "0" for chunk in chunks)
    assert all(chunk["metadata"]["source_version"] == "ediçao comemorativa 30 anos" for chunk in chunks)
    assert all("article_key" in chunk["metadata"] for chunk in chunks)
