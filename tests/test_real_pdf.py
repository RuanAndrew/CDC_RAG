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


def test_actual_uploaded_pdf_isolates_only_cdc():
    pdf_path = Path(__file__).parents[1] / "sample_data" / "cdc_2013_historico.pdf"
    pages = extract_pdf_pages(pdf_path.read_bytes())
    assert len(pages) == 161
    assert detect_cdc_page_range(pages) == (13, 60)

    sections = extract_legal_sections(pages)
    articles = [section for section in sections if section["article_num"] > 0]
    keys = {section["article_key"] for section in articles}

    assert len(articles) == 120  # arts. 1 a 119 + art. 42-A
    assert "1" in keys and "119" in keys and "42-A" in keys
    assert all(13 <= section["page_start"] <= section["page_end"] <= 60 for section in articles)
    assert not any(section["article_key"] == "2" and "Decreto nº 2.181" in section["text"] for section in articles)

    report = sanity_report(articles, expected_last=119)
    assert report["missing"] == []
    assert report["duplicates"] == []


def test_real_pdf_chunk_metadata_has_version_and_article_key():
    pdf_path = Path(__file__).parents[1] / "sample_data" / "cdc_2013_historico.pdf"
    pages = extract_pdf_pages(pdf_path.read_bytes())
    sections = extract_legal_sections(pages)
    chunks = create_rag_chunks(
        sections,
        WordTokenizer(),
        "cdc_2013_historico.pdf",
        "abc123",
        source_version="2013 - histórico",
        source_url="https://example.test/cdc",
    )
    assert chunks
    assert all(chunk["metadata"]["article_key"] != "0" for chunk in chunks)
    assert all(chunk["metadata"]["source_version"] == "2013 - histórico" for chunk in chunks)
    assert all("article_key" in chunk["metadata"] for chunk in chunks)
