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

def test_real_pdf_chunk_metadata_has_version_and_article_key():
    pdf_path = Path(__file__).parents[1] / "sample_data" / "cdc_bolso.pdf"
    pages = extract_pdf_pages(pdf_path.read_bytes())
    sections = extract_legal_sections(pages)
    chunks = create_rag_chunks(
        sections,
        WordTokenizer(),
        "cdc_bolso.pdf",
        "abc123",
        source_version="ediçao comemorativa 30 anos",
        source_url="https://example.test/cdc",
    )
    assert chunks
    assert all(chunk["metadata"]["article_key"] != "0" for chunk in chunks)
    assert all(chunk["metadata"]["source_version"] == "ediçao comemorativa 30 anos" for chunk in chunks)
    assert all("article_key" in chunk["metadata"] for chunk in chunks)
