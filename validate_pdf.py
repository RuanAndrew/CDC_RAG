"""Valida a extração da Lei 8.078 de um PDF, sem baixar modelo de embeddings."""
from __future__ import annotations

import argparse
from pathlib import Path

from extract import detect_cdc_page_range, extract_legal_sections, extract_pdf_pages, sanity_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Caminho para o PDF do CDC")
    parser.add_argument("--expected-last", type=int, default=119)
    parser.add_argument(
        "--single-law-pdf", action="store_true",
        help="Confirma que o PDF contém exclusivamente o CDC quando os marcadores formais não forem reconhecidos.",
    )
    args = parser.parse_args()

    pages = extract_pdf_pages(args.pdf.read_bytes())
    page_range = detect_cdc_page_range(pages)
    if page_range is None and not args.single_law_pdf:
        raise SystemExit(
            "Não foi possível isolar automaticamente a Lei 8.078. "
            "Se você verificou que o PDF contém somente o CDC, rode novamente com --single-law-pdf."
        )
    sections = extract_legal_sections(pages, isolate_cdc=page_range is not None)
    articles = [section for section in sections if section["article_num"] > 0]
    report = sanity_report(articles, args.expected_last)

    print(f"Arquivo: {args.pdf}")
    print(f"Páginas no PDF: {len(pages)}")
    print(f"Intervalo identificado para a Lei 8.078: {page_range or 'não reconhecido'}")
    print(f"Dispositivos extraídos: {len(articles)}")
    print(f"Artigos únicos: {report['articles_found']}")
    print(f"Lacunas em 1..{args.expected_last}: {report['missing']}")
    print(f"Duplicados: {report['duplicates']}")
    print(f"Fora de ordem: {report['out_of_order']}")
    print(f"Artigos integralmente vetados/revogados: {report['inactive']}")
    print(f"Artigos com partes marcadas: {report['with_revoked_parts']}")

    if not articles or report["missing"] or report["duplicates"] or report["out_of_order"]:
        raise SystemExit("Validação requer revisão antes de indexar.")
    print("\nValidação estrutural concluída. Revise o conteúdo e a versão jurídica da fonte.")


if __name__ == "__main__":
    main()
