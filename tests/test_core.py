"""Testes sem dependências pesadas: pytest -q  (ou python tests/test_core.py)."""
from __future__ import annotations

import re

from chunking import create_rag_chunks, split_text_by_tokens
from evaluate import compute_metrics, parse_expected
from extract import ARTICLE_RE, HEADING_RE, extract_legal_sections, sanity_report
from retrieval import parse_article_refs, rrf


class FakeTokenizer:
    """Um token por palavra; mesmas chamadas que o tokenizador rápido do HF."""
    is_fast = True

    def encode(self, text, add_special_tokens=False):
        return [m.group() for m in re.finditer(r"\S+", text)]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        spans = [m.span() for m in re.finditer(r"\S+", text)]
        return {"input_ids": list(range(len(spans))), "offset_mapping": spans}


HEADER = ["Presidência da República", "Casa Civil", "Lei nº 8.078 - página {n}"]
FOOTER = ["Este texto não substitui o publicado no DOU", "{n}"]

BODY = [
    ["LEI Nº 8.078, DE 11 DE SETEMBRO DE 1990.", "Dispõe sobre a proteção do consumidor e dá outras providências.",
     "TÍTULO I", "Dos Direitos do Consumidor", "CAPÍTULO I", "Disposições Gerais",
     "Art. 1° O presente código estabelece normas de proteção e defesa do consumidor.",
     "Art. 2° Consumidor é toda pessoa física ou jurídica que adquire produto como destinatário final.",
     "Parágrafo único. Equipara-se a consumidor a coletividade de pessoas."],
    ["Art. 3° Fornecedor é toda pessoa física ou jurídica, pública ou privada.",
     "Art. 4° A Política Nacional das Relações de Consumo tem por objetivo o atendimento das necessidades",
     "dos consumidores, o respeito à sua dignidade, saúde e segurança, bem como o con-",
     "sumidor deve ser protegido em todas as relações.",
     "Art. 5° Para a execução da Política Nacional, contará o poder público com vários instrumentos.",
     "Art. 6° São direitos básicos do consumidor:",
     "I - a proteção da vida, saúde e segurança contra os riscos;",
     "II - a educação e divulgação sobre o consumo adequado;"],
    ["Art. 7° Os direitos previstos neste código não excluem outros.",
     "Art. 8° Os produtos e serviços colocados no mercado não acarretarão riscos.",
     "Art. 9° (Vetado).",
     "Art. 3 Lei de referência citada em outro texto, fora da sequência.",
     "Art. 18 do CDC estabelece prazos que se aplicam aqui, como explicado acima neste dispositivo.",
     "Art. 39, VI, desta lei trata de práticas abusivas na contratação."],
    ["Art. 54. Contrato de adesão é aquele cujas cláusulas foram estabelecidas unilateralmente.",
     "Art. 54-A. Dispositivos específicos sobre crédito ao consumidor e prevenção do superendividamento.",
     "Art. 54-B. No fornecimento de crédito, o fornecedor deve informar o consumidor.",
     "Parágrafo único. Dispositivo extra para teste."],
    ["Art. 55. A União, os Estados e o Distrito Federal poderão legislar.",
     "Art. 56. Revogam-se as disposições em contrário.",
     "Brasília, 11 de setembro de 1990; 169º da Independência.",
     "FERNANDO COLLOR"],
]


def make_pages():
    pages = []
    for n, body in enumerate(BODY, start=1):
        lines = [h.format(n=n) for h in HEADER] + body + [f.format(n=n) for f in FOOTER]
        pages.append((n, "\n".join(lines)))
    return pages


def test_article_regex():
    assert ARTICLE_RE.match("Art. 54-A. Texto").group("suffix") == "A"
    assert ARTICLE_RE.match("Art. 6º São direitos").group("num") == "6"
    assert ARTICLE_RE.match("Art. 7º - Os direitos").group("suffix") is None
    assert not ARTICLE_RE.match("Art. 39, VI, desta lei")
    assert not ARTICLE_RE.match("Art. 18 do CDC estabelece")
    assert HEADING_RE.match("CAPÍTULO I") and HEADING_RE.match("Seção II")
    assert not HEADING_RE.match("Parte do preço pago") and not HEADING_RE.match("capítulo I deste título")


def test_extraction():
    rejected = []
    sections = extract_legal_sections(make_pages(), rejected_out=rejected)
    keys = [s["article_key"] for s in sections]
    assert keys == ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "54", "54-A", "54-B", "55", "56"]
    by = {s["article_key"]: s for s in sections}

    # "Art. 3 Lei de..." quebra a sequência; "Art. 18 do CDC" e "Art. 39, VI" nem casam o regex
    assert len(rejected) == 1 and rejected[0][0].startswith("Art. 3 Lei")
    assert "Art. 18 do CDC" in by["9"]["text"] and "Art. 39, VI" in by["9"]["text"]
    assert (by["54"]["page_start"], by["54"]["page_end"]) == (4, 4)  # salto 9 -> 54 confirmado por 54-A

    # ruído de cabeçalho/rodapé removido
    assert all("Casa Civil" not in s["text"] and "não substitui" not in s["text"] for s in sections)
    # hifenização recomposta
    assert "consumidor deve" in by["4"]["text"] and "con-" not in by["4"]["text"]
    # hierarquia preservada (título E capítulo)
    assert by["1"]["titulo"].startswith("TÍTULO I Dos Direitos")
    assert by["1"]["capitulo"].startswith("CAPÍTULO I Disposições Gerais")
    assert by["1"]["heading"].count(">") == 1
    # vigência
    assert by["9"]["status"] == "vetado" and by["1"]["status"] == "vigente"
    # assinatura fora do texto; páginas por seção
    assert "FERNANDO" not in by["56"]["text"] and "Brasília" not in by["56"]["text"]


def test_sanity_report():
    report = sanity_report(extract_legal_sections(make_pages()), expected_last=60)
    assert 57 in report["missing"] and 1 not in report["missing"]
    assert report["duplicates"] == [] and report["out_of_order"] == []
    assert report["inactive"] == ["Art. 9"]


def test_split_uses_original_text():
    text = "Art. 5º, caput - o fornecedor (inclusive importador) responde, ainda que sem culpa."
    chunks = split_text_by_tokens(text, FakeTokenizer(), chunk_tokens=5, overlap_tokens=2)
    assert all(c in text for c in chunks)  # fatias literais, sem decode alterando espaços
    assert chunks[0].startswith("Art.") and chunks[-1].endswith("culpa.")


def long_article_pages():
    lines = ["Art. 1° O fornecedor de produtos e serviços responde nas seguintes hipóteses legais:"]
    for roman in ["I", "II", "III", "IV", "V", "VI"]:
        lines.append(f"{roman} - " + " ".join(f"palavra{roman}{i}" for i in range(30)) + ";")
    lines.append("§ 1º " + " ".join(f"par{i}" for i in range(40)) + ".")
    lines.append("a) " + " ".join(f"alinea{i}" for i in range(20)) + ";")
    lines.append("§ 2º Texto curto do segundo parágrafo.")
    return [(1, "\n".join(lines[:5])), (2, "\n".join(lines[5:]))]


def test_structure_aware_chunking():
    sections = extract_legal_sections(long_article_pages())
    chunks = create_rag_chunks(sections, FakeTokenizer(), "teste.pdf", "abc", max_tokens=90, overlap_tokens=10)
    art = [c for c in chunks if c["metadata"]["article_key"] == "1"]
    assert len(art) > 2
    caput = "O fornecedor de produtos e serviços responde"
    assert all(caput in c["text"] for c in art)                      # caput repetido em todos
    assert any(c["metadata"]["scope"] == "partial" for c in art)
    assert any("§ 1º" in c["metadata"]["paragraphs"] for c in art)
    assert any(c["metadata"]["locator"].startswith("Art. 1, § 1º") for c in art)
    assert [c["id"] for c in chunks] == sorted(c["id"] for c in chunks)
    # páginas por chunk (não por seção): o 1º chunk está só na página 1
    assert art[0]["metadata"]["page_start"] == 1
    assert max(c["metadata"]["page_end"] for c in art) == 2
    # nenhum chunk estoura muito o orçamento (prefixo + corpo)
    tok = FakeTokenizer()
    assert all(len(tok.encode(c["text"].split("Trecho legal:\n", 1)[1])) <= 100 for c in art)


def test_short_article_is_single_chunk():
    sections = extract_legal_sections(make_pages())
    chunks = create_rag_chunks(sections, FakeTokenizer(), "t.pdf", "h")
    article_count = sum(section["article_num"] > 0 for section in sections)
    assert len(chunks) == article_count  # preâmbulo não entra no índice de dispositivos
    assert all(c["metadata"]["scope"] == "article" for c in chunks)
    assert next(c for c in chunks if c["metadata"]["article_key"] == "9")["metadata"]["status"] == "vetado"


def test_retrieval_helpers():
    assert parse_article_refs("O que diz o art. 49 do CDC?") == ["49"]
    assert parse_article_refs("arts. 18 e 19") == ["18", "19"]
    assert parse_article_refs("Artigo 54-a") == ["54-A"]
    assert parse_article_refs("posso devolver a compra?") == []
    scores = rrf([["a", "b", "c"], ["b", "a", "d"]])
    assert set(scores) == {"a", "b", "c", "d"} and scores["a"] == scores["b"] > scores["c"]


def test_metrics():
    m = compute_metrics(["12", "49", "18"], parse_expected("49;18"), [1, 3])
    assert m["hit@1"] == 0.0 and m["hit@3"] == 1.0
    assert m["recall@1"] == 0.0 and m["recall@3"] == 1.0
    assert m["mrr"] == 0.5
    assert compute_metrics(["1"], {"2"}, [1])["mrr"] == 0.0


def test_metrics_deduplicate_chunks_by_article():
    # Três chunks do mesmo artigo não devem ocupar três posições do ranking de artigos.
    m = compute_metrics(["18", "18", "18", "19"], {"19"}, [1, 2])
    assert m["hit@1"] == 0.0
    assert m["hit@2"] == 1.0
    assert m["recall@2"] == 1.0
    assert m["mrr"] == 0.5


def test_parse_article_refs_with_multiple_and_suffixes():
    assert parse_article_refs("arts. 18, 19 e 20") == ["18", "19", "20"]
    assert parse_article_refs("Artigos 54-A e 54-B") == ["54-A", "54-B"]


def test_chunking_separates_revoked_unit_from_active_units():
    lines = [
        ("Art. 5º O artigo estabelece regras importantes para relações de consumo.", 1),
        ("§ 1º Texto vigente com conteúdo jurídico suficiente para formar uma unidade.", 1),
        ("§ 2º (Revogado pela Lei nº 1).", 1),
        ("§ 3º Outro texto vigente que deve permanecer recuperável e contextualizado.", 1),
    ]
    sections = extract_legal_sections([(1, "\n".join(line for line, _ in lines))])
    chunks = create_rag_chunks(
        sections, FakeTokenizer(), "teste.pdf", "status", max_tokens=14, overlap_tokens=2
    )
    statuses = [chunk["metadata"]["status"] for chunk in chunks]
    assert "revogado" in statuses
    assert "vigente" in statuses


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
