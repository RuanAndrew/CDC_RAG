from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from chunking import create_rag_chunks
from currency import OUTDATED_NOTE, possibly_outdated
from extract import detect_cdc_page_range, extract_legal_sections, extract_pdf_pages, sanity_report
from indexing import DEFAULT_CHROMA_DIR, index_chunks, load_active_manifest, write_jsonl_atomic
from llm import describe_citation_issues, finalize_answer, generate_answer, get_model_presets
from retrieval import Retriever, contextualize_query

APP_TITLE = "CDC em Linguagem Simples"
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
OFFICIAL_CDC_URL = (
    "https://www2.camara.leg.br/legin/fed/lei/1990/lei-8078-11-setembro-1990-365086-normaatualizada-pl.html"
)
OFFICIAL_CDC_PDF_URL = (
    "https://www2.camara.leg.br/legin/fed/lei/1990/lei-8078-11-setembro-1990-365086-normaatualizada-pl.pdf"
)
# Implantação pública: o índice é estado global em disco e a URL do provedor é livre.
LOCK_ENDPOINTS = os.getenv("LLM_LOCK_ENDPOINTS", "0") == "1"
ALLOW_INGESTION = os.getenv("CDC_ALLOW_INGESTION", "1") != "0"

st.set_page_config(page_title=APP_TITLE, page_icon="⚖️", layout="wide")


@st.cache_resource(show_spinner="Carregando o modelo de embeddings...")
def load_embedding_resources(model_name: str):
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    model = SentenceTransformer(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("O chunking jurídico exige um tokenizador rápido com offsets.")
    return model, tokenizer


@st.cache_resource(show_spinner="Carregando o recuperador...")
def load_retriever(chroma_path: str, collection_name: str, model_name: str, rerank: bool):
    import chromadb
    from sentence_transformers import CrossEncoder

    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(collection_name)
    model, _ = load_embedding_resources(model_name)
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3") if rerank else None
    return Retriever(collection, model, reranker)


def format_reference(metadata: dict[str, Any]) -> str:
    locator = metadata.get("locator") or metadata.get("article") or "Dispositivo não identificado"
    start = metadata.get("page_start", "?")
    end = metadata.get("page_end", start)
    page = f"p. {start}" if start == end else f"pp. {start}–{end}"
    return f"{locator} · {page}"


def process_pdf(
    pdf_bytes: bytes,
    file_name: str,
    source_version: str,
    source_url: str,
    embedding_model_name: str,
    confirm_unisolated_source: bool = False,
) -> dict[str, Any]:
    if len(pdf_bytes) > 100 * 1024 * 1024:
        raise ValueError("O PDF ultrapassa o limite de 100 MB.")

    pages = extract_pdf_pages(pdf_bytes)
    detected_range = detect_cdc_page_range(pages)
    if detected_range is None and not confirm_unisolated_source:
        raise ValueError(
            "Não consegui identificar com segurança o início e o fim da Lei 8.078 neste PDF. "
            "Para evitar misturar o CDC com decretos, portarias ou outras leis, a indexação foi "
            "interrompida. Confirme explicitamente que o arquivo contém somente a Lei 8.078/1990 "
            "ou envie uma versão oficial cujo intervalo possa ser reconhecido automaticamente."
        )
    # Sem intervalo detectável, só processa o documento inteiro após confirmação explícita.
    sections = extract_legal_sections(pages, isolate_cdc=detected_range is not None)
    article_sections = [section for section in sections if int(section.get("article_num", 0)) > 0]
    if not article_sections:
        raise ValueError(
            "Nenhum artigo foi identificado. Confira se o arquivo contém o texto da Lei 8.078/1990."
        )

    report = sanity_report(article_sections, expected_last=119)
    if report["duplicates"]:
        raise ValueError(f"A extração gerou artigos duplicados: {report['duplicates']}.")
    if report["out_of_order"]:
        raise ValueError(f"A extração identificou artigos fora de ordem: {report['out_of_order']}.")
    if report["missing"]:
        raise ValueError(
            "A extração não encontrou todos os artigos-base de 1 a 119. "
            f"Lacunas: {report['missing']}. Revise o PDF antes de criar o índice."
        )
    if len(article_sections) < 80:
        raise ValueError(
            f"Somente {len(article_sections)} artigos foram encontrados. "
            "O arquivo pode estar incompleto ou conter uma extração de texto defeituosa."
        )

    # Verificação estrutural: independe do que o usuário digitou como "versão".
    outdated = possibly_outdated(section["article_key"] for section in article_sections)
    effective_version = source_version + (OUTDATED_NOTE if outdated else "")

    embedding_model, tokenizer = load_embedding_resources(embedding_model_name)
    document_hash = hashlib.sha256(pdf_bytes).hexdigest()[:16]
    chunks = create_rag_chunks(
        sections,
        tokenizer,
        source_name=file_name,
        document_hash=document_hash,
        source_version=effective_version,
        source_url=source_url,
    )
    if not chunks:
        raise ValueError("O chunking não produziu trechos indexáveis.")

    data_dir = Path("data") / "processed"
    jsonl_path = data_dir / f"chunks_{document_hash}.jsonl"
    jsonl_bytes = write_jsonl_atomic(chunks, jsonl_path)
    manifest = index_chunks(
        chunks,
        embedding_model,
        model_name=embedding_model_name,
        chroma_dir=DEFAULT_CHROMA_DIR,
        source_name=file_name,
        source_version=effective_version,
        source_url=source_url,
        source_scope=(
            "automatic_cdc_range" if detected_range is not None else "manual_single_law_confirmation"
        ),
        extra_manifest={"possibly_outdated": outdated},
    )
    return {
        "manifest": manifest,
        "jsonl_bytes": jsonl_bytes,
        "jsonl_path": str(jsonl_path),
        "pages_read": len(pages),
        "cdc_page_range": detected_range,
        "article_sections": len(article_sections),
        "unique_articles": report["articles_found"],
        "chunks": len(chunks),
        "missing_articles": report["missing"],
        "inactive_articles": report["inactive"],
        "with_revoked_parts": report["with_revoked_parts"],
        "possibly_outdated": outdated,
    }


def render_sources(sources: list[dict[str, Any]]) -> None:
    with st.expander(f"Fontes recuperadas ({len(sources)})", expanded=False):
        for i, hit in enumerate(sources, start=1):
            meta = hit.get("metadata", {}) or {}
            st.markdown(f"**S{i} · {format_reference(meta)}**")
            st.caption(
                f"Status: {meta.get('status', 'desconhecido')} · "
                f"Fonte: {meta.get('source', 'não identificada')} · "
                f"Versão: {meta.get('source_version', 'não informada')} · "
                f"Origem: {hit.get('origin', 'recuperação')}"
            )
            source_url = str(meta.get("source_url", "")).strip()
            if source_url.startswith(("https://", "http://")):
                st.markdown(f"[Abrir fonte indicada]({source_url})")
            st.write(hit.get("text", ""))
            if i < len(sources):
                st.divider()


def render_chat_history() -> None:
    for message in st.session_state.get("chat_history", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("warning"):
                st.warning(message["warning"])
            if message["role"] == "assistant" and message.get("sources"):
                render_sources(message["sources"])


st.title("CDC em Linguagem Simples")
st.write(
    "Faça perguntas sobre direitos do consumidor. O sistema busca dispositivos do Código de Defesa "
    "do Consumidor e usa um modelo de linguagem para explicar os trechos encontrados."
)
st.caption(
    "Ferramenta educacional. Não substitui a leitura da legislação oficial nem orientação jurídica profissional."
)

with st.sidebar:
    st.header("Configurações")
    st.subheader("Embeddings")
    embedding_model_name = st.text_input(
        "Modelo de embeddings",
        value=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        help="Use o mesmo modelo ao indexar e consultar a base.",
    ).strip()

    st.divider()
    st.subheader("Recuperação")
    search_mode = st.selectbox("Estratégia", ["hybrid", "vector", "bm25"], index=0)
    top_k = st.slider("Quantidade de trechos", min_value=2, max_value=12, value=6)
    lookup_enabled = st.checkbox("Priorizar artigos citados explicitamente", value=True)
    glossary_enabled = st.checkbox(
        "Traduzir termos leigos para termos jurídicos", value=True,
        help="Ex.: 'negativado' → 'cadastros de inadimplentes'. Meça o efeito com evaluate.py.",
    )
    expand_enabled = st.checkbox(
        "Incluir trechos vizinhos do mesmo artigo", value=True,
        help="Evita que o modelo veja só um parágrafo isolado de um artigo longo.",
    )
    active_only = st.checkbox("Mostrar apenas trechos marcados como vigentes", value=True)
    rerank_enabled = st.checkbox("Aplicar reranking (mais lento)", value=False)
    max_distance_value = st.number_input(
        "Limiar de distância para abstenção (0 = desligado)",
        min_value=0.0, max_value=2.0, step=0.01,
        value=float(os.getenv("MAX_DISTANCE", "0") or 0),
        help="Se o trecho mais próximo estiver mais longe que isso, o app não responde. "
             "Calibre com `python evaluate.py` (eval/out_of_scope.csv).",
    )

    st.divider()
    st.subheader("Modelo de linguagem")
    model_presets = get_model_presets()
    preset_ids = list(model_presets)
    preset_labels = {key: value["label"] for key, value in model_presets.items()}
    selected_preset = st.selectbox(
        "Perfil predefinido",
        options=preset_ids,
        format_func=lambda key: preset_labels[key],
        index=0,
        help="Escolha um perfil pronto ou use a opção compatível com OpenAI para configurar outro endpoint.",
    )
    selected_config = model_presets[selected_preset]
    provider = selected_config["provider"]
    base_url_input = st.text_input(
        "URL base", value=selected_config["base_url"], key=f"llm_base_url_{selected_preset}",
        disabled=LOCK_ENDPOINTS,
        help="Bloqueada nesta implantação." if LOCK_ENDPOINTS else None,
    ).strip()
    base_url = selected_config["base_url"] if LOCK_ENDPOINTS else base_url_input
    llm_model = st.text_input(
        "Identificador do modelo", value=selected_config["model"],
        key=f"llm_model_{selected_preset}",
        help="Use o identificador exato aceito pelo provedor selecionado.",
    ).strip()
    api_key = ""
    api_key_env = selected_config.get("api_key_env", "")
    if api_key_env:
        env_key = selected_config.get("api_key", "")
        # A chave do ambiente nunca é enviada ao navegador: o campo começa vazio.
        typed_key = st.text_input(
            "Chave da API", value="", type="password", key=f"llm_api_key_{selected_preset}",
            placeholder=f"Usando {api_key_env} do ambiente" if env_key else "Cole sua chave",
            help=f"Deixe vazio para usar a variável de ambiente {api_key_env}.",
        )
        api_key = typed_key.strip() or env_key
    if provider != "ollama":
        st.warning(
            "Ao usar um provedor externo, a pergunta e os trechos legais recuperados serão enviados "
            "à API selecionada. Não inclua dados pessoais desnecessários."
        )
    temperature = st.slider(
        "Temperatura", min_value=0.0, max_value=1.0, value=0.2, step=0.1,
        disabled=(provider == "anthropic"),
        help="Controla a variação da geração. A API Messages da Anthropic não recebe esse parâmetro neste cliente.",
    )
    max_tokens = st.number_input(
        "Limite de tokens da resposta", min_value=300, max_value=8000, value=2000, step=100,
        help="Modelos de raciocínio gastam parte desse limite pensando; aumente se a resposta vier vazia.",
    )
    if st.button("Limpar conversa"):
        st.session_state["chat_history"] = []
        st.rerun()

    st.divider()
    st.caption("As chaves de API ficam somente na sessão e não são gravadas no projeto.")

# Ingestão
st.header("1. Preparar a base legal")
st.markdown(
    f"Fonte oficial recomendada: [texto atualizado da Lei 8.078/1990 na Câmara dos Deputados]({OFFICIAL_CDC_URL}) "
    f"([PDF]({OFFICIAL_CDC_PDF_URL}))."
)
st.info(
    "O PDF de 2013 incluído no projeto serve apenas para teste histórico. Ele não contém alterações "
    "posteriores, como as introduzidas pela Lei 14.181/2021. Para respostas atuais, baixe o texto "
    "atualizado no portal oficial e envie o PDF correspondente."
)
if not ALLOW_INGESTION:
    st.info("A indexação pela interface está desativada nesta implantação (CDC_ALLOW_INGESTION=0).")

uploaded_pdf = st.file_uploader("Envie o PDF do CDC", type=["pdf"], disabled=not ALLOW_INGESTION)
source_version = st.text_input(
    "Versão ou data declarada da fonte",
    value="",
    placeholder="Ex.: texto atualizado em 10/11/2026 ou edição de 2013",
    help="Informe a edição/data que aparece no documento. O sistema também verifica se o texto contém "
         "as alterações da Lei 14.181/2021, mas não verifica a vigência automaticamente.",
    disabled=not ALLOW_INGESTION,
)
source_url = st.text_input(
    "URL oficial da fonte (opcional)",
    value="",
    placeholder=OFFICIAL_CDC_URL,
    help="Informe o endereço que realmente corresponde ao PDF enviado. Não deixe um link atual associado a um PDF histórico.",
    disabled=not ALLOW_INGESTION,
)
confirm_unisolated_source = st.checkbox(
    "Se a detecção automática falhar, confirmo que este PDF contém exclusivamente a Lei nº 8.078/1990, sem outros atos normativos.",
    value=False,
    help="Só será necessário quando o extrator não reconhecer automaticamente a seção formal do CDC.",
    disabled=not ALLOW_INGESTION,
)

col_a, col_b = st.columns([1, 1])
with col_a:
    process_uploaded = st.button(
        "Indexar PDF enviado",
        type="primary",
        disabled=(not ALLOW_INGESTION) or uploaded_pdf is None or not source_version.strip()
        or not embedding_model_name,
        use_container_width=True,
    )
with col_b:
    process_sample = st.button(
        "Indexar PDF histórico incluído",
        disabled=(not ALLOW_INGESTION) or not embedding_model_name,
        use_container_width=True,
    )

if (process_uploaded or process_sample) and ALLOW_INGESTION:
    try:
        if process_uploaded and uploaded_pdf is not None:
            pdf_bytes = uploaded_pdf.getvalue()
            file_name = uploaded_pdf.name
            version = source_version.strip()
            indexed_source_url = source_url.strip()
        else:
            sample_path = Path("sample_data") / "cdc_2013_historico.pdf"
            if not sample_path.exists():
                raise FileNotFoundError("PDF de exemplo não encontrado em sample_data/.")
            pdf_bytes = sample_path.read_bytes()
            file_name = sample_path.name
            version = "2013 — edição histórica incluída no projeto"
            indexed_source_url = ""

        with st.spinner("Extraindo artigos, preservando a estrutura legal e criando embeddings..."):
            report = process_pdf(
                pdf_bytes,
                file_name,
                version,
                indexed_source_url,
                embedding_model_name,
                confirm_unisolated_source=confirm_unisolated_source,
            )
        st.session_state["last_ingestion"] = report
        st.session_state["chunks_download"] = report["jsonl_bytes"]
        st.session_state["chunks_filename"] = f"chunks_{report['manifest']['document_hash']}.jsonl"
        st.success(
            f"Índice criado com {report['chunks']} chunks, abrangendo "
            f"{report['article_sections']} dispositivos numerados."
        )
        if report["possibly_outdated"]:
            st.warning(
                "Este texto não contém os arts. 54-A e 104-A (Lei 14.181/2021): parece anterior a 2021. "
                "A base foi marcada como possivelmente desatualizada e as respostas avisarão isso."
            )
        if report["cdc_page_range"] is None:
            st.warning(
                "O arquivo foi indexado integralmente após sua confirmação manual. "
                "Revise o conteúdo extraído para garantir que não há outras normas misturadas."
            )
    except Exception as exc:
        st.error(f"Falha na ingestão: {exc}")

last_ingestion = st.session_state.get("last_ingestion")
if last_ingestion:
    report = last_ingestion
    if report["cdc_page_range"]:
        st.write(
            f"Seção da Lei 8.078 identificada nas páginas físicas "
            f"{report['cdc_page_range'][0]}–{report['cdc_page_range'][1]}; outros atos da compilação foram excluídos."
        )
    else:
        st.warning(
            "Não foi possível isolar automaticamente uma seção identificada como Lei 8.078. "
            "O documento inteiro foi processado; confira se ele contém somente o CDC."
        )
    if report["missing_articles"]:
        st.warning(
            "A sequência apresenta lacunas em artigos-base: "
            + ", ".join(map(str, report["missing_articles"]))
        )
    st.caption(
        f"Páginas lidas: {report['pages_read']} · Artigos/dispositivos: {report['article_sections']} · "
        f"Chunks: {report['chunks']} · Arquivo JSONL: {report['jsonl_path']}"
    )
    st.download_button(
        "Baixar chunks em JSONL",
        data=st.session_state["chunks_download"],
        file_name=st.session_state["chunks_filename"],
        mime="application/x-ndjson",
    )

try:
    manifest = load_active_manifest(DEFAULT_CHROMA_DIR)
except RuntimeError as exc:
    manifest = None
    st.error(f"O manifesto do índice está inválido: {exc}. Refaça a indexação antes de consultar.")
if manifest:
    st.success(
        f"Base ativa: {manifest['source_name']} · {manifest['collection_count']} chunks · "
        f"embeddings: {manifest['embedding_model']} · escopo: {manifest.get('source_scope', 'não registrado')}"
    )
    if str(manifest.get("source_url", "")).startswith(("https://", "http://")):
        st.markdown(f"[Abrir fonte legal indexada]({manifest['source_url']})")
    version_text = str(manifest.get("source_version", "")).lower()
    # Índices antigos não têm a marca estrutural; nesses cai-se na heurística textual.
    flagged_outdated = manifest.get("possibly_outdated")
    if flagged_outdated is None:
        flagged_outdated = "2013" in version_text or "históric" in version_text
    if flagged_outdated:
        st.warning(
            "A base ativa parece anterior às alterações de 2021 ou está marcada como histórica. "
            "As respostas serão limitadas à edição indexada e não devem ser interpretadas como "
            "confirmação da legislação vigente."
        )
else:
    st.warning("Ainda não existe uma base indexada. Envie um PDF atualizado na seção acima.")

# Chat RAG
st.divider()
st.header("2. Pergunte sobre seus direitos")
if not manifest:
    st.info("Depois de indexar um PDF, a caixa de perguntas ficará disponível.")
else:
    render_chat_history()
    prompt = st.chat_input("Ex.: A loja pode se recusar a cumprir uma oferta anunciada?")
    if prompt:
        st.session_state.setdefault("chat_history", [])
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Buscando os dispositivos legais relevantes..."):
                    if manifest["embedding_model"] != embedding_model_name:
                        st.warning(
                            f"O índice usa {manifest['embedding_model']}. A consulta usará esse mesmo modelo."
                        )
                    retriever = load_retriever(
                        str(DEFAULT_CHROMA_DIR),
                        manifest["collection_name"],
                        manifest["embedding_model"],
                        rerank_enabled,
                    )
                    previous_questions = [
                        message["content"]
                        for message in st.session_state["chat_history"][:-1]
                        if message.get("role") == "user"
                    ][-5:]
                    retrieval_query = contextualize_query(prompt, previous_questions)
                    hits = retriever.search(
                        retrieval_query,
                        top_k=top_k,
                        mode=search_mode,
                        lookup=lookup_enabled,
                        only_active=active_only,
                        rerank=rerank_enabled,
                        expand=expand_enabled,
                        glossary=glossary_enabled,
                        max_distance=max_distance_value or None,
                    )
                warning_text = ""
                if not hits:
                    answer = (
                        "Não encontrei trechos legais classificados como adequados para responder. "
                        "Tente reformular a pergunta, verifique se o assunto é coberto pelo CDC ou "
                        "desative temporariamente o filtro de vigência/limiar para inspecionar a base."
                    )
                else:
                    with st.spinner(f"Gerando explicação com {llm_model}..."):
                        raw_answer = generate_answer(
                            prompt,
                            hits,
                            provider=provider,
                            model=llm_model,
                            base_url=base_url,
                            api_key=api_key,
                            temperature=temperature,
                            conversation_history=previous_questions,
                            max_tokens=int(max_tokens),
                        )
                    answer, citation_report = finalize_answer(raw_answer, hits)
                    if citation_report["needs_review"]:
                        warning_text = describe_citation_issues(citation_report)
                st.markdown(answer)
                if warning_text:
                    st.warning(warning_text)
                render_sources(hits)
                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": answer,
                    "sources": hits,
                    "warning": warning_text,
                })
            except Exception as exc:
                error_message = f"Não foi possível concluir a resposta: {exc}"
                st.error(error_message)
                st.session_state["chat_history"].append({
                    "role": "assistant", "content": error_message, "sources": []
                })

st.divider()
st.caption(
    "Importante: a qualidade depende da atualização e da integridade do documento indexado, "
    "da recuperação de contexto e do modelo de linguagem escolhido. Sempre confira as citações na fonte oficial."
)
